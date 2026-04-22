import time
import random
import threading
import grpc
from concurrent import futures
import torch
import io

import gossip_pb2
import gossip_pb2_grpc
from network.gossip_servicer import GossipServicer
from config import (
    GOSSIP_JITTER_MIN,
    GOSSIP_JITTER_MAX,
    PACKET_LOSS_PROB,
    OUTGOING_DELAY_MIN,
    OUTGOING_DELAY_MAX,
    ENABLE_SLOW_NODE,
    SLOW_NODE_ID,
    SLOW_NODE_EXTRA_DELAY_MIN,
    SLOW_NODE_EXTRA_DELAY_MAX,
    GRPC_TIMEOUT,
)


class Node:
    def __init__(self, node_id, port, peers, ml_model):
        self.node_id = node_id
        self.port = port
        self.peers = peers
        self.model = ml_model

        self.model_lock = threading.Lock()

        # Объект сервера gRPC
        self.server = None

        self.gossip_attempts = 0
        self.lost_packets = 0

        self.delayed_messages = 0
        self.total_delay_time = 0.0

    def start_server(self):
        """Инициализация и запуск gRPC сервера"""
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        gossip_pb2_grpc.add_GossipNodeServicer_to_server(
            GossipServicer(self), self.server
        )

        address = f"[::]:{self.port}"
        self.server.add_insecure_port(address)

        self.server.start()
        print(f"[{self.node_id}] 🟢 gRPC сервер запущен на {address}")

    def stop_server(self):
        """Корректная остановка сервера"""
        if self.server:
            print(f"[{self.node_id}] 🛑 Останавливаю сервер...")
            self.server.stop(0)

    def handle_incoming_gossip(self, peer_weights_dict, peer_accuracy):
        """Метод, который вызывает GossipServicer при получении данных"""
        with self.model_lock:
            averaged_weights = self.model.aggregate_weights(peer_weights_dict, peer_accuracy)
            return averaged_weights

    def _get_next_gossip_interval(self, base_interval):
        """
        Добавляем небольшой случайный сдвиг интервала,
        чтобы gossip-обмены происходили не идеально синхронно.
        """
        jitter = random.uniform(GOSSIP_JITTER_MIN, GOSSIP_JITTER_MAX)
        interval = base_interval + jitter

        # Не даем интервалу стать слишком маленьким
        return max(0.5, interval)

    def _should_drop_outgoing_message(self):
        """
        Имитация периодической потери пакета на отправке.
        """
        return random.random() < PACKET_LOSS_PROB

    def _simulate_outgoing_delay(self, peer):
        """
        Имитация задержки перед отправкой gossip-запроса.
        Делаем мягкой, чтобы обучение продолжало сходиться.
        """
        delay = random.uniform(OUTGOING_DELAY_MIN, OUTGOING_DELAY_MAX)

        # Одна нода может быть "медленной"
        if ENABLE_SLOW_NODE and self.node_id == SLOW_NODE_ID:
            delay += random.uniform(
                SLOW_NODE_EXTRA_DELAY_MIN,
                SLOW_NODE_EXTRA_DELAY_MAX
            )

        if delay > 0:
            self.delayed_messages += 1
            self.total_delay_time += delay
            time.sleep(delay)

    def _get_network_stats(self):
        packet_loss_rate = (
            self.lost_packets / self.gossip_attempts * 100
            if self.gossip_attempts > 0 else 0.0
        )

        avg_delay = (
            self.total_delay_time / self.delayed_messages
            if self.delayed_messages > 0 else 0.0
        )

        return {
            "packet_loss_rate": packet_loss_rate,
            "avg_delay": avg_delay,
        }

    def initiate_gossip(self):
        """
        Исходящий gossip-обмен:
        1) выбираем случайного соседа,
        2) иногда теряем пакет,
        3) иногда ждем перед отправкой,
        4) отправляем веса,
        5) получаем усредненные веса обратно.
        """
        if not self.peers:
            return

        peer = random.choice(self.peers)
        self.gossip_attempts += 1

        if self._should_drop_outgoing_message():
            self.lost_packets += 1
            print(f"[Node {self.node_id}] 📦❌ Packet to {peer} was lost artificially")
            return

        # Искусственная задержка перед отправкой
        self._simulate_outgoing_delay(peer)

        with self.model_lock:
            my_weights = self.model.get_weights()
            my_acc = self.model.current_accuracy

        buffer_out = io.BytesIO()
        torch.save(my_weights, buffer_out)
        weights_bytes = buffer_out.getvalue()

        try:
            with grpc.insecure_channel(peer) as channel:
                stub = gossip_pb2_grpc.GossipNodeStub(channel)

                message = gossip_pb2.WeightMessage(
                    node_id=self.node_id,
                    model_weights=weights_bytes,
                    accuracy=my_acc
                )

                response = stub.ExchangeWeights(message, timeout=GRPC_TIMEOUT)

                if response.success:
                    incoming_bytes = response.averaged_weights
                    buffer_in = io.BytesIO(incoming_bytes)
                    new_weights = torch.load(buffer_in, weights_only=False)

                    with self.model_lock:
                        self.model.model.load_state_dict(new_weights)

        except grpc.RpcError as e:
            print(f"[Node {self.node_id}] Не удалось связаться с {peer}: статус {e.code().name}")

    def run(self, gossip_interval=5):
        """
        Главный жизненный цикл узла.
        Здесь совмещается обучение и периодический запуск gossip.
        """
        self.start_server()

        last_gossip_time = time.time()
        next_gossip_interval = self._get_next_gossip_interval(gossip_interval)

        try:
            print(f"[{self.node_id}] Начинаю цикл обучения...")
            while True:
                with self.model_lock:
                    loss = self.model.train_step(num_batches=1)

                current_time = time.time()

                if current_time - last_gossip_time > next_gossip_interval:
                    self.initiate_gossip()
                    last_gossip_time = current_time
                    next_gossip_interval = self._get_next_gossip_interval(gossip_interval)

                if self.model.global_step % 100 == 0:
                    acc, test_loss = self.model.evaluate()
                    net_stats = self._get_network_stats()

                    print(
                        f"[{self.node_id}] Step: {self.model.global_step} | "
                        f"Accuracy: {acc:.2f}% | Loss: {loss:.4f} | "
                        f"Net: packet_loss_rate={net_stats['packet_loss_rate']:.1f}%, "
                        f"avg_delay={net_stats['avg_delay']:.2f}s"
                    )

                time.sleep(0.05)

        except KeyboardInterrupt:
            print(f"\n[Node {self.node_id}] Сигнал остановки! Считаю финальную точность на всем датасете...")
            with self.model_lock:
                final_acc, final_loss = self.model.evaluate()

            print(f"[Node {self.node_id}] ФИНАЛЬНЫЙ РЕЗУЛЬТАТ | Accuracy: {final_acc:.2f}% | Loss: {final_loss:.4f}")
            self.stop_server()
