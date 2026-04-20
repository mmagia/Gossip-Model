import time
import random
import threading
import grpc
from concurrent import futures
import torch
import io

import gossip_pb2
import gossip_pb2_grpc
from src.network.gossip_servicer import GossipServicer


class Node:
    def __init__(self, node_id, port, peers, ml_model):
        self.node_id = node_id
        self.port = port
        self.peers = peers
        self.model = ml_model

        self.model_lock = threading.Lock()

        # Объект сервера gRPC
        self.server = None

    def start_server(self):
        """Инициализация и запуск gRPC сервера"""
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        gossip_pb2_grpc.add_GossipNodeServicer_to_server(
            GossipServicer(self), self.server
        )

        # Слушаем на всех интерфейсах [::] на указанном порту
        address = f"[::]:{self.port}"
        self.server.add_insecure_port(address)

        self.server.start()
        print(f"[{self.node_id}] 🟢 gRPC сервер запущен на {address}")

    def stop_server(self):
        """Корректная остановка сервера"""
        if self.server:
            print(f"[{self.node_id}] 🛑 Останавливаю сервер...")
            # 0 означает "остановить немедленно"
            self.server.stop(0)

    def handle_incoming_gossip(self, peer_weights_dict, peer_accuracy):
        """Метод, который вызывает GossipServicer при получении данных"""
        with self.model_lock:
            averaged_weights = self.model.aggregate_weights(peer_weights_dict, peer_accuracy)
            return averaged_weights

    def initiate_gossip(self):
        if not self.peers:
            return

        peer = random.choice(self.peers)

        with self.model_lock:
            my_weights = self.model.get_weights()
            my_acc = self.model.current_accuracy

        buffer_out = io.BytesIO()
        torch.save(my_weights, buffer_out)
        weights_bytes = buffer_out.getvalue()

        try:
            with grpc.insecure_channel(peer) as channel:
                stub = gossip_pb2_grpc.GossipNodeStub(channel)

                # Мы формируем простое сообщение, только ID и веса
                message = gossip_pb2.WeightMessage(
                    node_id=self.node_id,
                    model_weights=weights_bytes,
                    accuracy=my_acc
                )

                response = stub.ExchangeWeights(message, timeout=5.0)

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

        try:
            print(f"[{self.node_id}] Начинаю цикл обучения...")
            while True:
                with self.model_lock:
                    loss = self.model.train_step(num_batches=1)

                current_time = time.time()
                if current_time - last_gossip_time > gossip_interval:
                    self.initiate_gossip()
                    last_gossip_time = current_time

                if self.model.global_step % 100 == 0:
                    acc, test_loss = self.model.evaluate()
                    print(f"[{self.node_id}] Step: {self.model.global_step} | Accuracy: {acc:.2f}% | Loss: {loss:.4f}")
                time.sleep(0.05)

        except KeyboardInterrupt:

            print(f"\n[Node {self.node_id}] Сигнал остановки! Считаю финальную точность на всем датасете...")
            with self.model_lock:
                final_acc, final_loss = self.model.evaluate()

            print(f"[Node {self.node_id}] ФИНАЛЬНЫЙ РЕЗУЛЬТАТ | Accuracy: {final_acc:.2f}% | Loss: {final_loss:.4f}")
            self.stop_server()
