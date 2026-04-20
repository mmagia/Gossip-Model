import multiprocessing as mp
import time

from src.ml.trainer import DecentralizedTrainer
from src.network.node import Node
from src.ml.dataset import get_dataloader


def run_node_process(node_id: int, port: str, peers: list, num_nodes: int = 5):
    print(f"Запускается Узел {node_id} (Порт: {port}). Соседи: {peers}")

    train_loader = get_dataloader(node_id=node_id, num_nodes=num_nodes, is_train=True)
    test_loader = get_dataloader(node_id=node_id, num_nodes=num_nodes, is_train=False)

    ml_model = DecentralizedTrainer(
        node_id=node_id,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=0.01
    )

    node = Node(
        node_id=node_id,
        port=port,
        peers=peers,
        ml_model=ml_model
    )

    node.run(gossip_interval=3)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    node_configs = [
        {"id": 0, "port": "50051"},
        {"id": 1, "port": "50052"},
        {"id": 2, "port": "50053"},
        {"id": 3, "port": "50054"},
        {"id": 4, "port": "50055"},
    ]

    processes = []
    try:
        for config in node_configs:
            my_id = config["id"]
            my_port = config["port"]

            my_peers = [f"localhost:{c['port']}" for c in node_configs if c["id"] != my_id]
            p = mp.Process(
                target=run_node_process,
                args=(my_id, my_port, my_peers, len(node_configs))
            )
            p.start()
            processes.append(p)

            time.sleep(3)

        print("\nВСЕ 5 НОД ЗАПУЩЕНЫ!\n" + "-" * 50)

        for p in processes:
            p.join()


    except KeyboardInterrupt:
        print("\nПолучен сигнал остановки (Ctrl+C). Ждем, пока узлы подведут итоги...")
        for p in processes:
            p.join()
        print("Сеть успешно отключена. Все данные сохранены.")
