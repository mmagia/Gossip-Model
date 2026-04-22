import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ml.trainer import DecentralizedTrainer
from ml.model import set_seed
from ml.dataset import get_dataloader
from ml.aggregator import calculate_mse


def run_full_integration_test():
    print("🚀 Starting FULL Integration Test: Gossip + Non-IID Data")
    print("-" * 50)

    # 1. Setup
    set_seed(42)
    num_nodes = 2
    batch_size = 32

    # 2. Prepare Non-IID Data
    print("📦 Loading Non-IID datasets...")
    train_loader_0 = get_dataloader(node_id=0, num_nodes=num_nodes, batch_size=batch_size, is_train=True)
    train_loader_1 = get_dataloader(node_id=1, num_nodes=num_nodes, batch_size=batch_size, is_train=True)

    # Global test set (contains all digits 0-9)
    test_loader = get_dataloader(node_id=0, num_nodes=num_nodes, batch_size=1000, is_train=False)

    # 3. Initialize Trainers
    print("\n👩‍🎓 Initializing Node 0 and Node 1...")
    node_0 = DecentralizedTrainer(node_id=0, train_loader=train_loader_0, test_loader=test_loader)
    node_1 = DecentralizedTrainer(node_id=1, train_loader=train_loader_1, test_loader=test_loader)

    # 4. Initial Evaluation
    print("\n📊 Initial Accuracy (random weights):")
    acc_0_start, _ = node_0.evaluate()
    acc_1_start, _ = node_1.evaluate()
    print(f"Node 0: {acc_0_start:.2f}% | Node 1: {acc_1_start:.2f}%")

    # 5. Local Training
    print("\n🧠 Nodes are training on their EXCLUSIVE data sharded by classes...")
    node_0.train_step(num_batches=150)
    node_1.train_step(num_batches=150)

    acc_0_trained, _ = node_0.evaluate()
    acc_1_trained, _ = node_1.evaluate()

    print(f"After Local Training:")
    print(f"Node 0 (knows 0-4): {acc_0_trained:.2f}%")
    print(f"Node 1 (knows 5-9): {acc_1_trained:.2f}%")

    # 6. Gossip Exchange
    print("\n📡 Performing Gossip Weight Aggregation...")

    # Calculate Divergence (MSE) before swap
    w0 = node_0.get_weights()
    w1 = node_1.get_weights()
    initial_mse = calculate_mse(w0, w1)
    print(f"Initial Model Divergence (MSE): {initial_mse:.6f}")

    # Получаем текущие точности для обмена
    current_acc_0 = node_0.current_accuracy
    current_acc_1 = node_1.current_accuracy

    # Node 0 receives weights from Node 1
    node_0.aggregate_weights(w1, current_acc_1)  # Исправлено: передаем peer_acc
    # Node 1 receives weights from Node 0
    node_1.aggregate_weights(w0, current_acc_0)  # Исправлено: передаем peer_acc

    # 7. Final Evaluation
    print("\n📊 Evaluation after Gossip Exchange:")
    acc_0_final, _ = node_0.evaluate()
    acc_1_final, _ = node_1.evaluate()
    print(f"Node 0: {acc_0_final:.2f}%")
    print(f"Node 1: {acc_1_final:.2f}%")

    # Post-exchange MSE
    final_mse = calculate_mse(node_0.get_weights(), node_1.get_weights())
    print(f"Final Model Divergence (MSE): {final_mse:.6f}")

    # 8. Validation
    if acc_0_final > acc_0_trained and acc_1_final > acc_1_trained:
        print("\n✅ SUCCESS: Gossip helped models to learn digits from neighbors!")
    else:
        print("\n⚠️ WARNING: Accuracy did not improve as expected. Check aggregation logic.")

    if final_mse < initial_mse:
        print("✅ SUCCESS: Models converged (MSE decreased)!")

    print("\n🔄 Запуск серии обменов для генерации истории логов...")
    for i in range(3):
        node_0.train_step(num_batches=20)
        node_1.train_step(num_batches=20)

        # Обмениваемся весами
        w0 = node_0.get_weights()
        w1 = node_1.get_weights()
        
        # Получаем текущие точности
        acc_0 = node_0.current_accuracy
        acc_1 = node_1.current_accuracy

        # Исправлено: передаем peer_acc, а не alpha
        node_0.aggregate_weights(w1, acc_1)
        node_1.aggregate_weights(w0, acc_0)

        # Записываем состояние в логи
        node_0.evaluate()
        node_1.evaluate()
        print(f"Цикл {i + 1} завершен...")


if __name__ == "__main__":
    run_full_integration_test()