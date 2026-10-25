import numpy as np
from nn.layers import MLP
from nn.optim import Adam
from prune.scheduler import PolynomialDecayPruningSchedule
from prune.mask import SelfPruner
from train.dataset import load_data
from train.engine import train_epoch, evaluate

def run_self_pruning(target_sparsity: float = 0.90, criterion: str = "taylor"):
    print("=" * 60)
    print(f"PART 3: Self-Pruning Network (Target Sparsity: {target_sparsity*100:.0f}%, Criterion: {criterion.upper()})")
    print("=" * 60)

    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_data(seed=42)

    model = MLP(in_dim=64, hidden_dim=64, out_dim=10)
    optimizer = Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

    steps_per_epoch = int(np.ceil(len(X_train) / 64))
    start_step = 5 * steps_per_epoch
    end_step = 25 * steps_per_epoch

    schedule = PolynomialDecayPruningSchedule(0.0, target_sparsity, start_step, end_step)
    pruner = SelfPruner([model.fc1, model.fc2, model.fc3], schedule, criterion=criterion, allow_regrowth=True)

    step_counter = 0
    for epoch in range(1, 35):
        loss, step_counter = train_epoch(model, optimizer, X_train, y_train, batch_size=64, pruner=pruner, step_counter=step_counter)
        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, X_test, y_test)
            total_zeros = sum(np.sum(l.mask == 0) for l in [model.fc1, model.fc2, model.fc3])
            total_weights = sum(l.mask.size for l in [model.fc1, model.fc2, model.fc3])
            actual_sp = total_zeros / total_weights
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Test Acc: {acc * 100:.2f}% | Active Sparsity: {actual_sp * 100:.1f}%")

    final_acc = evaluate(model, X_test, y_test)
    total_zeros = sum(np.sum(l.mask == 0) for l in [model.fc1, model.fc2, model.fc3])
    total_weights = sum(l.mask.size for l in [model.fc1, model.fc2, model.fc3])
    achieved_sparsity = total_zeros / total_weights
    print(f"\nFinal Achieved Sparsity: {achieved_sparsity * 100:.2f}% | Final Test Accuracy: {final_acc * 100:.2f}%\n")
    return achieved_sparsity, final_acc

if __name__ == "__main__":
    run_self_pruning(0.90, "taylor")
