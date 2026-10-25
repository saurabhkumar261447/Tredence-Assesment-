import numpy as np
from nn.layers import MLP
from nn.optim import Adam
from train.dataset import load_data
from train.engine import train_epoch, evaluate

def run_dense_training():
    print("=" * 60)
    print("PART 2: Dense Baseline MLP Training (Digits Classification)")
    print("=" * 60)
    
    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_data(seed=42)

    model = MLP(in_dim=64, hidden_dim=64, out_dim=10)
    optimizer = Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

    step_counter = 0
    for epoch in range(1, 31):
        loss, step_counter = train_epoch(model, optimizer, X_train, y_train, batch_size=64, step_counter=step_counter)
        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, X_test, y_test)
            print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f} | Test Accuracy: {acc * 100:.2f}%")

    final_acc = evaluate(model, X_test, y_test)
    print(f"\nFinal Dense Accuracy: {final_acc * 100:.2f}%\n")
    return final_acc

if __name__ == "__main__":
    run_dense_training()
