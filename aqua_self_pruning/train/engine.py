import numpy as np
from engine.tensor import Tensor
from engine.ops import softmax_cross_entropy

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    preds = []
    batch_size = 128
    for i in range(0, len(X_test), batch_size):
        xb = Tensor(X_test[i:i+batch_size], requires_grad=False)
        out = model(xb)
        preds.append(np.argmax(out.data, axis=1))
    all_preds = np.concatenate(preds)
    return float(np.mean(all_preds == y_test))

def train_epoch(model, optimizer, X: np.ndarray, y: np.ndarray, batch_size: int = 64, pruner=None, step_counter: int = 0):
    N = len(X)
    indices = np.random.permutation(N)
    total_loss = 0.0
    num_batches = 0

    prunable_layers = [model.fc1, model.fc2, model.fc3]

    for start in range(0, N, batch_size):
        step_counter += 1
        end = min(start + batch_size, N)
        xb = Tensor(X[indices[start:end]], requires_grad=False)
        yb = y[indices[start:end]]

        model.zero_grad()
        out = model(xb)
        loss = softmax_cross_entropy(out, yb)
        loss.backward()

        optimizer.step(layers=prunable_layers)

        if pruner is not None:
            pruner.step(step_counter)

        total_loss += loss.data.item()
        num_batches += 1

    return total_loss / num_batches, step_counter
