import time
import json
import numpy as np
import matplotlib.pyplot as plt
from nn.layers import MLP
from nn.optim import Adam
from prune.scheduler import PolynomialDecayPruningSchedule
from prune.mask import SelfPruner
from train.dataset import load_data
from train.engine import train_epoch, evaluate

def train_and_eval(target_sparsity: float, criterion: str, seed: int):
    np.random.seed(seed)
    X_train, y_train, X_test, y_test = load_data(seed=seed)

    model = MLP(in_dim=64, hidden_dim=64, out_dim=10)
    optimizer = Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

    steps_per_epoch = int(np.ceil(len(X_train) / 64))
    start_step = 3 * steps_per_epoch
    end_step = 25 * steps_per_epoch

    schedule = PolynomialDecayPruningSchedule(0.0, target_sparsity, start_step, end_step)
    pruner = SelfPruner([model.fc1, model.fc2, model.fc3], schedule, criterion=criterion, allow_regrowth=(criterion == "taylor"))

    step_counter = 0
    for epoch in range(1, 35):
        _, step_counter = train_epoch(model, optimizer, X_train, y_train, batch_size=64, pruner=pruner, step_counter=step_counter)

    acc = evaluate(model, X_test, y_test)

    total_params = sum(l.mask.size for l in [model.fc1, model.fc2, model.fc3])
    active_params = sum(int(np.sum(l.mask)) for l in [model.fc1, model.fc2, model.fc3])
    dense_flops = 2 * (64 * 64 + 64 * 64 + 64 * 10)
    sparse_flops = 2 * active_params

    start_sparse = time.perf_counter()
    for _ in range(30):
        h1 = np.maximum(0, model.fc1.forward_sparse_inference(X_test))
        h2 = np.maximum(0, model.fc2.forward_sparse_inference(h1))
        _ = model.fc3.forward_sparse_inference(h2)
    sparse_time_ms = (time.perf_counter() - start_sparse) / 30 * 1000

    return {
        "accuracy": acc,
        "active_params": active_params,
        "total_params": total_params,
        "active_param_pct": active_params / total_params,
        "flops_reduction": (dense_flops - sparse_flops) / dense_flops,
        "sparse_time_ms": sparse_time_ms
    }

def run_pareto_sweep():
    sparsity_levels = [0.0, 0.50, 0.75, 0.90, 0.95]
    seeds = [42, 43]

    results = {"taylor": {}, "magnitude": {}}

    print("=" * 70)
    print("PART 4: PARETO SWEEP & REAL COST MEASUREMENTS")
    print("=" * 70)

    for sp in sparsity_levels:
        results["taylor"][sp] = []
        results["magnitude"][sp] = []
        for seed in seeds:
            print(f"Running Sparsity {sp*100:02.0f}% | Seed {seed}...")
            r_taylor = train_and_eval(sp, "taylor", seed)
            r_mag = train_and_eval(sp, "magnitude", seed)
            results["taylor"][sp].append(r_taylor)
            results["magnitude"][sp].append(r_mag)

    summary = {"sparsity": sparsity_levels, "taylor_mean": [], "taylor_std": [], "mag_mean": [], "mag_std": [], "flop_reduction": []}
    for sp in sparsity_levels:
        t_accs = [r["accuracy"] for r in results["taylor"][sp]]
        m_accs = [r["accuracy"] for r in results["magnitude"][sp]]
        summary["taylor_mean"].append(float(np.mean(t_accs)))
        summary["taylor_std"].append(float(np.std(t_accs)))
        summary["mag_mean"].append(float(np.mean(m_accs)))
        summary["mag_std"].append(float(np.std(m_accs)))
        summary["flop_reduction"].append(results["taylor"][sp][0]["flops_reduction"])

    print("\n" + "=" * 80)
    print(f"{'Sparsity':<10} | {'Taylor Saliency Acc':<22} | {'Magnitude Acc':<22} | {'FLOPs Cut':<10}")
    print("=" * 80)
    for i, sp in enumerate(sparsity_levels):
        tm, ts = summary["taylor_mean"][i] * 100, summary["taylor_std"][i] * 100
        mm, ms = summary["mag_mean"][i] * 100, summary["mag_std"][i] * 100
        fl = summary["flop_reduction"][i] * 100
        print(f"{sp*100:>8.1f}%  | {tm:5.2f}% (+/- {ts:4.2f}%)     | {mm:5.2f}% (+/- {ms:4.2f}%)     | {fl:>8.1f}%")
    print("=" * 80)

    with open("pareto_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[+] Raw metrics written to pareto_results.json")

    plt.figure(figsize=(9, 5))
    plt.errorbar([s * 100 for s in sparsity_levels], [m * 100 for m in summary["taylor_mean"]],
                 yerr=[s * 100 for s in summary["taylor_std"]], fmt='-o', label='First-Order Taylor Saliency (Ours)', color='#10b981', lw=2, capsize=4)
    plt.errorbar([s * 100 for s in sparsity_levels], [m * 100 for m in summary["mag_mean"]],
                 yerr=[s * 100 for s in summary["mag_std"]], fmt='--s', label='Magnitude Baseline', color='#ef4444', lw=2, capsize=4)
    plt.title("Sparsity-Accuracy Pareto Curve", fontsize=13, fontweight='bold')
    plt.xlabel("Target Sparsity (%)", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("pareto_curve.png", dpi=300)
    print("[+] Plot saved to pareto_curve.png")

if __name__ == "__main__":
    run_pareto_sweep()
