# Tredence-Assesment-
A reverse-mode autodiff engine and self-pruning neural network built from scratch in pure NumPy, featuring First-Order Taylor saliency, dynamic regrowth, and mask-aware optimizer state isolation.

---

## Executive Summary & System Overview

This repository provides an end-to-end, zero-framework implementation of a dynamic, self-pruning neural network driven by a custom reverse-mode automatic differentiation (autodiff) engine[cite: 2, 3]. Built exclusively on top of raw Python and **NumPy**[cite: 2], this system trains an artificial neural network that progressively determines its own sparse architectural topology during training under a hard compute budget[cite: 2, 4].

Standard deep learning frameworks (PyTorch, TensorFlow, JAX) conceal graph transformations, gradient routing, memory allocation, and momentum buffer dynamics under C++/CUDA abstractions[cite: 2]. When pruning neural networks during training, naive implementations across industry codebases introduce subtle mathematical corruptions—specifically around gradient propagation through masked parameters and momentum leakage in adaptive optimizers like Adam[cite: 2, 3]. 

This project solves those fundamental challenges from first mathematical principles[cite: 1]:
1. **Reverse-Mode Autodiff Engine:** Complete Directed Acyclic Graph (DAG) construction with reverse topological evaluation, dynamic tensor broadcasting/unbroadcasting, and numerically stable loss functions[cite: 2, 3].
2. **Honest Masked Gradients & Optimizer Synchronization:** Rigorous isolation of pruned connections, strictly zeroing mathematical gradients and eliminating optimizer momentum drift on frozen parameters[cite: 3, 4].
3. **First-Order Taylor Saliency:** Connection importance derived from Taylor series approximations of loss perturbation $\Delta \mathcal{L}$ rather than uncalibrated weight magnitude heuristics[cite: 3].
4. **Dynamic Connection Regrowth:** Bidirectional connectivity learning (RigL/SET-style) where pruned connections can resurrect if their gradient sensitivity warrants reinstatement[cite: 4].
5. **Real Compute Reduction Measurement:** A sparse inference execution path evaluating true non-zero floating-point multiply-accumulate operations (FLOPs) rather than masked dense multiplications[cite: 4].

---

## Tools & Technologies Used

This implementation strictly complies with zero-framework restrictions: **no deep learning frameworks (PyTorch, TensorFlow, JAX, Keras, Hugging Face) or automated autodiff/pruning libraries are used for any part of the core engine, layers, optimizer, or training pipeline**[cite: 2].

| Tool / Technology | Category | Purpose & Implementation Scope |
| :--- | :--- | :--- |
| **Python (3.10+)**[cite: 5, 9, 11] | Core Language | Powers the computational graph, closure-based backward functions, operator overloading, and module architecture[cite: 2, 29]. |
| **NumPy (`>=1.24.0`)**[cite: 11] | Numerical Engine | Powers raw $n$-dimensional memory buffers (`ndarray`), memory views, vectorized matrix multiplications (`matmul`), and linear algebra operations without high-level framework wrappers[cite: 2, 28]. |
| **Scikit-Learn (`>=1.2.0`)**[cite: 11] | Dataset Utility | **Permitted utility only:** Used strictly to fetch the standard 8x8 handwritten Digits benchmark (`load_digits`)[cite: 2, 9, 18]. Completely absent from models, forward passes, gradients, or optimization[cite: 2]. |
| **Matplotlib (`>=3.7.0`)**[cite: 11] | Visualization | Generates the multi-seed Sparsity-Accuracy Pareto curve with error bars (`pareto_curve.png`) comparing Taylor Saliency against Magnitude Pruning[cite: 4, 12]. |
| **Pytest (`>=7.0.0`)**[cite: 11] | Testing & Verification | Automated test runner executing finite-difference numerical gradient checks (`test_gradcheck.py`) and masked weight optimizer state isolation tests (`test_masking_optimizer.py`)[cite: 9, 15, 16]. |
| **Setuptools (`>=61.0`) / PEP 621**[cite: 11] | Packaging & Build | Standardized `pyproject.toml` configuration enabling clean dependency resolution and editable local installation (`pip install -e .`)[cite: 9, 11]. |
| **Python Standard Library (`json`, `time`, `os`)** | System & Profiling | Benchmarking wall-clock sparse inference latency, calculating true FLOP counts, and persisting raw Pareto metrics to disk (`pareto_results.json`)[cite: 12]. |

---

## Core Architecture & Deep-Dive Explanations

### 1. Reverse-Mode Autodiff Engine (`engine/`)

The core compute abstraction is built around the `Tensor` class and functional primitives defined in `engine/ops.py`[cite: 2, 9, 30]:

- **Computational Graph (DAG) Tracking:** Every intermediate operation wraps data in a `Tensor` and stores backward closures alongside directed references to its parent tensors[cite: 2, 3, 29].
- **Topological Sort via Depth-First Search (DFS):** Reverse-mode backpropagation requires executing backward operations in the exact reverse order of dependency[cite: 2, 29]. When `.backward()` is called on a scalar loss, the engine executes a recursive post-order traversal to build an ordered linear tape of dependencies before invoking local vector-Jacobian products (VJPs)[cite: 2, 29].
- **Exact Gradient Unbroadcasting:** Binary operations frequently apply NumPy broadcasting (e.g., adding a bias vector of shape `(D,)` to an activation batch of shape `(N, D)`)[cite: 2, 29]. During the reverse pass, gradients naturally take the broadcasted shape `(N, D)`. Our `unbroadcast()` utility dynamically detects dimension additions and singleton axis expansions, summing across those axes to accumulate mathematically precise gradients back into the source tensor[cite: 2, 29].
- **Numerically Stable Softmax Cross-Entropy:** Directly composing $\log(\text{softmax}(z))$ introduces severe floating-point underflow/overflow[cite: 2, 28]. We implement a fused softmax cross-entropy primitive that subtracts row-wise maximums ($\hat{z}_i = z_i - \max(z)$), normalizes exponentials with dynamic epsilons, and directly derives the analytic gradient:
  $$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{p_i - y_i}{N}$$
  preventing catastrophic cancellation and ensuring zero-NaN training stability[cite: 2, 3, 28].

---

### 2. The Masked Weight Gradient & Optimizer Corruption Fix

#### The Mathematical Trap
When neural connections are pruned, the weight tensor $W$ is multiplied element-wise by a binary mask $M \in \{0, 1\}^{d_{\text{in}} \times d_{\text{out}}}$:
$$W_{\text{eff}} = W \odot M$$

By the multidimensional chain rule, the loss gradient with respect to the raw parameter tensor $W$ is:
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial W_{\text{eff}}} \odot \frac{\partial W_{\text{eff}}}{\partial W} = \frac{\partial \mathcal{L}}{\partial W_{\text{eff}}} \odot M$$

Consequently, for any pruned weight where $M_{ij} = 0$, the mathematical gradient $\nabla_{W_{ij}} \mathcal{L} \equiv 0$ identically[cite: 3, 10].

#### The Adam Optimizer Drift Bug
In standard Adam, first ($m_t$) and second ($v_t$) raw moment buffers update as follows[cite: 10, 24]:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

If an engine merely outputs $g_t = 0$ for masked weights without synchronizing optimizer state, Adam updates:
$$m_t = \beta_1 m_{t-1}, \quad v_t = \beta_2 v_{t-1}$$
$$\Delta W_t = -\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \approx -\eta \frac{m_0 \beta_1^t \sqrt{1 - \beta_2^t}}{\sqrt{v_0 \beta_2^t}\sqrt{1 - \beta_1^t}}$$

Because standard decay rates differ radically ($\beta_1 = 0.9$ while $\sqrt{\beta_2} = \sqrt{0.999} \approx 0.9995$), **$\Delta W_t$ does not vanish immediately**. Instead, the optimizer continues actively mutating and updating the "pruned" zero weight under the mask using stale exponential momentum[cite: 3, 10]! 

Furthermore, if the connection is later dynamically **revived**, the contaminated momentum buffer triggers catastrophic gradient spikes, destabilizing training[cite: 3, 4, 10].

#### The Engine Solution (`nn/optim.py`)
Our `Adam.step()` implementation takes explicit awareness of layer masks[cite: 24]:
1. Forces $W_{ij} \equiv 0$ on all masked indices[cite: 10, 24].
2. Synchronously zeros the first moment buffer: $m_{t, ij} = 0$[cite: 10, 24].
3. Synchronously zeros the second moment buffer: $v_{t, ij} = 0$[cite: 10, 24].
4. Clears accumulated gradient buffers: $g_{ij} = 0$[cite: 24].

When a connection is revived via dynamic regrowth, it starts with an uncorrupted, pristine state[cite: 10, 15].

---

### 3. Saliency Criterion & Sparsity Scheduling (`prune/`)

#### Derivation of First-Order Taylor Saliency
Trivial magnitude pruning assumes that smaller absolute weights $\vert{}W_{ij}\vert{}$ are less important[cite: 3, 10, 22]. However, a large weight operating on a dead/saturated activation channel contributes nothing to the network's output, while a tiny weight sitting on a sensitive classification boundary will destroy task performance if zeroed.

To quantify the exact loss perturbation $\Delta \mathcal{L}_{ij}$ when pruning connection $W_{ij} \to 0$[cite: 3, 10]:
$$\Delta \mathcal{L}_{ij} = \mathcal{L}(W - W_{ij} e_i e_j^T) - \mathcal{L}(W)$$

Expanding this via a Taylor series around the current weight configuration $W$[cite: 3, 10]:
$$\Delta \mathcal{L}_{ij} = -\frac{\partial \mathcal{L}}{\partial W_{ij}} W_{ij} + \frac{1}{2} \frac{\partial^2 \mathcal{L}}{\partial W_{ij}^2} W_{ij}^2 - \mathcal{O}(\Vert{}W_{ij}\Vert{}^3)$$

Neglecting higher-order curvature terms during backpropagation, the loss sensitivity is approximated directly by the first-order Taylor expansion[cite: 3, 10]:
$$S_{ij} = \left\vert{} W_{ij} \cdot \frac{\partial \mathcal{L}}{\partial W_{ij}} \right\vert{}$$

Connections with the smallest $S_{ij}$ score contribute the least to objective minimization and are safely pruned[cite: 3, 20].

#### Cubic Sparsity Ramp Schedule
One-shot pruning shocks the parameter manifold, often causing unrecoverable accuracy degradation[cite: 3, 4]. We implement a cubic polynomial decay ramp (Zhu & Gupta, 2017)[cite: 21]:
$$s_t = s_f + (s_i - s_f) \left( 1 - \frac{t - t_0}{t_f - t_0} \right)^3$$
where:
- $s_i = 0.0$ (initial dense model)[cite: 13, 21]
- $s_f$ = target sparsity (e.g., $0.90$ for 90% connection removal)[cite: 13, 21]
- $t_0, t_f$ = start and end optimization steps for pruning[cite: 21]

This gradually phases in sparsity, giving surviving weights ample gradient steps to co-adapt and recover capacity[cite: 3, 4].

#### Dynamic Regrowth
To prevent permanent architectural dead-ends caused by early pruning mistakes, the pruner inspects masked weights ($M_{ij} = 0$) at each scheduled step[cite: 4, 20]. Connections exhibiting the highest incoming gradient magnitudes $\left\vert{}\frac{\partial \mathcal{L}}{\partial W_{ij}}\right\vert{}$ earn the right to revive ($M_{ij} \to 1$), enabling the graph to continuously explore optimal sparse topologies[cite: 2, 4, 20].

---

### 4. Real Hardware Cost vs. "Dense $\times$ Zero" (`nn/layers.py`)

Multiplying an activation matrix by a dense matrix filled with zeros (`W * 0.0`) still runs at $\mathcal{O}(N \cdot D_{\text{in}} \cdot D_{\text{out}})$ FLOPs on CPU/GPU hardware[cite: 4, 25]. Claiming this as an inference speedup is fundamentally misleading[cite: 4].

To validate true computational reduction[cite: 4]:
- `Linear.forward_sparse_inference()` unpacks non-zero connections in Compressed Sparse Row/Column formats, computing inner dot products **only** over active input indices[cite: 25]:
  $$\text{FLOPs}_{\text{dense}} = 2 \cdot D_{\text{in}} \cdot D_{\text{out}}$$
  $$\text{FLOPs}_{\text{sparse}} = 2 \cdot \Vert{}M\Vert{}_0$$
- At 90% sparsity, arithmetic operations scale down by an honest $10\times$ factor, demonstrating genuine hardware cost reduction.

---

## Directory & File Structure

```text
Tredence-Assesment-/
├── pyproject.toml              # Build specifications, PEP 621 metadata, dependencies
├── README.md                   # Full operational documentation & architecture walkthrough
├── DESIGN.md                   # Mathematical derivations, optimizer mechanics, scaling analysis
├── engine/                     # Reverse-Mode Autodiff Engine
│   ├── __init__.py             # Exports Tensor and fundamental operations
│   ├── tensor.py               # Tensor DAG node, topological sort, unbroadcasting
│   └── ops.py                  # Differentiable forward/backward implementations
├── nn/                         # Neural Network & Optimization Layer
│   ├── __init__.py             # Exports Module, Parameter, Linear, MLP, Adam
│   ├── module.py               # Base Module and Parameter abstractions
│   ├── layers.py               # Linear layer (dense + sparse inference) & MLP container
│   └── optim.py                # Mask-aware Adam optimizer with state synchronization
├── prune/                      # Connection Pruning & Scheduling
│   ├── __init__.py             # Exports SelfPruner, criteria, and schedules
│   ├── criteria.py             # Taylor Saliency and Magnitude scoring
│   ├── scheduler.py            # Polynomial cubic sparsity decay ramp
│   └── mask.py                 # Pruner engine with dynamic gradient regrowth
├── train/                      # Training Pipeline
│   ├── __init__.py             # Exports dataset utilities and training step runner
│   ├── dataset.py              # Permitted Digits dataset normalization and train/test split
│   └── engine.py               # Mini-batch gradient step and evaluation loops
├── tests/                      # Automated Verification Suite
│   ├── test_gradcheck.py       # Finite-difference gradient checks across all operators
│   └── test_masking_optimizer.py # Test verifying masked gradient zeroing and Adam state isolation
└── experiments/                # Reproducibility Entrypoints
    ├── run_part2_dense.py      # Part 2: Baseline dense MLP training
    ├── run_part3_prune.py      # Part 3: 90% Self-pruning training run
    └── run_part4_pareto.py     # Part 4: Multi-seed Pareto sweep, FLOP analysis, plot generation
