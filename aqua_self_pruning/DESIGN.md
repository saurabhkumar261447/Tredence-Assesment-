# AQUA Platform Challenge: Design & Technical Defense

## 1. Derivation of Importance Criterion
We seek to determine the loss change caused by removing connection $W_{ij}$ ($W_{ij} 	o 0$):
$$\Delta \mathcal{L}_{ij} = \mathcal{L}(W - W_{ij} e_i e_j^T) - \mathcal{L}(W)$$

Using a 1st-order Taylor series expansion about parameter $W$:
$$\Delta \mathcal{L}_{ij} = - \frac{\partial \mathcal{L}}{\partial W_{ij}} W_{ij} + \mathcal{O}(\|W_{ij}\|^2)$$

Taking magnitude to score positive and negative contributions:
$$S_{ij} = \left| W_{ij} \cdot \frac{\partial \mathcal{L}}{\partial W_{ij}} \right|$$

### Why this surpasses magnitude-only pruning:
Magnitude assumes large weights always dominate output variance. In deep models, a large weight can connect to saturated or dead neurons (loss gradient near 0), while a small weight with massive gradient sensitivity will catastrophically alter predictions if removed. Taylor saliency measures actual loss perturbation.

---

## 2. Gradient of a Masked Weight & Optimizer State Mechanics

### The Gradient:
Effective weight forward computation is:
$$W_{\text{eff}} = W \odot M$$
By the chain rule:
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial W_{\text{eff}}} \odot M$$
For all indices where $M_{ij} = 0$, $\frac{\partial \mathcal{L}}{\partial W_{ij}} \equiv 0$.

### The Optimizer Momentum Corrupter Bug:
In standard Adam:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
If $g_t = 0$, $m_t = \beta_1 m_{t-1}$ and $v_t = \beta_2 v_{t-1}$.
The step update is:
$$\Delta W_t = -\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
Since $\beta_1 \neq \sqrt{\beta_2}$ ($0.90 \neq \sqrt{0.999} \approx 0.9995$), $\Delta W_t \neq 0$. Unmasked optimizers continuously mutate frozen zero weights under the hood!

**Our Fix:**
Our engine explicitly zeros $W_{ij} = 0$, sets $m_{t, ij} = 0$, and sets $v_{t, ij} = 0$ for all $(i,j) \in \{M_{ij} = 0\}$. If dynamic regrowth reactivates a connection, it starts with an uncorrupted, fresh momentum buffer.

---

## 3. Autodiff Engine Bottlenecks & Optimization Roadmap
1. **Dynamic DAG Allocation Overhead:** Python class instantiations per operation incur interpreter latency. Can be optimized by static tape recording (XLA / JAX style) and kernel fusion.
2. **Intermediate Activations:** Uncheckpointed forward graphs retain large matrices. Gradient checkpointing can trade compute for memory footprint.

---

## 4. Serving Self-Pruned Models at Scale (Multi-Tenant Inference)
1. **True CSR / Block-Sparse Kernels:** Convert binary masks into Compressed Sparse Row (CSR) or NVIDIA $2:4$ structured sparse formats to realize actual hardware acceleration.
2. **Cache Memory Bandwidth:** Pruned weights consume $\approx 10\times$ less memory bandwidth, allowing models to fit into on-chip SRAM/L2 cache and slashing DRAM access bottlenecks in high-concurrency multi-tenant clusters.
