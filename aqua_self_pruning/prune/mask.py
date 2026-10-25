import numpy as np
from typing import List
from nn.layers import Linear
from prune.criteria import compute_taylor_saliency, compute_magnitude

class SelfPruner:
    def __init__(
        self,
        layers: List[Linear],
        schedule,
        criterion: str = "taylor",
        allow_regrowth: bool = True,
        regrowth_rate: float = 0.15
    ):
        self.layers = layers
        self.schedule = schedule
        self.criterion = criterion
        self.allow_regrowth = allow_regrowth
        self.regrowth_rate = regrowth_rate

    def step(self, current_step: int) -> float:
        target_sparsity = self.schedule.get_sparsity(current_step)
        if target_sparsity <= 0.0:
            return 0.0

        for layer in self.layers:
            if self.criterion == "taylor":
                scores = compute_taylor_saliency(layer)
            elif self.criterion == "magnitude":
                scores = compute_magnitude(layer)
            else:
                raise ValueError(f"Unknown criterion {self.criterion}")

            total_weights = scores.size
            num_pruned = int(np.round(total_weights * target_sparsity))

            if num_pruned >= total_weights:
                num_pruned = total_weights - 1

            if num_pruned == 0:
                continue

            current_mask = layer.mask.copy()

            # Dynamic Regrowth: revive pruned weights that exhibit high incoming gradients
            if self.allow_regrowth and (current_step < self.schedule.end_step):
                num_regrow = int(np.round(self.regrowth_rate * num_pruned))
                if num_regrow > 0 and layer.weight.grad is not None:
                    masked_indices = np.where(current_mask == 0)
                    if len(masked_indices[0]) > 0:
                        grad_magnitudes = np.abs(layer.weight.grad[masked_indices])
                        top_regrow_k = min(num_regrow, len(grad_magnitudes))
                        best_masked_idx = np.argpartition(grad_magnitudes, -top_regrow_k)[-top_regrow_k:]
                        
                        regrow_rows = masked_indices[0][best_masked_idx]
                        regrow_cols = masked_indices[1][best_masked_idx]
                        current_mask[regrow_rows, regrow_cols] = 1.0

            # Prune lowest scores to meet target sparsity
            flat_scores = scores.flatten()
            flat_idx = np.argpartition(flat_scores, num_pruned)[:num_pruned]
            new_mask_flat = np.ones_like(flat_scores)
            new_mask_flat[flat_idx] = 0.0
            new_mask = new_mask_flat.reshape(layer.mask.shape)

            layer.set_mask(new_mask)

        return target_sparsity
