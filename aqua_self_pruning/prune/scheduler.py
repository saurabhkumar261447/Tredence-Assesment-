import numpy as np

class PolynomialDecayPruningSchedule:
    """
    Cubic Sparsity Ramp:
    s_t = s_f + (s_i - s_f) * (1 - (t - t_0) / (t_f - t_0))^3
    """
    def __init__(self, initial_sparsity: float, target_sparsity: float, start_step: int, end_step: int):
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.start_step = start_step
        self.end_step = end_step

    def get_sparsity(self, step: int) -> float:
        if step < self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.target_sparsity
        
        progress = (step - self.start_step) / float(self.end_step - self.start_step)
        factor = (1.0 - progress) ** 3
        return self.target_sparsity + (self.initial_sparsity - self.target_sparsity) * factor
