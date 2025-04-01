from dataclasses import dataclass

@dataclass
class Config:
    epochs: int = 1000
    gmin: float = 1e-3
    l2_lambda: float = 0.3
    l1_lambda: float = 0.3
    epsilon: float = 1e-3
    tolerance: float = 1e-3

    learning_rate: float = 5
    tolerance_grad: float = 1e-6

    k: int = 10

    seed: int = 42
