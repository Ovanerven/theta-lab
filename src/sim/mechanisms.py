from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List

import torch


@dataclass(frozen=True)
class Mechanism:
    name: str
    n_state: int
    n_param: int
    state_names: List[str]
    param_names: List[str]
    rhs: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (B,n_state),(B,n_param)->(B,n_state)


# -------------------------
# RHS definitions
# -------------------------
# Here we can define different mechanisms by writing out their ODE right-hand sides. 
def full13_rhs(y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    A, B, C, D, E, F, G, H, I, J, K, L, M = y.unbind(dim=-1)
    (kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kf10, kf11, kf12,
     kr1, kr3, kr5, kr7, kr9, kr11, kr12) = k.unbind(dim=-1)

    dA = -kf1*A  + kr1*B
    dB =  kf1*A  - kr1*B  - kf2*B
    dC =  kf2*B  - kf3*C  + kr3*D
    dD =  kf3*C  - kr3*D  - kf4*D
    dE =  kf4*D  - kf5*E  + kr5*F
    dF =  kf5*E  - kr5*F  - kf6*F
    dG =  kf6*F  - kf7*G  + kr7*H
    dH =  kf7*G  - kr7*H  - kf8*H
    dI =  kf8*H  - kf9*I  + kr9*J
    dJ =  kf9*I  - kr9*J  - kf10*J
    dK =  kf10*J - kf11*K + kr11*L
    dL =  kf11*K - kr11*L - kf12*L + kr12*M
    dM =  kf12*L - kr12*M

    return torch.stack([dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ, dK, dL, dM], dim=-1)


def reduced5_rhs(y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    A, D, G, J, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kf4, kr1, kr2, kr3, kr4 = k.unbind(dim=-1)

    dA = -kf1*A + kr1*D
    dD =  kf1*A - kr1*D - kf2*D + kr2*G
    dG =  kf2*D - kr2*G - kf3*G + kr3*J
    dJ =  kf3*G - kr3*J - kf4*J + kr4*M
    dM =  kf4*J - kr4*M
    return torch.stack([dA, dD, dG, dJ, dM], dim=-1)


# 7-state chain: [A, C, E, G, I, K, M]
def reduced7_rhs(y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    A, C, E, G, I, K, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kf4, kf5, kf6, kr1, kr2, kr3, kr4, kr5, kr6 = k.unbind(dim=-1)

    dA = -kf1*A + kr1*C
    dC =  kf1*A - kr1*C - kf2*C + kr2*E
    dE =  kf2*C - kr2*E - kf3*E + kr3*G
    dG =  kf3*E - kr3*G - kf4*G + kr4*I
    dI =  kf4*G - kr4*I - kf5*I + kr5*K
    dK =  kf5*I - kr5*K - kf6*K + kr6*M
    dM =  kf6*K - kr6*M
    return torch.stack([dA, dC, dE, dG, dI, dK, dM], dim=-1)


# 8-state chain: [A, B, D, F, H, J, L, M]
def reduced8_rhs(y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    A, B, D, F, H, J, L, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kf4, kf5, kf6, kf7, kr1, kr2, kr3, kr4, kr5, kr6, kr7 = k.unbind(dim=-1)

    dA = -kf1*A + kr1*B
    dB =  kf1*A - kr1*B - kf2*B + kr2*D
    dD =  kf2*B - kr2*D - kf3*D + kr3*F
    dF =  kf3*D - kr3*F - kf4*F + kr4*H
    dH =  kf4*F - kr4*H - kf5*H + kr5*J
    dJ =  kf5*H - kr5*J - kf6*J + kr6*L
    dL =  kf6*J - kr6*L - kf7*L + kr7*M
    dM =  kf7*L - kr7*M
    return torch.stack([dA, dB, dD, dF, dH, dJ, dL, dM], dim=-1)


# 9-state chain: [A, B, C, E, G, I, K, L, M]
def reduced9_rhs(y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    A, B, C, E, G, I, K, L, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kr1, kr2, kr3, kr4, kr5, kr6, kr7, kr8 = k.unbind(dim=-1)

    dA = -kf1*A + kr1*B
    dB =  kf1*A - kr1*B - kf2*B + kr2*C
    dC =  kf2*B - kr2*C - kf3*C + kr3*E
    dE =  kf3*C - kr3*E - kf4*E + kr4*G
    dG =  kf4*E - kr4*G - kf5*G + kr5*I
    dI =  kf5*G - kr5*I - kf6*I + kr6*K
    dK =  kf6*I - kr6*K - kf7*K + kr7*L
    dL =  kf7*K - kr7*L - kf8*L + kr8*M
    dM =  kf8*L - kr8*M
    return torch.stack([dA, dB, dC, dE, dG, dI, dK, dL, dM], dim=-1)

# Use a dictionary to store information about every mechanism.
MECH = {
    "full13": Mechanism(
        name="full13",
        n_state=13,
        n_param=19,
        state_names=list("ABCDEFGHIJKLM"),
        param_names=[
            "kf1","kf2","kf3","kf4","kf5","kf6","kf7","kf8","kf9","kf10","kf11","kf12",
            "kr1","kr3","kr5","kr7","kr9","kr11","kr12",
        ],
        rhs=full13_rhs,
    ),
    "reduced5": Mechanism(
        name="reduced5",
        n_state=5,
        n_param=8,
        state_names=["A","D","G","J","M"],
        param_names=["kf1","kf2","kf3","kf4","kr1","kr2","kr3","kr4"],
        rhs=reduced5_rhs,
    ),
    "reduced7": Mechanism(
        name="reduced7",
        n_state=7,
        n_param=12,
        state_names=["A","C","E","G","I","K","M"],
        param_names=["kf1","kf2","kf3","kf4","kf5","kf6","kr1","kr2","kr3","kr4","kr5","kr6"],
        rhs=reduced7_rhs,
    ),
    "reduced8": Mechanism(
        name="reduced8",
        n_state=8,
        n_param=14,
        state_names=["A","B","D","F","H","J","L","M"],
        param_names=["kf1","kf2","kf3","kf4","kf5","kf6","kf7","kr1","kr2","kr3","kr4","kr5","kr6","kr7"],
        rhs=reduced8_rhs,
    ),
    "reduced9": Mechanism(
        name="reduced9",
        n_state=9,
        n_param=16,
        state_names=["A","B","C","E","G","I","K","L","M"],
        param_names=["kf1","kf2","kf3","kf4","kf5","kf6","kf7","kf8","kr1","kr2","kr3","kr4","kr5","kr6","kr7","kr8"],
        rhs=reduced9_rhs,
    ),
}
