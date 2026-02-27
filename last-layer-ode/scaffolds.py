from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch


@dataclass(frozen=True)
class Scaffold:
    P: int
    theta_dim: int
    state_names: tuple[str, ...]
    rhs: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def rk4_step(rhs, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    if dt.ndim == 1:
        dt = dt[:, None]
    k1 = rhs(y, theta)
    k2 = rhs(y + 0.5 * dt * k1, theta)
    k3 = rhs(y + 0.5 * dt * k2, theta)
    k4 = rhs(y + dt * k3, theta)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.clamp_min(y_next, 0.0)


# ============================================================
# RHS LIBRARY
# Full model states: ['A','B','C','D','E','F','G','H','I','J','K','L','M']
# ============================================================

# -------------------------
# reduced2 (odd): [A, M]
# -------------------------
def rhs_2_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, M = y.unbind(dim=-1)
    kf, kr = theta.unbind(dim=-1)
    dA = -kf * A + kr * M
    dM =  kf * A - kr * M
    return torch.stack([dA, dM], dim=-1)


# -------------------------
# reduced3 (odd): [A, J, M]
# -------------------------
def rhs_3_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, J, M = y.unbind(dim=-1)
    kf1, kf2, kr1, kr2 = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * J
    dJ =  kf1 * A - kr1 * J - kf2 * J + kr2 * M
    dM =  kf2 * J - kr2 * M
    return torch.stack([dA, dJ, dM], dim=-1)


# -------------------------
# reduced4 (even variant): [A, G, J, L]  (end at L, no M)
# theta: (B,6) [kf1,kf2,kf3, kr1,kr2,kr3] with last pair J<->L
# -------------------------
def rhs_4_noM_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, G, J, L = y.unbind(dim=-1)
    kf1, kf2, kf3, kr1, kr2, kr3 = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * G
    dG =  kf1 * A - kr1 * G - kf2 * G + kr2 * J
    dJ =  kf2 * G - kr2 * J - kf3 * J + kr3 * L
    dL =  kf3 * J - kr3 * L
    return torch.stack([dA, dG, dJ, dL], dim=-1)


# -------------------------
# reduced5 (odd): [A, D, G, J, M]
# -------------------------
def rhs_5_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, D, G, J, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kf4, kr1, kr2, kr3, kr4 = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * D
    dD =  kf1 * A - kr1 * D - kf2 * D + kr2 * G
    dG =  kf2 * D - kr2 * G - kf3 * G + kr3 * J
    dJ =  kf3 * G - kr3 * J - kf4 * J + kr4 * M
    dM =  kf4 * J - kr4 * M
    return torch.stack([dA, dD, dG, dJ, dM], dim=-1)


# -------------------------
# reduced6 (even variant): [A, B, D, G, J, L]  (end at L, no M)
# theta: (B,10) [kfAB,kfBD,kfDG,kfGJ,kfJL,  krAB,krBD,krDG,krGJ,krJL]
# -------------------------
def rhs_6_noM_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, B, D, G, J, L = y.unbind(dim=-1)
    kfAB, kfBD, kfDG, kfGJ, kfJL, krAB, krBD, krDG, krGJ, krJL = theta.unbind(dim=-1)

    dA = -kfAB * A + krAB * B
    dB =  kfAB * A - krAB * B - kfBD * B + krBD * D
    dD =  kfBD * B - krBD * D - kfDG * D + krDG * G
    dG =  kfDG * D - krDG * G - kfGJ * G + krGJ * J
    dJ =  kfGJ * G - krGJ * J - kfJL * J + krJL * L
    dL =  kfJL * J - krJL * L
    return torch.stack([dA, dB, dD, dG, dJ, dL], dim=-1)

def rhs_6_ADGJLM_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, D, G, J, L, M = y.unbind(dim=-1)
    kfAD, kfDG, kfGJ, kfJL, kfLM, krAD, krDG, krGJ, krJL, krLM = theta.unbind(dim=-1)

    dA = -kfAD * A + krAD * D
    dD =  kfAD * A - krAD * D - kfDG * D + krDG * G
    dG =  kfDG * D - krDG * G - kfGJ * G + krGJ * J
    dJ =  kfGJ * G - krGJ * J - kfJL * J + krJL * L
    dL =  kfJL * J - krJL * L - kfLM * L + krLM * M
    dM =  kfLM * L - krLM * M

    return torch.stack([dA, dD, dG, dJ, dL, dM], dim=-1)

def rhs_6_AGHILM_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, G, H, I, L, M = y.unbind(dim=-1)
    kfAG, kfGH, kfHI, kfIL, kfLM, krAG, krGH, krIL, krLM = theta.unbind(dim=-1)

    dA = -kfAG * A + krAG * G

    dG =  kfAG * A - krAG * G - kfGH * G + krGH * H
    dH =  kfGH * G - krGH * H - kfHI * H

    dI =  kfHI * H - kfIL * I + krIL * L
    dL =  kfIL * I - krIL * L - kfLM * L + krLM * M
    dM =  kfLM * L - krLM * M

    return torch.stack([dA, dG, dH, dI, dL, dM], dim=-1)

def rhs_6_DGHILM_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    D, G, H, I, L, M = y.unbind(dim=-1)
    kfDG, kfGH, kfHI, kfIL, kfLM, krDG, krGH, krIL, krLM = theta.unbind(dim=-1)

    dD = -kfDG * D + krDG * G

    dG =  kfDG * D - krDG * G - kfGH * G + krGH * H
    dH =  kfGH * G - krGH * H - kfHI * H

    dI =  kfHI * H - kfIL * I + krIL * L
    dL =  kfIL * I - krIL * L - kfLM * L + krLM * M
    dM =  kfLM * L - krLM * M

    return torch.stack([dD, dG, dH, dI, dL, dM], dim=-1)

def rhs_6_ABCDLM_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, B, C, D, L, M = y.unbind(dim=-1)
    kfAB, kfBC, kfCD, kfDL, kfLM, krAB, krCD, krDL, krLM = theta.unbind(dim=-1)

    dA = -kfAB * A + krAB * B
    dB =  kfAB * A - krAB * B - kfBC * B
    dC =  kfBC * B - kfCD * C + krCD * D
    dD =  kfCD * C - krCD * D - kfDL * D + krDL * L

    dL =  kfDL * D - krDL * L - kfLM * L + krLM * M
    dM =  kfLM * L - krLM * M

    return torch.stack([dA, dB, dC, dD, dL, dM], dim=-1)

# -------------------------
# reduced7 (odd): [A, D, G, J, K, L, M]
# -------------------------
def rhs_7_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, D, G, J, K, L, M = y.unbind(dim=-1)
    kfAD, kfDG, kfGJ, kf10, kf11, kf12, krAD, krDG, krGJ, kr11, kr12 = theta.unbind(dim=-1)

    dA = -kfAD * A + krAD * D
    dD =  kfAD * A - krAD * D - kfDG * D + krDG * G
    dG =  kfDG * D - krDG * G - kfGJ * G + krGJ * J
    dJ =  kfGJ * G - krGJ * J - kf10 * J
    dK =  kf10 * J - kf11 * K + kr11 * L
    dL =  kf11 * K - kr11 * L - kf12 * L + kr12 * M
    dM =  kf12 * L - kr12 * M
    return torch.stack([dA, dD, dG, dJ, dK, dL, dM], dim=-1)


# -------------------------
# reduced8 (even variant): [A, D, G, H, I, J, K, L]  (end at L, no M)
# This mirrors reduced9's internal chain but truncates the last L<->M pair.
# theta: (B,12) = [kfAD,kfDG,kf7,kf8,kf9,kf10,kf11,  krAD,krDG,kr7,kr9,kr11]
# -------------------------
def rhs_8_noM_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, D, G, H, I, J, K, L = y.unbind(dim=-1)
    (
        kfAD, kfDG, kf7, kf8, kf9, kf10, kf11,
        krAD, krDG, kr7, kr9, kr11
    ) = theta.unbind(dim=-1)

    dA = -kfAD * A + krAD * D
    dD =  kfAD * A - krAD * D - kfDG * D + krDG * G

    dG =  kfDG * D - krDG * G - kf7 * G + kr7 * H
    dH =  kf7 * G - kr7 * H - kf8 * H
    dI =  kf8 * H - kf9 * I + kr9 * J
    dJ =  kf9 * I - kr9 * J - kf10 * J

    dK =  kf10 * J - kf11 * K + kr11 * L
    dL =  kf11 * K - kr11 * L
    return torch.stack([dA, dD, dG, dH, dI, dJ, dK, dL], dim=-1)


# -------------------------
# reduced9 (odd): [A, D, G, H, I, J, K, L, M]
# -------------------------
def rhs_9_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, D, G, H, I, J, K, L, M = y.unbind(dim=-1)
    (
        kfAD, kfDG, kf7, kf8, kf9, kf10, kf11, kf12,
        krAD, krDG, kr7, kr9, kr11, kr12
    ) = theta.unbind(dim=-1)

    dA = -kfAD * A + krAD * D
    dD =  kfAD * A - krAD * D - kfDG * D + krDG * G

    dG =  kfDG * D - krDG * G - kf7 * G + kr7 * H
    dH =  kf7 * G - kr7 * H - kf8 * H
    dI =  kf8 * H - kf9 * I + kr9 * J
    dJ =  kf9 * I - kr9 * J - kf10 * J

    dK =  kf10 * J - kf11 * K + kr11 * L
    dL =  kf11 * K - kr11 * L - kf12 * L + kr12 * M
    dM =  kf12 * L - kr12 * M
    return torch.stack([dA, dD, dG, dH, dI, dJ, dK, dL, dM], dim=-1)


# -------------------------
# reduced10 (even variant): [A, B, C, D, E, F, G, H, I, L]  (end at L, no M, no J)
# theta: (B,14) = [kf1,kf2,kf3,kf4,kf5,kf6,kf7,kf8,kf9,
#                  kr1,kr3,kr5,kr7,kr9] with last pair I<->L
# -------------------------
def rhs_10_noM_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, B, C, D, E, F, G, H, I, L = y.unbind(dim=-1)
    (
        kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9,
        kr1, kr3, kr5, kr7, kr9
    ) = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * B
    dB =  kf1 * A - kr1 * B - kf2 * B
    dC =  kf2 * B - kf3 * C + kr3 * D
    dD =  kf3 * C - kr3 * D - kf4 * D
    dE =  kf4 * D - kf5 * E + kr5 * F
    dF =  kf5 * E - kr5 * F - kf6 * F
    dG =  kf6 * F - kf7 * G + kr7 * H
    dH =  kf7 * G - kr7 * H - kf8 * H
    dI =  kf8 * H - kf9 * I + kr9 * L
    dL =  kf9 * I - kr9 * L
    return torch.stack([dA, dB, dC, dD, dE, dF, dG, dH, dI, dL], dim=-1)


# -------------------------
# reduced11 (odd): [A, B, C, D, E, F, G, H, I, J, M]
# -------------------------
def rhs_11_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, B, C, D, E, F, G, H, I, J, M = y.unbind(dim=-1)

    (
        kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kfJM,
        kr1, kr3, kr5, kr7, kr9, krJM,
    ) = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * B
    dB =  kf1 * A - kr1 * B - kf2 * B
    dC =  kf2 * B - kf3 * C + kr3 * D
    dD =  kf3 * C - kr3 * D - kf4 * D
    dE =  kf4 * D - kf5 * E + kr5 * F
    dF =  kf5 * E - kr5 * F - kf6 * F
    dG =  kf6 * F - kf7 * G + kr7 * H
    dH =  kf7 * G - kr7 * H - kf8 * H
    dI =  kf8 * H - kf9 * I + kr9 * J

    dJ =  kf9 * I - kr9 * J - kfJM * J + krJM * M
    dM =  kfJM * J - krJM * M

    return torch.stack([dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ, dM], dim=-1)


# -------------------------
# reduced12 (even variant): [A, B, C, D, E, F, G, H, I, J, K, L]  (this already excludes M)
# keep as-is (your original rhs_12_torch).
# -------------------------
def rhs_12_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, B, C, D, E, F, G, H, I, J, K, L = y.unbind(dim=-1)
    (
        kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kf10, kf11,
        kr1, kr3, kr5, kr7, kr9, kr11
    ) = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * B
    dB =  kf1 * A - kr1 * B - kf2 * B
    dC =  kf2 * B - kf3 * C + kr3 * D
    dD =  kf3 * C - kr3 * D - kf4 * D
    dE =  kf4 * D - kf5 * E + kr5 * F
    dF =  kf5 * E - kr5 * F - kf6 * F
    dG =  kf6 * F - kf7 * G + kr7 * H
    dH =  kf7 * G - kr7 * H - kf8 * H
    dI =  kf8 * H - kf9 * I + kr9 * J
    dJ =  kf9 * I - kr9 * J - kf10 * J
    dK =  kf10 * J - kf11 * K + kr11 * L
    dL =  kf11 * K - kr11 * L
    return torch.stack([dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ, dK, dL], dim=-1)


# -------------------------
# full13: keep original full model
# -------------------------
def rhs_13_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, B, C, D, E, F, G, H, I, J, K, L, M = y.unbind(dim=-1)
    (
        kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kf10, kf11, kf12,
        kr1, kr3, kr5, kr7, kr9, kr11, kr12
    ) = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * B
    dB =  kf1 * A - kr1 * B - kf2 * B
    dC =  kf2 * B - kf3 * C + kr3 * D
    dD =  kf3 * C - kr3 * D - kf4 * D
    dE =  kf4 * D - kf5 * E + kr5 * F
    dF =  kf5 * E - kr5 * F - kf6 * F
    dG =  kf6 * F - kf7 * G + kr7 * H
    dH =  kf7 * G - kr7 * H - kf8 * H
    dI =  kf8 * H - kf9 * I + kr9 * J
    dJ =  kf9 * I - kr9 * J - kf10 * J
    dK =  kf10 * J - kf11 * K + kr11 * L
    dL =  kf11 * K - kr11 * L - kf12 * L + kr12 * M
    dM =  kf12 * L - kr12 * M
    return torch.stack([dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ, dK, dL, dM], dim=-1)


SCAFFOLDS = {
    "reduced2": Scaffold(P=2, theta_dim=2, state_names=("A", "M"), rhs=rhs_2_torch),
    "reduced3": Scaffold(P=3, theta_dim=4, state_names=("A", "J", "M"), rhs=rhs_3_torch),
    "reduced5": Scaffold(P=5, theta_dim=8, state_names=("A", "D", "G", "J", "M"), rhs=rhs_5_torch),
    "reduced7": Scaffold(P=7, theta_dim=11, state_names=("A", "D", "G", "J", "K", "L", "M"), rhs=rhs_7_torch),
    "reduced9": Scaffold(P=9, theta_dim=14, state_names=("A", "D", "G", "H", "I", "J", "K", "L", "M"), rhs=rhs_9_torch),
    "reduced11": Scaffold(P=11, theta_dim=15, state_names=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "M"), rhs=rhs_11_torch),

    "reduced4": Scaffold(P=4, theta_dim=6, state_names=("A", "G", "J", "L"), rhs=rhs_4_noM_torch),
    "reduced6": Scaffold(P=6, theta_dim=10, state_names=("A", "B", "D", "G", "J", "L"), rhs=rhs_6_noM_torch),
    "reduced6_ADGJLM": Scaffold(P=6, theta_dim=10, state_names=("A", "D", "G", "J", "L", "M"), rhs=rhs_6_ADGJLM_torch),
    "reduced6_AGHILM": Scaffold(P=6, theta_dim=9, state_names=("A", "G", "H", "I", "L", "M"), rhs=rhs_6_AGHILM_torch),
    "reduced6_DGHILM": Scaffold(P=6, theta_dim=9, state_names=("D", "G", "H", "I", "L", "M"), rhs=rhs_6_DGHILM_torch),
    "reduced6_ABCDLM": Scaffold(P=6, theta_dim=9, state_names=("A", "B", "C", "D", "L", "M"), rhs=rhs_6_ABCDLM_torch),
    "reduced8": Scaffold(P=8, theta_dim=12, state_names=("A", "D", "G", "H", "I", "J", "K", "L"), rhs=rhs_8_noM_torch),
    "reduced10": Scaffold(P=10, theta_dim=14, state_names=("A", "B", "C", "D", "E", "F", "G", "H", "I", "L"), rhs=rhs_10_noM_torch),
    "reduced12": Scaffold(P=12, theta_dim=17, state_names=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"), rhs=rhs_12_torch),

    "full13": Scaffold(P=13, theta_dim=19, state_names=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"), rhs=rhs_13_torch),
}