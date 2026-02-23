from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import torch


@dataclass(frozen=True)
class ScaffoldSpec:
    P: int
    theta_dim: int


class MechanisticScaffold(torch.nn.Module):
    def __init__(self, P: int, theta_dim: int):
        super().__init__()
        self.spec = ScaffoldSpec(P=P, theta_dim=theta_dim)

    def rhs(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.jit.export
    def step(self, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        k1 = self.rhs(y, theta)
        k2 = self.rhs(y + 0.5 * dt * k1, theta)
        k3 = self.rhs(y + 0.5 * dt * k2, theta)
        k4 = self.rhs(y + dt * k3, theta)
        y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return torch.clamp_min(y_next, 0.0)


class CustomRHSScaffold(MechanisticScaffold):
    def __init__(
        self,
        P: int,
        theta_dim: int,
        rhs_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        state_names: Optional[Sequence[str]] = None,
        param_names: Optional[Sequence[str]] = None,
    ):
        super().__init__(P=P, theta_dim=theta_dim)
        self._rhs_fn = rhs_fn
        self._state_names = list(state_names) if state_names is not None else None
        self._param_names = list(param_names) if param_names is not None else None

        if self._state_names is not None and len(self._state_names) != P:
            raise ValueError(f"state_names must have length P={P}")
        if self._param_names is not None and len(self._param_names) != theta_dim:
            raise ValueError(f"param_names must have length theta_dim={theta_dim}")

    def rhs(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self._rhs_fn(y, theta)

    def state_names(self) -> list[str]:
        if self._state_names is None:
            return [f"s{i}" for i in range(self.spec.P)]
        return list(self._state_names)

    def param_names(self) -> list[str]:
        if self._param_names is None:
            return [f"θ{i}" for i in range(self.spec.theta_dim)]
        return list(self._param_names)


# ============================================================
# RHS LIBRARY
# Full model states: ['A','B','C','D','E','F','G','H','I','J','K','L','M']
# ============================================================

# =========================
# Reduced 2-state reversible pair: [A, M]
# theta: (B,2) [kf, kr]
# =========================
def rhs_2_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, M = y.unbind(dim=-1)
    kf, kr = theta.unbind(dim=-1)

    dA = -kf * A + kr * M
    dM =  kf * A - kr * M
    return torch.stack([dA, dM], dim=-1)


# =========================
# Reduced 3-state chain: [A, J, M]
# theta: (B,4) [kf1,kf2,kr1,kr2]
# =========================
def rhs_3_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, J, M = y.unbind(dim=-1)
    kf1, kf2, kr1, kr2 = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * J
    dJ =  kf1 * A - kr1 * J - kf2 * J + kr2 * M
    dM =  kf2 * J - kr2 * M

    return torch.stack([dA, dJ, dM], dim=-1)


# =========================
# Reduced 4-state chain: [A, G, J, M]
# theta: (B,6) [kf1,kf2,kf3, kr1,kr2,kr3]
# =========================
def rhs_4_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, G, J, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kr1, kr2, kr3 = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * G
    dG =  kf1 * A - kr1 * G - kf2 * G + kr2 * J
    dJ =  kf2 * G - kr2 * J - kf3 * J + kr3 * M
    dM =  kf3 * J - kr3 * M

    return torch.stack([dA, dG, dJ, dM], dim=-1)


# =========================
# Reduced 5-state chain: [A, D, G, J, M]
# theta: (B,8) [kf1,kf2,kf3,kf4, kr1,kr2,kr3,kr4]
# =========================
def rhs_5_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, D, G, J, M = y.unbind(dim=-1)
    kf1, kf2, kf3, kf4, kr1, kr2, kr3, kr4 = theta.unbind(dim=-1)

    dA = -kf1 * A + kr1 * D
    dD =  kf1 * A - kr1 * D - kf2 * D + kr2 * G
    dG =  kf2 * D - kr2 * G - kf3 * G + kr3 * J
    dJ =  kf3 * G - kr3 * J - kf4 * J + kr4 * M
    dM =  kf4 * J - kr4 * M

    return torch.stack([dA, dD, dG, dJ, dM], dim=-1)


# =========================
# 6-state model: [A, B, D, G, J, M]
# theta: (B,10) = [kfAB,kfBD,kfDG,kfGJ,kfJM,  krAB,krBD,krDG,krGJ,krJM]
# =========================
def rhs_6_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, B, D, G, J, M = y.unbind(dim=-1)
    kfAB, kfBD, kfDG, kfGJ, kfJM, krAB, krBD, krDG, krGJ, krJM = theta.unbind(dim=-1)

    dA = -kfAB * A + krAB * B
    dB =  kfAB * A - krAB * B - kfBD * B + krBD * D
    dD =  kfBD * B - krBD * D - kfDG * D + krDG * G
    dG =  kfDG * D - krDG * G - kfGJ * G + krGJ * J
    dJ =  kfGJ * G - krGJ * J - kfJM * J + krJM * M
    dM =  kfJM * J - krJM * M

    return torch.stack([dA, dB, dD, dG, dJ, dM], dim=-1)


# =========================
# Reduced 7-state chain: [A, D, G, J, K, L, M]
# theta: (B,11) = [kfAD, kfDG, kfGJ, kf10, kf11, kf12,  krAD, krDG, krGJ, kr11, kr12]
# =========================
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


# =========================
# Reduced 8-state chain: [A, D, G, H, I, J, L, M]
# (i.e. drop K; jump J->L with kf10, and L->M with kf12)
# theta: (B,12) = [kfAD,kfDG,kf7,kf8,kf9,kf10,kf12,  krAD,krDG,kr7,kr9,kr12]
# =========================
def rhs_8_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, D, G, H, I, J, L, M = y.unbind(dim=-1)
    (
        kfAD, kfDG, kf7, kf8, kf9, kf10, kf12,
        krAD, krDG, kr7, kr9, kr12
    ) = theta.unbind(dim=-1)

    dA = -kfAD * A + krAD * D
    dD =  kfAD * A - krAD * D - kfDG * D + krDG * G

    dG =  kfDG * D - krDG * G - kf7 * G + kr7 * H
    dH =  kf7 * G - kr7 * H - kf8 * H
    dI =  kf8 * H - kf9 * I + kr9 * J
    dJ =  kf9 * I - kr9 * J - kf10 * J

    dL =  kf10 * J - kf12 * L + kr12 * M
    dM =  kf12 * L - kr12 * M

    return torch.stack([dA, dD, dG, dH, dI, dJ, dL, dM], dim=-1)


# =========================
# Reduced 9-state chain: [A, D, G, H, I, J, K, L, M]
# theta: (B,14) = [kfAD, kfDG, kf7, kf8, kf9, kf10, kf11, kf12,
#                  krAD, krDG, kr7, kr9, kr11, kr12]
# =========================
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


# =========================
# Reduced 10-state: [A, B, C, D, E, F, G, H, I, J]
# (truncate full chain at J; no K/L/M)
# theta: (B,14) = [kf1,kf2,kf3,kf4,kf5,kf6,kf7,kf8,kf9,
#                  kr1,kr3,kr5,kr7,kr9]
# =========================
def rhs_10_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, B, C, D, E, F, G, H, I, J = y.unbind(dim=-1)
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
    dI =  kf8 * H - kf9 * I + kr9 * J
    dJ =  kf9 * I - kr9 * J

    return torch.stack([dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ], dim=-1)


# =========================
# Reduced 11-state: [A, B, C, D, E, F, G, H, I, J, K]
# (truncate at K; no L/M; includes irreversible J->K via kf10)
# theta: (B,15) = [kf1..kf10,  kr1,kr3,kr5,kr7,kr9]
# =========================
def rhs_11_torch(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    A, B, C, D, E, F, G, H, I, J, K = y.unbind(dim=-1)
    (
        kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kf10,
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
    dI =  kf8 * H - kf9 * I + kr9 * J
    dJ =  kf9 * I - kr9 * J - kf10 * J
    dK =  kf10 * J

    return torch.stack([dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ, dK], dim=-1)


# =========================
# Reduced 12-state: [A, B, C, D, E, F, G, H, I, J, K, L]
# (truncate at L; no M; includes K<->L via kf11/kr11)
# theta: (B,17) = [kf1..kf11,  kr1,kr3,kr5,kr7,kr9,kr11]
# =========================
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


# =========================
# Full 13-state model: [A, B, C, D, E, F, G, H, I, J, K, L, M]
# theta: (B,19) = [kf1..kf12, kr1,kr3,kr5,kr7,kr9,kr11,kr12]
# =========================
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


def make_scaffold(name: str) -> CustomRHSScaffold:
    if name == "reduced2":
        return CustomRHSScaffold(
            P=2,
            theta_dim=2,
            rhs_fn=rhs_2_torch,
            state_names=["A", "M"],
            param_names=["kfAM", "krAM"],
        )

    if name == "reduced3":
        return CustomRHSScaffold(
            P=3,
            theta_dim=4,
            rhs_fn=rhs_3_torch,
            state_names=["A", "J", "M"],
            param_names=["kf1", "kf2", "kr1", "kr2"],
        )

    if name == "reduced4":
        return CustomRHSScaffold(
            P=4,
            theta_dim=6,
            rhs_fn=rhs_4_torch,
            state_names=["A", "G", "J", "M"],
            param_names=["kf1", "kf2", "kf3", "kr1", "kr2", "kr3"],
        )

    if name == "reduced5":
        return CustomRHSScaffold(
            P=5,
            theta_dim=8,
            rhs_fn=rhs_5_torch,
            state_names=["A", "D", "G", "J", "M"],
            param_names=["kf1", "kf2", "kf3", "kf4", "kr1", "kr2", "kr3", "kr4"],
        )

    if name == "reduced6":
        return CustomRHSScaffold(
            P=6,
            theta_dim=10,
            rhs_fn=rhs_6_torch,
            state_names=["A", "B", "D", "G", "J", "M"],
            param_names=["kfAB", "kfBD", "kfDG", "kfGJ", "kfJM", "krAB", "krBD", "krDG", "krGJ", "krJM"],
        )

    if name == "reduced7":
        return CustomRHSScaffold(
            P=7,
            theta_dim=11,
            rhs_fn=rhs_7_torch,
            state_names=["A", "D", "G", "J", "K", "L", "M"],
            param_names=["kfAD", "kfDG", "kfGJ", "kf10", "kf11", "kf12", "krAD", "krDG", "krGJ", "kr11", "kr12"],
        )

    if name == "reduced8":
        return CustomRHSScaffold(
            P=8,
            theta_dim=12,
            rhs_fn=rhs_8_torch,
            state_names=["A", "D", "G", "H", "I", "J", "L", "M"],
            param_names=["kfAD", "kfDG", "kf7", "kf8", "kf9", "kf10", "kf12", "krAD", "krDG", "kr7", "kr9", "kr12"],
        )

    if name == "reduced9":
        return CustomRHSScaffold(
            P=9,
            theta_dim=14,
            rhs_fn=rhs_9_torch,
            state_names=["A", "D", "G", "H", "I", "J", "K", "L", "M"],
            param_names=[
                "kfAD", "kfDG", "kf7", "kf8", "kf9", "kf10", "kf11", "kf12",
                "krAD", "krDG", "kr7", "kr9", "kr11", "kr12",
            ],
        )

    if name == "reduced10":
        return CustomRHSScaffold(
            P=10,
            theta_dim=14,
            rhs_fn=rhs_10_torch,
            state_names=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            param_names=["kf1", "kf2", "kf3", "kf4", "kf5", "kf6", "kf7", "kf8", "kf9", "kr1", "kr3", "kr5", "kr7", "kr9"],
        )

    if name == "reduced11":
        return CustomRHSScaffold(
            P=11,
            theta_dim=15,
            rhs_fn=rhs_11_torch,
            state_names=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
            param_names=["kf1", "kf2", "kf3", "kf4", "kf5", "kf6", "kf7", "kf8", "kf9", "kf10", "kr1", "kr3", "kr5", "kr7", "kr9"],
        )

    if name == "reduced12":
        return CustomRHSScaffold(
            P=12,
            theta_dim=17,
            rhs_fn=rhs_12_torch,
            state_names=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
            param_names=["kf1", "kf2", "kf3", "kf4", "kf5", "kf6", "kf7", "kf8", "kf9", "kf10", "kf11", "kr1", "kr3", "kr5", "kr7", "kr9", "kr11"],
        )

    if name == "full13":
        return CustomRHSScaffold(
            P=13,
            theta_dim=19,
            rhs_fn=rhs_13_torch,
            state_names=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
            param_names=[
                "kf1", "kf2", "kf3", "kf4", "kf5", "kf6", "kf7", "kf8", "kf9", "kf10", "kf11", "kf12",
                "kr1", "kr3", "kr5", "kr7", "kr9", "kr11", "kr12",
            ],
        )

    raise ValueError(f"Unknown scaffold: {name}")