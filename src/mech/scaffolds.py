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
    
def make_scaffold(name: str) -> CustomRHSScaffold:
    if name == "reduced5":
        return CustomRHSScaffold(
            P=5,
            theta_dim=8,
            rhs_fn=rhs_5_torch,
            state_names=["A","D","G","J","M"],
            param_names=["kf1","kf2","kf3","kf4","kr1","kr2","kr3","kr4"],
        )

    if name == "reduced7":
        return CustomRHSScaffold(
            P=7,
            theta_dim=11,
            rhs_fn=rhs_7_torch,
            state_names=["A","D","G","J","K","L","M"],
            param_names=["kfAD","kfDG","kfGJ","kf10","kf11","kf12","krAD","krDG","krGJ","kr11","kr12"],
        )

    if name == "reduced9":
        return CustomRHSScaffold(
            P=9,
            theta_dim=14,
            rhs_fn=rhs_9_torch,
            state_names=["A","D","G","H","I","J","K","L","M"],
            param_names=["kfAD","kfDG","kf7","kf8","kf9","kf10","kf11","kf12",
                         "krAD","krDG","kr7","kr9","kr11","kr12"],
        )

    raise ValueError(f"Unknown scaffold: {name}")


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