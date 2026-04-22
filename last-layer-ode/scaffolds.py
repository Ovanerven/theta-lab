import torch
import torch.nn as nn
import math

class MechanisticScaffold(nn.Module):
    def __init__(self, P: int, theta_dim: int):
        super().__init__()
        self.P = int(P)
        self.theta_dim = int(theta_dim)
        self.state_names: list[str] = []
        # Per-parameter bounds — set by subclasses. None means use scalar fallback.
        self.theta_lo_vec: "list[float] | None" = None
        self.theta_hi_vec: "list[float] | None" = None

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Reduced2Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=2, theta_dim=2)
        self.state_names = ["A", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, M = y.unbind(dim=-1)
        kf, kr = theta.unbind(dim=-1)
        dA = -kf * A + kr * M
        dM =  kf * A - kr * M
        return torch.stack((dA, dM), dim=-1)


class Reduced3Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=3, theta_dim=4)
        self.state_names = ["A", "J", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, J, M = y.unbind(dim=-1)
        kf1, kf2, kr1, kr2 = theta.unbind(dim=-1)
        dA = -kf1 * A + kr1 * J
        dJ =  kf1 * A - kr1 * J - kf2 * J + kr2 * M
        dM =  kf2 * J - kr2 * M
        return torch.stack((dA, dJ, dM), dim=-1)


class Reduced4Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=4, theta_dim=6)
        self.state_names = ["A", "G", "J", "L"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, G, J, L = y.unbind(dim=-1)
        kf1, kf2, kf3, kr1, kr2, kr3 = theta.unbind(dim=-1)
        dA = -kf1 * A + kr1 * G
        dG =  kf1 * A - kr1 * G - kf2 * G + kr2 * J
        dJ =  kf2 * G - kr2 * J - kf3 * J + kr3 * L
        dL =  kf3 * J - kr3 * L
        return torch.stack((dA, dG, dJ, dL), dim=-1)


class Reduced4AEIMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=4, theta_dim=6)
        self.state_names = ["A", "E", "I", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, E, I, M = y.unbind(dim=-1)
        kf1, kf2, kf3, kr1, kr2, kr3 = theta.unbind(dim=-1)
        dA = -kf1 * A + kr1 * E
        dE =  kf1 * A - kr1 * E - kf2 * E + kr2 * I
        dI =  kf2 * E - kr2 * I - kf3 * I + kr3 * M
        dM =  kf3 * I - kr3 * M
        return torch.stack((dA, dE, dI, dM), dim=-1)


class Reduced5Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=5, theta_dim=8)
        self.state_names = ["A", "D", "G", "J", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, D, G, J, M = y.unbind(dim=-1)
        kf1, kf2, kf3, kf4, kr1, kr2, kr3, kr4 = theta.unbind(dim=-1)
        dA = -kf1 * A + kr1 * D
        dD =  kf1 * A - kr1 * D - kf2 * D + kr2 * G
        dG =  kf2 * D - kr2 * G - kf3 * G + kr3 * J
        dJ =  kf3 * G - kr3 * J - kf4 * J + kr4 * M
        dM =  kf4 * J - kr4 * M
        return torch.stack((dA, dD, dG, dJ, dM), dim=-1)


class Reduced6Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=6, theta_dim=10)
        self.state_names = ["A", "B", "D", "G", "J", "L"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, B, D, G, J, L = y.unbind(dim=-1)
        kfAB, kfBD, kfDG, kfGJ, kfJL, krAB, krBD, krDG, krGJ, krJL = theta.unbind(dim=-1)
        dA = -kfAB * A + krAB * B
        dB =  kfAB * A - krAB * B - kfBD * B + krBD * D
        dD =  kfBD * B - krBD * D - kfDG * D + krDG * G
        dG =  kfDG * D - krDG * G - kfGJ * G + krGJ * J
        dJ =  kfGJ * G - krGJ * J - kfJL * J + krJL * L
        dL =  kfJL * J - krJL * L
        return torch.stack((dA, dB, dD, dG, dJ, dL), dim=-1)


class Reduced6ADGJLMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=6, theta_dim=10)
        self.state_names = ["A", "D", "G", "J", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, D, G, J, L, M = y.unbind(dim=-1)
        kfAD, kfDG, kfGJ, kfJL, kfLM, krAD, krDG, krGJ, krJL, krLM = theta.unbind(dim=-1)
        dA = -kfAD * A + krAD * D
        dD =  kfAD * A - krAD * D - kfDG * D + krDG * G
        dG =  kfDG * D - krDG * G - kfGJ * G + krGJ * J
        dJ =  kfGJ * G - krGJ * J - kfJL * J + krJL * L
        dL =  kfJL * J - krJL * L - kfLM * L + krLM * M
        dM =  kfLM * L - krLM * M
        return torch.stack((dA, dD, dG, dJ, dL, dM), dim=-1)


class Reduced6AGHILMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=6, theta_dim=9)
        self.state_names = ["A", "G", "H", "I", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, G, H, I, L, M = y.unbind(dim=-1)
        kfAG, kfGH, kfHI, kfIL, kfLM, krAG, krGH, krIL, krLM = theta.unbind(dim=-1)
        dA = -kfAG * A + krAG * G
        dG =  kfAG * A - krAG * G - kfGH * G + krGH * H
        dH =  kfGH * G - krGH * H - kfHI * H
        dI =  kfHI * H - kfIL * I + krIL * L
        dL =  kfIL * I - krIL * L - kfLM * L + krLM * M
        dM =  kfLM * L - krLM * M
        return torch.stack((dA, dG, dH, dI, dL, dM), dim=-1)


class Reduced6DGHILMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=6, theta_dim=9)
        self.state_names = ["D", "G", "H", "I", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        D, G, H, I, L, M = y.unbind(dim=-1)
        kfDG, kfGH, kfHI, kfIL, kfLM, krDG, krGH, krIL, krLM = theta.unbind(dim=-1)
        dD = -kfDG * D + krDG * G
        dG =  kfDG * D - krDG * G - kfGH * G + krGH * H
        dH =  kfGH * G - krGH * H - kfHI * H
        dI =  kfHI * H - kfIL * I + krIL * L
        dL =  kfIL * I - krIL * L - kfLM * L + krLM * M
        dM =  kfLM * L - krLM * M
        return torch.stack((dD, dG, dH, dI, dL, dM), dim=-1)


class Reduced6ABCDLMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=6, theta_dim=9)
        self.state_names = ["A", "B", "C", "D", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, B, C, D, L, M = y.unbind(dim=-1)
        kfAB, kfBC, kfCD, kfDL, kfLM, krAB, krCD, krDL, krLM = theta.unbind(dim=-1)
        dA = -kfAB * A + krAB * B
        dB =  kfAB * A - krAB * B - kfBC * B
        dC =  kfBC * B - kfCD * C + krCD * D
        dD =  kfCD * C - krCD * D - kfDL * D + krDL * L
        dL =  kfDL * D - krDL * L - kfLM * L + krLM * M
        dM =  kfLM * L - krLM * M
        return torch.stack((dA, dB, dC, dD, dL, dM), dim=-1)


class Reduced6ACFHKMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=6, theta_dim=10)
        self.state_names = ["A", "C", "F", "H", "K", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, C, F, H, K, M = y.unbind(dim=-1)
        kf1, kf2, kf3, kf4, kf5, kr1, kr2, kr3, kr4, kr5 = theta.unbind(dim=-1)
        dA = -kf1 * A + kr1 * C
        dC =  kf1 * A - kr1 * C - kf2 * C + kr2 * F
        dF =  kf2 * C - kr2 * F - kf3 * F + kr3 * H
        dH =  kf3 * F - kr3 * H - kf4 * H + kr4 * K
        dK =  kf4 * H - kr4 * K - kf5 * K + kr5 * M
        dM =  kf5 * K - kr5 * M
        return torch.stack((dA, dC, dF, dH, dK, dM), dim=-1)


class Reduced7Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=7, theta_dim=11)
        self.state_names = ["A", "D", "G", "J", "K", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, D, G, J, K, L, M = y.unbind(dim=-1)
        kfAD, kfDG, kfGJ, kf10, kf11, kf12, krAD, krDG, krGJ, kr11, kr12 = theta.unbind(dim=-1)
        dA = -kfAD * A + krAD * D
        dD =  kfAD * A - krAD * D - kfDG * D + krDG * G
        dG =  kfDG * D - krDG * G - kfGJ * G + krGJ * J
        dJ =  kfGJ * G - krGJ * J - kf10 * J
        dK =  kf10 * J - kf11 * K + kr11 * L
        dL =  kf11 * K - kr11 * L - kf12 * L + kr12 * M
        dM =  kf12 * L - kr12 * M
        return torch.stack((dA, dD, dG, dJ, dK, dL, dM), dim=-1)


class Reduced8Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=8, theta_dim=12)
        self.state_names = ["A", "D", "G", "H", "I", "J", "K", "L"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
        return torch.stack((dA, dD, dG, dH, dI, dJ, dK, dL), dim=-1)


class Reduced8ACEGIJLMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=8, theta_dim=14)
        self.state_names = ["A", "C", "E", "G", "I", "J", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, C, E, G, I, J, L, M = y.unbind(dim=-1)
        (
            kf1, kf2, kf3, kf4, kf5, kf6, kf7,
            kr1, kr2, kr3, kr4, kr5, kr6, kr7
        ) = theta.unbind(dim=-1)
        dA = -kf1 * A + kr1 * C
        dC =  kf1 * A - kr1 * C - kf2 * C + kr2 * E
        dE =  kf2 * C - kr2 * E - kf3 * E + kr3 * G
        dG =  kf3 * E - kr3 * G - kf4 * G + kr4 * I
        dI =  kf4 * G - kr4 * I - kf5 * I + kr5 * J
        dJ =  kf5 * I - kr5 * J - kf6 * J + kr6 * L
        dL =  kf6 * J - kr6 * L - kf7 * L + kr7 * M
        dM =  kf7 * L - kr7 * M
        return torch.stack((dA, dC, dE, dG, dI, dJ, dL, dM), dim=-1)


class Reduced8ADGHJKLMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=8, theta_dim=14)
        self.state_names = ["A", "D", "G", "H", "J", "K", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, D, G, H, J, K, L, M = y.unbind(dim=-1)
        (
            kf1, kf2, kf3, kf4, kf5, kf6, kf7,
            kr1, kr2, kr3, kr4, kr5, kr6, kr7
        ) = theta.unbind(dim=-1)
        dA = -kf1 * A + kr1 * D
        dD =  kf1 * A - kr1 * D - kf2 * D + kr2 * G
        dG =  kf2 * D - kr2 * G - kf3 * G + kr3 * H
        dH =  kf3 * G - kr3 * H - kf4 * H + kr4 * J
        dJ =  kf4 * H - kr4 * J - kf5 * J + kr5 * K
        dK =  kf5 * J - kr5 * K - kf6 * K + kr6 * L
        dL =  kf6 * K - kr6 * L - kf7 * L + kr7 * M
        dM =  kf7 * L - kr7 * M
        return torch.stack((dA, dD, dG, dH, dJ, dK, dL, dM), dim=-1)


class Reduced9Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=9, theta_dim=14)
        self.state_names = ["A", "D", "G", "H", "I", "J", "K", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
        return torch.stack((dA, dD, dG, dH, dI, dJ, dK, dL, dM), dim=-1)


class Reduced10Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=10, theta_dim=14)
        self.state_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
        return torch.stack((dA, dB, dC, dD, dE, dF, dG, dH, dI, dL), dim=-1)


class Reduced10WithMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=10, theta_dim=14)
        self.state_names = ["A", "D", "F", "G", "H", "I", "J", "K", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, D, F, G, H, I, J, K, L, M = y.unbind(dim=-1)
        (
            kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9,
            kr1, kr4, kr6, kr8, kr9
        ) = theta.unbind(dim=-1)
        dA =  -kf1 * A + kr1 * D
        dD =   kf1 * A - kr1 * D - kf2 * D
        dF =   kf2 * D - kf3 * F
        dG =   kf3 * F - kf4 * G + kr4 * H
        dH =   kf4 * G - kr4 * H - kf5 * H
        dI =   kf5 * H - kf6 * I + kr6 * J
        dJ =   kf6 * I - kr6 * J - kf7 * J
        dK =   kf7 * J - kf8 * K + kr8 * L
        dL =   kf8 * K - kr8 * L - kf9 * L + kr9 * M
        dM =   kf9 * L - kr9 * M
        return torch.stack((dA, dD, dF, dG, dH, dI, dJ, dK, dL, dM), dim=-1)


class Reduced11Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=11, theta_dim=16)
        self.state_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
        return torch.stack((dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ, dM), dim=-1)


class Reduced11WithMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=11, theta_dim=16)
        self.state_names = ["A", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, D, E, F, G, H, I, J, K, L, M = y.unbind(dim=-1)
        (
            kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kf10,
            kr1, kr3, kr5, kr7, kr9, kr10
        ) = theta.unbind(dim=-1)
        dA =  -kf1 * A + kr1 * D
        dD =   kf1 * A - kr1 * D  - kf2 * D
        dE =   kf2 * D  - kf3 * E  + kr3 * F
        dF =   kf3 * E  - kr3 * F  - kf4 * F
        dG =   kf4 * F  - kf5 * G  + kr5 * H
        dH =   kf5 * G  - kr5 * H  - kf6 * H
        dI =   kf6 * H  - kf7 * I  + kr7 * J
        dJ =   kf7 * I  - kr7 * J  - kf8 * J
        dK =   kf8 * J  - kf9 * K  + kr9 * L
        dL =   kf9 * K  - kr9 * L  - kf10 * L + kr10 * M
        dM =   kf10 * L - kr10 * M
        return torch.stack((dA, dD, dE, dF, dG, dH, dI, dJ, dK, dL, dM), dim=-1)


class Reduced12Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=12, theta_dim=17)
        self.state_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
        return torch.stack((dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ, dK, dL), dim=-1)


class Reduced12WithMScaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=12, theta_dim=17)
        self.state_names = ["A", "B", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, B, D, E, F, G, H, I, J, K, L, M = y.unbind(dim=-1)
        (
            kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kf10, kf11,
            kr1, kr4, kr6, kr8, kr10, kr11
        ) = theta.unbind(dim=-1)
        dA =  -kf1 * A + kr1 * B
        dB =   kf1 * A - kr1 * B  - kf2 * B
        dD =   kf2 * B             - kf3 * D
        dE =   kf3 * D  - kf4 * E  + kr4 * F
        dF =   kf4 * E  - kr4 * F  - kf5 * F
        dG =   kf5 * F  - kf6 * G  + kr6 * H
        dH =   kf6 * G  - kr6 * H  - kf7 * H
        dI =   kf7 * H  - kf8 * I  + kr8 * J
        dJ =   kf8 * I  - kr8 * J  - kf9 * J
        dK =   kf9 * J  - kf10 * K + kr10 * L
        dL =   kf10 * K - kr10 * L - kf11 * L + kr11 * M
        dM =   kf11 * L - kr11 * M
        return torch.stack((dA, dB, dD, dE, dF, dG, dH, dI, dJ, dK, dL, dM), dim=-1)


class Full13Scaffold(MechanisticScaffold):
    def __init__(self):
        super().__init__(P=13, theta_dim=19)
        self.state_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
        return torch.stack((dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ, dK, dL, dM), dim=-1)



class MOFSynthesis12Scaffold(MechanisticScaffold):
    """
    Full 12-state MOF synthesis scaffold. Preserves all mechanistic structure
    from MOF_model.py; all 16 kinetic constants are learned as θ(t).

    States (12): Met, LigH, Lig_minus, H_plus, Base, Mod,
                 SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C
    Control inputs (bolused): Base (idx 4), Mod (idx 5)

    Parameters θ (16):
      0  k_deprot  : LigH + Base -> Lig_minus deprotonation rate
      1  k_prot    : Lig_minus + H+ -> LigH reprotonation rate
      2  k_oli     : Met^a * Lig_minus^b -> SBU oligomerization rate
      3  k_cap     : SBU + Mod -> SBU_capped capping rate
      4  k_uncap   : SBU_capped -> SBU + Mod uncapping rate
      5  K_I       : modulator inhibition constant for crystalline growth
      6  knuc_A    : amorphous nucleation prefactor
      7  kgro_A    : amorphous growth rate
      8  kagg_A    : amorphous aggregation rate
      9  n_A       : SBU exponent for amorphous nucleation
      10 knuc_C    : crystalline nucleation prefactor
      11 kgro_C    : crystalline growth rate
      12 kagg_C    : crystalline aggregation rate
      13 n_C       : SBU exponent for crystalline nucleation
      14 a         : Met exponent in oligomerization
      15 b         : Lig_minus exponent in oligomerization
    """
    def __init__(self):
        super().__init__(P=12, theta_dim=16)
        self.state_names = [
            "Met", "LigH", "Lig_minus", "H_plus",
            "Base", "Mod", "SBU", "SBU_capped",
            "Nuc_A", "Am", "Nuc_C", "MOF_C",
        ]
        # Per-parameter bounds (true values: k_deprot=5, k_prot=1, k_oli=3, k_cap=2,
        # k_uncap=0.5, K_I=0.1, knuc_A=10, kgro_A=1, kagg_A=1, n_A=3,
        # knuc_C=0.5, kgro_C=4, kagg_C=1, n_C=1.5, a=1, b=1)
        self.theta_lo_vec = [0.1,  0.01, 0.01, 0.01, 0.001, 0.001,
                             0.1,  0.01, 0.01, 0.5,
                             0.001, 0.01, 0.01, 0.5,
                             0.1, 0.1]
        self.theta_hi_vec = [50.0, 20.0, 30.0, 20.0, 10.0, 2.0,
                             100.0, 20.0, 20.0, 10.0,
                             20.0, 50.0, 20.0, 8.0,
                             5.0, 5.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        (
            Met, LigH, Lig_minus, H_plus,
            Base, Mod, SBU, SBU_capped,
            Nuc_A, Am, Nuc_C, MOF_C,
        ) = y.unbind(dim=-1)
        (
            k_deprot, k_prot, k_oli, k_cap, k_uncap, K_I,
            knuc_A, kgro_A, kagg_A, n_A,
            knuc_C, kgro_C, kagg_C, n_C,
            a, b,
        ) = theta.unbind(dim=-1)

        Met_p         = torch.clamp_min(Met, 0.0)
        LigH_p        = torch.clamp_min(LigH, 0.0)
        Lig_minus_p   = torch.clamp_min(Lig_minus, 0.0)
        H_plus_p      = torch.clamp_min(H_plus, 0.0)
        Base_p        = torch.clamp_min(Base, 0.0)
        Mod_p         = torch.clamp_min(Mod, 0.0)
        SBU_p         = torch.clamp_min(SBU, 0.0)
        SBU_capped_p  = torch.clamp_min(SBU_capped, 0.0)
        Nuc_A_p       = torch.clamp_min(Nuc_A, 0.0)
        Am_p          = torch.clamp_min(Am, 0.0)
        Nuc_C_p       = torch.clamp_min(Nuc_C, 0.0)
        MOF_C_p       = torch.clamp_min(MOF_C, 0.0)

        k_deprot = torch.clamp_min(k_deprot, 0.0)
        k_prot   = torch.clamp_min(k_prot,   0.0)
        k_oli    = torch.clamp_min(k_oli,    0.0)
        k_cap    = torch.clamp_min(k_cap,    0.0)
        k_uncap  = torch.clamp_min(k_uncap,  0.0)
        K_I      = torch.clamp_min(K_I,      1e-6)
        knuc_A   = torch.clamp_min(knuc_A,   0.0)
        kgro_A   = torch.clamp_min(kgro_A,   0.0)
        kagg_A   = torch.clamp_min(kagg_A,   0.0)
        n_A      = torch.clamp_min(n_A,      1e-6)
        knuc_C   = torch.clamp_min(knuc_C,   0.0)
        kgro_C   = torch.clamp_min(kgro_C,   0.0)
        kagg_C   = torch.clamp_min(kagg_C,   0.0)
        n_C      = torch.clamp_min(n_C,      1e-6)
        a        = torch.clamp_min(a,        1e-6)
        b        = torch.clamp_min(b,        1e-6)

        r_deprot = k_deprot * LigH_p * Base_p
        r_prot   = k_prot * Lig_minus_p * H_plus_p
        r_oli    = k_oli * (Met_p + 1e-8).pow(a) * (Lig_minus_p + 1e-8).pow(b)
        r_cap    = k_cap * SBU_p * Mod_p
        r_uncap  = k_uncap * SBU_capped_p
        r_nuc_A  = knuc_A * (SBU_p + 1e-8).pow(n_A)
        r_nuc_C  = knuc_C * (SBU_p + 1e-8).pow(n_C)
        r_gro_A  = kgro_A * SBU_p * Am_p
        r_agg_A  = kagg_A * Nuc_A_p.pow(2.0)
        inhib    = K_I / (K_I + Mod_p + 1e-6)
        r_gro_C  = kgro_C * SBU_p * MOF_C_p * inhib
        r_agg_C  = kagg_C * Nuc_C_p.pow(2.0)

        dMet        = -r_oli
        dLigH       = -r_deprot + r_prot
        dLig_minus  =  r_deprot - r_prot - r_oli
        dH_plus     =  r_deprot - r_prot + r_oli
        dBase       = -r_deprot
        dMod        = -r_cap + r_uncap
        dSBU        =  r_oli - r_cap + r_uncap - r_nuc_A - r_gro_A - r_nuc_C - r_gro_C
        dSBU_capped =  r_cap - r_uncap
        dNuc_A      =  r_nuc_A - r_agg_A
        dAm         =  r_agg_A + r_gro_A
        dNuc_C      =  r_nuc_C - r_agg_C
        dMOF_C      =  r_agg_C + r_gro_C

        return torch.stack((
            dMet, dLigH, dLig_minus, dH_plus,
            dBase, dMod, dSBU, dSBU_capped,
            dNuc_A, dAm, dNuc_C, dMOF_C,
        ), dim=-1)


class MOFSynthesis8Scaffold(MechanisticScaffold):
    """
    8-state MOF synthesis scaffold. Collapses the four deprotonation species
    (Met, LigH, Lig_minus, H_plus) into an effective SBU production term driven
    by Base. Retains SBU_capped explicitly so Mod dynamics are exact. Includes
    cooperative nucleation exponents n_A, n_C as learned θ parameters.

    States (8): Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C
    Control inputs (bolused): Base (idx 0), Mod (idx 1)

    Parameters θ (13):
      0  k_base_decay : effective Base consumption rate
      1  k_oli_eff    : effective SBU production rate from Base
      2  k_cap        : SBU + Mod -> SBU_capped capping rate
      3  k_uncap      : SBU_capped -> SBU + Mod uncapping rate
      4  K_I          : modulator inhibition constant
      5  knuc_A       : amorphous nucleation prefactor
      6  kgro_A       : amorphous growth rate
      7  kagg_A       : amorphous aggregation rate
      8  n_A          : SBU exponent for amorphous nucleation
      9  knuc_C       : crystalline nucleation prefactor
      10 kgro_C       : crystalline growth rate
      11 kagg_C       : crystalline aggregation rate
      12 n_C          : SBU exponent for crystalline nucleation
    """
    def __init__(self):
        super().__init__(P=8, theta_dim=13)
        self.state_names = [
            "Base", "Mod", "SBU", "SBU_capped",
            "Nuc_A", "Am", "Nuc_C", "MOF_C",
        ]
        # Per-parameter bounds (k_base_decay, k_oli_eff, k_cap, k_uncap, K_I,
        # knuc_A, kgro_A, kagg_A, n_A, knuc_C, kgro_C, kagg_C, n_C)
        self.theta_lo_vec = [0.1,  0.01, 0.01, 0.001, 0.001,
                             0.1,  0.01, 0.01, 0.5,
                             0.001, 0.01, 0.01, 0.5]
        self.theta_hi_vec = [50.0, 30.0, 20.0, 10.0,  2.0,
                             100.0, 20.0, 20.0, 10.0,
                             20.0,  50.0, 20.0, 8.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C = y.unbind(dim=-1)
        (
            k_base_decay, k_oli_eff, k_cap, k_uncap, K_I,
            knuc_A, kgro_A, kagg_A, n_A,
            knuc_C, kgro_C, kagg_C, n_C,
        ) = theta.unbind(dim=-1)

        Base_p        = torch.clamp_min(Base, 0.0)
        Mod_p         = torch.clamp_min(Mod, 0.0)
        SBU_p         = torch.clamp_min(SBU, 0.0)
        SBU_capped_p  = torch.clamp_min(SBU_capped, 0.0)
        Nuc_A_p       = torch.clamp_min(Nuc_A, 0.0)
        Am_p          = torch.clamp_min(Am, 0.0)
        Nuc_C_p       = torch.clamp_min(Nuc_C, 0.0)
        MOF_C_p       = torch.clamp_min(MOF_C, 0.0)

        k_base_decay = torch.clamp_min(k_base_decay, 0.0)
        k_oli_eff    = torch.clamp_min(k_oli_eff,    0.0)
        k_cap        = torch.clamp_min(k_cap,        0.0)
        k_uncap      = torch.clamp_min(k_uncap,      0.0)
        K_I          = torch.clamp_min(K_I,          1e-6)
        knuc_A       = torch.clamp_min(knuc_A,       0.0)
        kgro_A       = torch.clamp_min(kgro_A,       0.0)
        kagg_A       = torch.clamp_min(kagg_A,       0.0)
        n_A          = torch.clamp_min(n_A,          1e-6)
        knuc_C       = torch.clamp_min(knuc_C,       0.0)
        kgro_C       = torch.clamp_min(kgro_C,       0.0)
        kagg_C       = torch.clamp_min(kagg_C,       0.0)
        n_C          = torch.clamp_min(n_C,          1e-6)

        r_cap    = k_cap * SBU_p * Mod_p
        r_uncap  = k_uncap * SBU_capped_p
        r_nuc_A  = knuc_A * (SBU_p + 1e-8).pow(n_A)
        r_nuc_C  = knuc_C * (SBU_p + 1e-8).pow(n_C)
        r_gro_A  = kgro_A * SBU_p * Am_p
        r_agg_A  = kagg_A * Nuc_A_p.pow(2.0)
        inhib    = K_I / (K_I + Mod_p + 1e-6)
        r_gro_C  = kgro_C * SBU_p * MOF_C_p * inhib
        r_agg_C  = kagg_C * Nuc_C_p.pow(2.0)

        dBase       = -k_base_decay * Base_p
        dMod        = -r_cap + r_uncap
        dSBU        =  k_oli_eff * Base_p - r_cap + r_uncap - r_nuc_A - r_gro_A - r_nuc_C - r_gro_C
        dSBU_capped =  r_cap - r_uncap
        dNuc_A      =  r_nuc_A - r_agg_A
        dAm         =  r_agg_A + r_gro_A
        dNuc_C      =  r_nuc_C - r_agg_C
        dMOF_C      =  r_agg_C + r_gro_C

        return torch.stack((
            dBase, dMod, dSBU, dSBU_capped,
            dNuc_A, dAm, dNuc_C, dMOF_C,
        ), dim=-1)


class MOFSynthesis6Scaffold(MechanisticScaffold):
    """
    6-state MOF synthesis scaffold. Applies two further reductions on top of
    MOFSynthesis8Scaffold: (1) quasi-steady-state on SBU_capped so dMod = 0
    between boluses (net capping flux is zero); (2) fast-nucleation collapse of
    Nuc_A directly into Am. Retains cooperative nucleation exponents n_A, n_C
    as learned θ parameters (advisor recommendation: option b).

    States (6): Base, Mod, SBU, Am, Nuc_C, MOF_C
    Control inputs (bolused): Base (idx 0), Mod (idx 1)

    Parameters θ (10):
      0  k_base_decay : effective Base consumption rate
      1  k_oli_eff    : effective SBU production rate from Base
      2  knuc_A       : amorphous nucleation prefactor (feeds Am directly)
      3  kgro_A       : amorphous growth rate
      4  n_A          : SBU exponent for amorphous nucleation
      5  knuc_C       : crystalline nucleation prefactor
      6  kgro_C       : crystalline growth rate
      7  kagg_C       : crystalline aggregation rate
      8  n_C          : SBU exponent for crystalline nucleation
      9  K_I          : modulator inhibition constant
    """
    def __init__(self):
        super().__init__(P=6, theta_dim=10)
        self.state_names = ["Base", "Mod", "SBU", "Am", "Nuc_C", "MOF_C"]
        # Per-parameter bounds (k_base_decay, k_oli_eff, knuc_A, kgro_A, n_A,
        # knuc_C, kgro_C, kagg_C, n_C, K_I)
        self.theta_lo_vec = [0.1,  0.01, 0.1,  0.01, 0.5,  0.001, 0.01, 0.01, 0.5,  0.001]
        self.theta_hi_vec = [50.0, 30.0, 100.0, 20.0, 10.0, 20.0, 50.0, 20.0, 8.0,  2.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Base, Mod, SBU, Am, Nuc_C, MOF_C = y.unbind(dim=-1)
        (
            k_base_decay, k_oli_eff,
            knuc_A, kgro_A, n_A,
            knuc_C, kgro_C, kagg_C, n_C,
            K_I,
        ) = theta.unbind(dim=-1)

        Base_p  = torch.clamp_min(Base,  0.0)
        Mod_p   = torch.clamp_min(Mod,   0.0)
        SBU_p   = torch.clamp_min(SBU,   0.0)
        Am_p    = torch.clamp_min(Am,    0.0)
        Nuc_C_p = torch.clamp_min(Nuc_C, 0.0)
        MOF_C_p = torch.clamp_min(MOF_C, 0.0)

        k_base_decay = torch.clamp_min(k_base_decay, 0.0)
        k_oli_eff    = torch.clamp_min(k_oli_eff,    0.0)
        knuc_A       = torch.clamp_min(knuc_A,       0.0)
        kgro_A       = torch.clamp_min(kgro_A,       0.0)
        n_A          = torch.clamp_min(n_A,          1e-6)
        knuc_C       = torch.clamp_min(knuc_C,       0.0)
        kgro_C       = torch.clamp_min(kgro_C,       0.0)
        kagg_C       = torch.clamp_min(kagg_C,       0.0)
        n_C          = torch.clamp_min(n_C,          1e-6)
        K_I          = torch.clamp_min(K_I,          1e-6)

        r_nuc_A  = knuc_A * (SBU_p + 1e-8).pow(n_A)
        r_nuc_C  = knuc_C * (SBU_p + 1e-8).pow(n_C)
        r_gro_A  = kgro_A * SBU_p * Am_p
        inhib    = K_I / (K_I + Mod_p + 1e-6)
        r_gro_C  = kgro_C * SBU_p * MOF_C_p * inhib
        r_agg_C  = kagg_C * Nuc_C_p.pow(2.0)

        dBase  = -k_base_decay * Base_p
        dMod   = torch.zeros_like(Base)   # QSS: r_cap == r_uncap between boluses
        dSBU   =  k_oli_eff * Base_p - r_nuc_A - r_gro_A - r_nuc_C - r_gro_C
        dAm    =  r_nuc_A + r_gro_A       # Nuc_A fast: collapses directly into Am
        dNuc_C =  r_nuc_C - r_agg_C
        dMOF_C =  r_agg_C + r_gro_C

        return torch.stack((dBase, dMod, dSBU, dAm, dNuc_C, dMOF_C), dim=-1)


class MOFSynthesis4Scaffold(MechanisticScaffold):
    """
    4-state MOF synthesis scaffold. Most aggressively reduced: no SBU tracked.
    Base acts as proxy for SBU availability; nucleation is linear (no cooperative
    exponent) since SBU is not an explicit state. Mod decays via a slow first-order
    approximation (GRU compensates for the full capping dynamics).

    States (4): Base, Mod, Am, MOF_C
    Control inputs (bolused): Base (idx 0), Mod (idx 1)

    Parameters θ (7):
      0  k_base   : effective Base decay rate
      1  k_mod    : effective Mod decay rate (first-order approximation)
      2  k_nuc_A  : amorphous nucleation rate (linear in Base)
      3  k_gro_A  : amorphous growth rate (Base * Am)
      4  k_nuc_C  : crystalline nucleation rate (linear in Base)
      5  k_gro_C  : crystalline growth rate (Base * MOF_C * inhibition)
      6  K_I      : modulator inhibition constant
    """
    def __init__(self):
        super().__init__(P=4, theta_dim=7)
        self.state_names = ["Base", "Mod", "Am", "MOF_C"]
        # Per-parameter bounds (k_base, k_mod, k_nuc_A, k_gro_A, k_nuc_C, k_gro_C, K_I)
        self.theta_lo_vec = [0.1,  0.001, 0.1,  0.01, 0.001, 0.01, 0.001]
        self.theta_hi_vec = [50.0, 10.0, 100.0, 20.0, 20.0,  50.0, 2.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Base, Mod, Am, MOF_C = y.unbind(dim=-1)
        k_base, k_mod, k_nuc_A, k_gro_A, k_nuc_C, k_gro_C, K_I = theta.unbind(dim=-1)

        Base_p  = torch.clamp_min(Base,  0.0)
        Mod_p   = torch.clamp_min(Mod,   0.0)
        Am_p    = torch.clamp_min(Am,    0.0)
        MOF_C_p = torch.clamp_min(MOF_C, 0.0)

        k_base  = torch.clamp_min(k_base,  0.0)
        k_mod   = torch.clamp_min(k_mod,   0.0)
        k_nuc_A = torch.clamp_min(k_nuc_A, 0.0)
        k_gro_A = torch.clamp_min(k_gro_A, 0.0)
        k_nuc_C = torch.clamp_min(k_nuc_C, 0.0)
        k_gro_C = torch.clamp_min(k_gro_C, 0.0)
        K_I     = torch.clamp_min(K_I,     1e-6)

        inhib  = K_I / (K_I + Mod_p + 1e-6)

        dBase  = -k_base * Base_p
        dMod   = -k_mod * Mod_p
        dAm    =  k_nuc_A * Base_p + k_gro_A * Base_p * Am_p
        dMOF_C =  k_nuc_C * Base_p + k_gro_C * Base_p * MOF_C_p * inhib

        return torch.stack((dBase, dMod, dAm, dMOF_C), dim=-1)


class SingleEnzymeLumpedScaffold(MechanisticScaffold):
    """
    2-state reduced scaffold for the Single Enzyme scenario.

    The full 6-state system is simulated but only A (substrate, idx 0) and
    C (product, idx 2) are observed. The scaffold approximates the dynamics
    with a simple first-order reversible reaction:

        dA_approx = -kf * A + kr * C
        dC_approx =  kf * A - kr * C

    This is structurally wrong in two ways:
      1. The true reaction is bimolecular (rate ∝ A·B); B is hidden
      2. There is no saturation / denominator term

    The neural network must learn time-varying kf(t) and kr(t) to compensate
    for the missing B dependence and the wrong kinetics.

    States (2): S ↔ A (observed substrate), P ↔ C (observed product)
    Control: A-bolus maps to S; B-bolus is a hidden input (seen by the GRU
             via u_seq but not directly reflected in the observed state)
    Parameters θ (2): kf (effective forward rate), kr (effective reverse rate)

    Use with: datasets/single_enzyme_lumped.npz  (--obs-indices 0,2)
    """
    def __init__(self):
        super().__init__(P=2, theta_dim=2)
        self.state_names = ["S", "P"]
        # True rates are kcat_f·E ≈ 10 and kcat_r·E ≈ 2, but with the denominator
        # the effective observed rate is much lower; use wide bounds.
        self.theta_lo_vec = [0.001, 0.001]
        self.theta_hi_vec = [100.0,  50.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        S, P = y.unbind(dim=-1)
        kf, kr = theta.unbind(dim=-1)

        S_p = torch.clamp_min(S, 0.0)
        P_p = torch.clamp_min(P, 0.0)
        kf  = torch.clamp_min(kf, 0.0)
        kr  = torch.clamp_min(kr, 0.0)

        v = kf * S_p - kr * P_p

        dS = -v
        dP =  v

        return torch.stack((dS, dP), dim=-1)


class SingleEnzymeReduced4Scaffold(MechanisticScaffold):
    """
    Reduced 4-state mass-action scaffold for the Single Enzyme scenario.

    The true system uses Reversible Bi-Bi (Michaelis-Menten) kinetics with a
    nonlinear denominator. This scaffold intentionally simplifies to plain
    mass-action, dropping the inert states E and I (which are constant in the
    data: E=1, I=0) and removing the denominator entirely:

        v  = kf * A * B  −  kr * C * D

    The scaffold structure (A+B → C+D reversibly) is topologically correct,
    but the kinetics are wrong. The neural network must learn time-varying
    kf(t) and kr(t) to compensate for the missing saturation terms.

    States (4): A, B, C, D
    Control inputs (bolused): A (idx 0), B (idx 1)
    Parameters θ (2): kf (effective forward rate), kr (effective reverse rate)

    Use with: datasets/single_enzyme_4.npz  (--obs-indices 0,1,2,3)
    Ground-truth Bi-Bi values for reference: kcat_f·E=10, kcat_r·E=2
    """
    def __init__(self):
        super().__init__(P=4, theta_dim=2)
        self.state_names = ["A", "B", "C", "D"]
        # Bounds: true effective forward rate ≈ 10, reverse ≈ 2
        self.theta_lo_vec = [0.01, 0.001]
        self.theta_hi_vec = [200.0, 100.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, B, C, D = y.unbind(dim=-1)
        kf, kr = theta.unbind(dim=-1)

        A_p = torch.clamp_min(A, 0.0)
        B_p = torch.clamp_min(B, 0.0)
        C_p = torch.clamp_min(C, 0.0)
        D_p = torch.clamp_min(D, 0.0)

        kf = torch.clamp_min(kf, 0.0)
        kr = torch.clamp_min(kr, 0.0)

        v = kf * A_p * B_p - kr * C_p * D_p

        dA = -v
        dB = -v
        dC =  v
        dD =  v

        return torch.stack((dA, dB, dC, dD), dim=-1)


class SingleEnzymeScaffold(MechanisticScaffold):
    """
    6-state Reversible Bi-Bi enzyme kinetics scaffold.

    Reaction: A + B <-> C + D  (catalysed by enzyme E, inhibitor I inert)

    States (6): A, B, C, D, E, I
    Control inputs (bolused): A (idx 0), B (idx 1)

    Parameters θ (6):
      0  kcat_f : forward catalytic rate constant
      1  kcat_r : reverse catalytic rate constant
      2  Ka     : Michaelis constant for substrate A
      3  Kb     : Michaelis constant for substrate B
      4  Kc     : Michaelis constant for product C
      5  Kd     : Michaelis constant for product D

    Ground-truth values: kcat_f=10.0, kcat_r=2.0, Ka=2.0, Kb=2.0, Kc=5.0, Kd=5.0
    Dataset: datasets/single_enzyme_6.npz  (--t-span 10 --n-steps 200)
    """
    def __init__(self):
        super().__init__(P=6, theta_dim=6)
        self.state_names = ["A", "B", "C", "D", "E", "I"]
        # Per-parameter bounds: wide enough to contain the true values with room to search
        self.theta_lo_vec = [0.1,  0.01, 0.01, 0.01, 0.01, 0.01]
        self.theta_hi_vec = [100.0, 50.0, 50.0, 50.0, 50.0, 50.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, B, C, D, E, I = y.unbind(dim=-1)
        kcat_f, kcat_r, Ka, Kb, Kc, Kd = theta.unbind(dim=-1)

        eps: float = 1e-12

        A_p = torch.clamp_min(A, 0.0)
        B_p = torch.clamp_min(B, 0.0)
        C_p = torch.clamp_min(C, 0.0)
        D_p = torch.clamp_min(D, 0.0)
        E_p = torch.clamp_min(E, 0.0)

        Ka = torch.clamp_min(Ka, eps)
        Kb = torch.clamp_min(Kb, eps)
        Kc = torch.clamp_min(Kc, eps)
        Kd = torch.clamp_min(Kd, eps)

        Vf = kcat_f * E_p
        Vr = kcat_r * E_p

        D0 = Ka * Kb
        denom = (
            D0 * (1.0 + C_p / Kc + D_p / Kd + (C_p * D_p) / (Kc * Kd))
            + (Kb * A_p) * (1.0 + D_p / Kd)
            + (Ka * B_p) * (1.0 + C_p / Kc)
            + (A_p * B_p)
            + eps
        )

        v = (Vf * A_p * B_p - Vr * C_p * D_p) / denom

        dA = -v
        dB = -v
        dC =  v
        dD =  v
        dE = E * 0.0   # conserved: always zero
        dI = I * 0.0   # inert: always zero

        return torch.stack((dA, dB, dC, dD, dE, dI), dim=-1)

# -----------------------------------------------------------------------------
# 3) JIT‐scripted analytic ODE: This is the simplest model This has to be integrated into same format as the rest of the scaffolds here. 
# -----------------------------------------------------------------------------
# @torch.jit.script
# def _step_integration(
#     m0: torch.Tensor, p0: torch.Tensor,
#     dt: torch.Tensor,
#     VTX: torch.Tensor, KTX: torch.Tensor,
#     dna: torch.Tensor, kdm: torch.Tensor,
#     VTL: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     eps = 1e-8
#     A     = VTX * dna / (KTX + dna + eps)
#     m_inf = A / (kdm + eps)
#     expT  = torch.exp(-kdm * dt)
#     m1    = m_inf + (m0 - m_inf) * expT
#     int_m = m_inf * dt + (m0 - m_inf) * (1.0 - expT) / (kdm + eps)
#     p1    = p0 + VTL * int_m
#     return m1.clamp(min=0.0), p1.clamp(min=0.0), int_m


# class TXTL_mRNAMaturation(MechanisticScaffold):
#     # states:    [R, m, mm, p, pm]  (optionally +DNA as 6th state)
#     # theta:     [lam, VTXmax, kdm, VTLmax, kmt, kmatm]
#     def __init__(self):
#         super().__init__(P=5, theta_dim=6)
#         self.state_names = ["R", "m", "mm", "p", "pm"]
#         self.theta_lo_vec = [1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5]
#         self.theta_hi_vec = [5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3]

#     def forward(self, y, theta, dna):  # or embed DNA as y[:,5]
#         R, m, mm, p, pm = y.unbind(-1)
#         lam, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(-1)
#         dR  = -lam * R
#         dm  = R * VTXmax * dna - (kdm + kmatm) * m
#         dmm = kmatm * m - kdm * mm
#         dp  = R * VTLmax * (m + mm) - kmt * p
#         dpm = kmt * p
#         return torch.stack([dR, dm, dmm, dp, dpm], dim=-1)

class TXTLMaturationDNAScaffold(MechanisticScaffold):
    """
    6-state TXTL scaffold with DNA as an explicit, bolus-driven state.

    The mechanism is the supervisor's `TXTL_mRNAMaturation`, with DNA promoted
    from an exogenous scalar to a latent state so no scaffold-API change is
    needed: the dataset's `u_to_y_jump` routes the "DNA c" (dilution-corrected
    concentration delta) column of u_seq onto state idx 5, and dDNA/dt = 0
    between jumps — so y[..., 5] at step k is exactly cumsum("DNA c") up to k.

    States (6): R (resource pool), m (immature mRNA), mm (mature mRNA,
                observed as Broccoli), p (immature protein),
                pm (mature protein, observed as mCherry / 2), DNA

    Parameters θ (6):
      0  lam    : resource decay rate
      1  VTXmax : transcription rate (per DNA per R)
      2  kdm    : mRNA degradation rate (applies to both m and mm)
      3  VTLmax : translation rate (per total mRNA per R)
      4  kmt    : protein maturation rate (p → pm)
      5  kmatm  : mRNA maturation rate (m → mm)

    Observed indices within P: [2, 4]  (mm=Broccoli, pm=mCherry/2)
    Use with: datasets/real_ivtt_full.npz (layout='full')
    """
    def __init__(self):
        super().__init__(P=6, theta_dim=6)
        self.state_names = ["R", "m", "mm", "p", "pm", "DNA"]
        # Supervisor's log-uniform bounds for TXTL_mRNAMaturation
        self.theta_lo_vec = [1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5]
        self.theta_hi_vec = [5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, m, mm, p, pm, DNA = y.unbind(dim=-1)
        lam, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(dim=-1)

        R_p   = torch.clamp_min(R,   0.0)
        m_p   = torch.clamp_min(m,   0.0)
        mm_p  = torch.clamp_min(mm,  0.0)
        p_p   = torch.clamp_min(p,   0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dR   = -lam * R_p
        dm   = R_p * VTXmax * DNA_p - (kdm + kmatm) * m_p
        dmm  = kmatm * m_p - kdm * mm_p
        dp   = R_p * VTLmax * (m_p + mm_p) - kmt * p_p
        dpm  = kmt * p_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dR, dm, dmm, dp, dpm, dDNA), dim=-1)


class TXTLSimpleDNAScaffold(MechanisticScaffold):
    """
    3-state minimal TXTL scaffold with DNA as an explicit, bolus-driven state.

    The simplest cascade DNA → mm → pm with first-order kinetics. No resource
    pool, no mRNA maturation, no protein maturation — the network must learn
    time-varying θ(t) to compensate for the missing structure.

    States (3): mm (Broccoli), pm (mCherry / 2), DNA

    Parameters θ (3):
      0  k_tx : transcription rate (DNA → mm)
      1  k_tl : translation rate (mm → pm)
      2  kdm  : mRNA degradation rate

    Observed indices within P: [0, 1]  (mm, pm)
    Use with: datasets/real_ivtt_simple.npz (layout='simple')
    """
    def __init__(self):
        super().__init__(P=3, theta_dim=3)
        self.state_names = ["mm", "pm", "DNA"]
        self.theta_lo_vec = [1e-5, 1e-5, 1e-5]
        self.theta_hi_vec = [1e-1, 1e-1, 1e-2]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        mm, pm, DNA = y.unbind(dim=-1)
        k_tx, k_tl, kdm = theta.unbind(dim=-1)

        mm_p  = torch.clamp_min(mm,  0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dmm  = k_tx * DNA_p - kdm * mm_p
        dpm  = k_tl * mm_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dmm, dpm, dDNA), dim=-1)


SCAFFOLDS: dict[str, MechanisticScaffold] = {
    "reduced2":          Reduced2Scaffold(),
    "reduced3":          Reduced3Scaffold(),
    "reduced4":          Reduced4Scaffold(),
    "reduced4_AEIM":     Reduced4AEIMScaffold(),
    "reduced5":          Reduced5Scaffold(),
    "reduced6":          Reduced6Scaffold(),
    "reduced6_ADGJLM":   Reduced6ADGJLMScaffold(),
    "reduced6_AGHILM":   Reduced6AGHILMScaffold(),
    "reduced6_DGHILM":   Reduced6DGHILMScaffold(),
    "reduced6_ABCDLM":   Reduced6ABCDLMScaffold(),
    "reduced6_ACFHKM":   Reduced6ACFHKMScaffold(),
    "reduced7":          Reduced7Scaffold(),
    "reduced8":          Reduced8Scaffold(),
    "reduced8_ACEGIJLM": Reduced8ACEGIJLMScaffold(),
    "reduced8_ADGHJKLM": Reduced8ADGHJKLMScaffold(),
    "reduced9":          Reduced9Scaffold(),
    "reduced10":         Reduced10Scaffold(),
    "reduced10_with_M":  Reduced10WithMScaffold(),
    "reduced11":         Reduced11Scaffold(),
    "reduced11_with_M":  Reduced11WithMScaffold(),
    "reduced12":         Reduced12Scaffold(),
    "reduced12_with_M":  Reduced12WithMScaffold(),
    "full13":            Full13Scaffold(),
    "mof_synthesis_12":  MOFSynthesis12Scaffold(),
    "mof_synthesis_8":   MOFSynthesis8Scaffold(),
    "mof_synthesis_6":   MOFSynthesis6Scaffold(),
    "mof_synthesis_4":   MOFSynthesis4Scaffold(),
    "single_enzyme_6":   SingleEnzymeScaffold(),
    "single_enzyme_4":   SingleEnzymeReduced4Scaffold(),
    "single_enzyme_lumped": SingleEnzymeLumpedScaffold(),
    "txtl_maturation_dna": TXTLMaturationDNAScaffold(),
    "txtl_simple_dna":     TXTLSimpleDNAScaffold(),
}
