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