"""Glycolysis ground-truth oracle — data-generation accessor.

The glycolysis scaffold *class* definitions now live centrally in
``scaffolds.py`` alongside every other scaffold. This module is the
generation-side entry point: it exposes those scaffolds (used as the
ground-truth ODE oracle by ``create_dataset.py``) via
``get_glycolysis_scaffold`` and a small name → instance registry.
"""
from __future__ import annotations

from typing import Dict

from scaffolds import (
    MechanisticScaffold,
    GlycolysisOracle22Scaffold,
    GlycolysisReduced12Scaffold,
    GlycolysisReduced8Scaffold,
    GlycolysisReduced4Scaffold,
)

SCAFFOLDS: Dict[str, MechanisticScaffold] = {
    "glycolysis_oracle22":  GlycolysisOracle22Scaffold(),
    "glycolysis_reduced12": GlycolysisReduced12Scaffold(),
    "glycolysis_reduced8":  GlycolysisReduced8Scaffold(),
    "glycolysis_reduced4":  GlycolysisReduced4Scaffold(),
}


def get_glycolysis_scaffold(name: str) -> MechanisticScaffold:
    if name not in SCAFFOLDS:
        valid = ", ".join(sorted(SCAFFOLDS.keys()))
        raise KeyError(f"Unknown scaffold '{name}'. Valid options are: {valid}")
    return SCAFFOLDS[name]


if __name__ == "__main__":
    import torch
    for name, scaffold in SCAFFOLDS.items():
        y = torch.rand(4, scaffold.P)
        theta = torch.rand(4, scaffold.theta_dim)
        dy = scaffold(y, theta)
        print(f"{name}: P={scaffold.P} theta_dim={scaffold.theta_dim} dy={tuple(dy.shape)}")
