
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 12:43:11 2026

@author: bob-van-sluijs
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleEnzymeODE:
    """Standardized wrapper for the ODE model."""
    
    # 1. Define metadata once here
    NAME = "Reduced Reversible Bi-Bi"
    STATES = ["A", "B", "C", "D", "E", "I"]
    PARAMS = ["kcat_f", "kcat_r", "Ka", "Kb", "Kc", "Kd"]
    
    # 2. Define the (low, high) bounds for each parameter in the same order
    PARAM_RANGES = [
        (1.0, 200.0), # kcat_f
        (1.0, 200.0), # kcat_r
        (1.0, 200.0), # Ka
        (1.0, 200.0), # Kb
        (1.0, 200.0), # Kc
        (1.0, 200.0), # Kd
        ]
    
    # 2. The JIT compiled math (Zero overhead, runs at C++ speed)
    @staticmethod
    @torch.jit.script
    def rhs(y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Unpack using the exact same order as above
        A, B, C, D, E, I = y.unbind(dim=-1)
        kcat_f, kcat_r, Ka, Kb, Kc, Kd = k.unbind(dim=-1)
        
        eps: float = 1e-12

        Apos = torch.clamp_min(A, 0.0)
        Bpos = torch.clamp_min(B, 0.0)
        Cpos = torch.clamp_min(C, 0.0)
        Dpos = torch.clamp_min(D, 0.0)
        Epos = torch.clamp_min(E, 0.0)

        Ka = torch.clamp_min(Ka, eps)
        Kb = torch.clamp_min(Kb, eps)
        Kc = torch.clamp_min(Kc, eps)
        Kd = torch.clamp_min(Kd, eps)

        Vf = kcat_f * Epos
        Vr = kcat_r * Epos

        D0 = Ka * Kb
        denom = (
            D0 * (1.0 + Cpos / Kc + Dpos / Kd + (Cpos * Dpos) / (Kc * Kd))
            + (Kb * Apos) * (1.0 + Dpos / Kd)
            + (Ka * Bpos) * (1.0 + Cpos / Kc)
            + (Apos * Bpos)
            + eps
        )

        v = (Vf * Apos * Bpos - Vr * Cpos * Dpos) / denom

        dA = -v 
        dB = -v 
        dC =  v 
        dD =  v 
        dE =  v-v 
        dI =  v-v    
        
        return torch.stack([dA, dB, dC, dD, dE, dI], dim=-1)
