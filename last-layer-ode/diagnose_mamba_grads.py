"""
Quick gradient flow check for OdeMambaSSM.
Runs one forward+backward and prints grad norms for SSM internal params.
If A_log.grad is zero/None, gradients are not reaching the SSM.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from models import MODELS
from scaffolds import SCAFFOLDS
from jumps import make_u_to_y_jump

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# minimal dummy data matching mof4 dims: P=4, U=2, K=20
B, K, P, U = 4, 20, 4, 2
y0    = torch.rand(B, P, device=device)
u_seq = torch.rand(B, K, U, device=device)
dt    = torch.ones(B, K, device=device) * 0.1
y_seq = torch.rand(B, K, P, device=device)
obs_idx = torch.arange(P, device=device)

# mof4: control indices [0,1] map to observed dims [0,1] (Base, Mod)
rhs = SCAFFOLDS["mof_synthesis_4"]
u_to_y_jump = make_u_to_y_jump([0, 1], list(range(P)), device=device)
model = MODELS["ode_mamba_ssm"](
    U=U, rhs=rhs, u_to_y_jump=u_to_y_jump,
    hidden=128, num_layers=2, device=device,
).to(device)

model.train()
y_out, th_out, _ = model(y0, u_seq, dt, obs_idx, y_seq=y_seq, teacher_forcing=True)
loss = (y_out - y_seq).pow(2).mean()
loss.backward()

print(f"\n{'='*50}")
print(f"Loss: {loss.item():.6f}")
print(f"\nSSM internal parameter gradients:")
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"  {name:50s}  grad_norm={p.grad.norm().item():.6e}")
    else:
        print(f"  {name:50s}  grad=None  ← NO GRADIENT")
print('='*50)
