from mamba_ssm.modules.mamba_simple import Mamba
import torch

m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
B, D = 2, 64
conv_state = torch.zeros(B, m.d_inner, m.d_conv - 1, device='cuda')
ssm_state  = torch.zeros(B, m.d_inner, m.d_state,  device='cuda')

out = 0
for k in range(10):
    x = torch.randn(B, 1, D, device='cuda', requires_grad=True)   # <-- (B, 1, D)
    z, conv_state, ssm_state = m.step(x, conv_state, ssm_state)
    out = out + z.sum()

out.backward()
print("A_log.grad:", m.A_log.grad.norm().item() if m.A_log.grad is not None else "None")
print("conv_state.requires_grad:", conv_state.requires_grad)
print("ssm_state.requires_grad:",  ssm_state.requires_grad)