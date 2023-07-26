import torch
from attention import attention

q = torch.randn((4, 48, 22009, 64), dtype=torch.float16, device="cuda", requires_grad=True)
k = torch.randn((4, 48, 22009, 64), dtype=torch.float16, device="cuda", requires_grad=True)
v = torch.randn((4, 48, 22009, 64), dtype=torch.float16, device="cuda", requires_grad=True)
casual=True
sm_scale=1.3

#forward pass
output = attention(q, k, v, casual, sm_scale)

#backward pass
result = output.backward(torch.ones_like(output)).shape

print(f'Result: {result}')
