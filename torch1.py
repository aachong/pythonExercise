import torch
import numpy as np

N, DIn, H, DOut = 64, 1000, 100, 10

X = torch.randn(N, DIn, requires_grad=True)
Y = torch.randn(N, DOut, requires_grad=True)

w1 = torch.randn(DIn, H, requires_grad=True)
w2 = torch.randn(H, DOut, requires_grad=True)

learing_rate = 1e-6

for it in range(500):
    # forward pass
    h = X.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # compute loss
    loss = (y_pred-Y).pow(2).sum()
    print(it, loss.item())

    # backword pass
    # compute gradient
    loss.backward()
    
    with torch.no_grad():
        w1-=learing_rate*w1.grad
        w2-=learing_rate*w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
