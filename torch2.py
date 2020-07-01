import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

N, DIn, H, DOut = 64, 1000, 100, 10

x = torch.randn(N, DIn)
y = torch.randn(N, DOut)

# model = nn.modules.Sequential(
#     nn.Linear(DIn, H),
#     nn.ReLU(),
#     nn.Linear(H, DOut)
# )

class twoLayerModel(nn.Module):
    def __init__(self,DIn,H,DOut):
        super(twoLayerModel,self).__init__()
        self.Linear1 = nn.Linear(DIn, H)
        self.Linear2 = nn.Linear(H, DOut)

    def forward(self,x):
        y_pred = self.Linear2(self.Linear1(x).clamp(min=0))
        return y_pred


# nn.init.normal_(model[0].weight)
# nn.init.normal_(model[2].weight) 
model = twoLayerModel(DIn, H, DOut)

Loss_fn = nn.MSELoss(reduction="sum")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for it in range(500):
    # forward pass
    y_pred = model(x)

    # compute loss
    loss = Loss_fn(y_pred,y)
    print(it, loss.item())

    # backword pass
    # compute gradient
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
