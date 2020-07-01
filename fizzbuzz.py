import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def fizzbuzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 3 == 0:
        return 1
    elif i % 5 == 0:
        return 2
    else:
        return 0

def fizzbuzz_decode(i, pred):
    return [str(i), 'fizz', 'buzz', 'fizzbuzz'][pred]

def helper(i):
    print(fizzbuzz_decode(i, fizzbuzz_encode(i)))

def binary_number(i):
    return np.array([i >> d & 1 for d in range(10)][::-1])

digit_size = 10
# -----------------------------------------


trX = torch.tensor([binary_number(i)for i in range(101,2**digit_size)],dtype=torch.float)
trY = torch.tensor([fizzbuzz_encode(i) for i in range(101,2**digit_size)])
trX.type()
trY.type()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

NUM_HIDDEN = 100
model = nn.Sequential(
    nn.Linear(digit_size,NUM_HIDDEN),
    nn.ReLU(),
    nn.Linear(NUM_HIDDEN,4)
).to(device=device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

BATCH_SIZE = 128

# loop over the dataset multiple times
for epoch in range(10000):
    for start in range(0,len(trX),BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end].cuda()
        batchY = trY[start:end].cuda()

        y_pred = model(batchX)
        loss = loss_fn(y_pred,batchY)

        print('epoch',epoch,loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

testX = torch.tensor([binary_number(i)for i in range(1,101)],dtype=torch.float).cuda()

with torch.no_grad():
    testY = model(testX)

print(testY.max(1)[1])

#testY.max(1)[1].cpu().data.tolist()

predictions = zip(range(1,101),testY.max(1)[1].cpu().data.tolist())

print([fizzbuzz_decode(i,x)for i,x in predictions])

print(trY)

torch.save(model,'model/model1.pkl')

#save and reload

model1 = torch.load('model/model1.pkl')

testX = torch.tensor([binary_number(i)for i in range(1,101)],dtype=torch.float).cuda()

with torch.no_grad():
    testY = model1(testX)

print(testY)

#testY.max(1)[1].cpu().data.tolist()

predictions = zip(range(1,101),testY.max(1)[1].cpu().data.tolist())

print([fizzbuzz_decode(i,x)for i,x in predictions])