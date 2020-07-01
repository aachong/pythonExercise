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


def load_data(digit_size):
    tr_x = np.array([binary_number(i) for i in range(101, 2**digit_size)])
    tr_y = np.array([fizzbuzz_encode(i) for i in range(101, 2**digit_size)])
    return tr_x, tr_y


tr_x, tr_y = load_data(digit_size)

a = np.array([i/2 for i in range(13)])
a.dtype


def get_batch_index(num, batch_size):
    a = np.array(list(range(0, num, batch_size)), dtype=int)
    np.random.shuffle(a)
    ret = []
    for i in a:
        ret.append(list(range(i, min(i+batch_size, num))))
    return ret


def get_batch(num, batch_size, tr_x, tr_y):
    batch_index = get_batch_index(num, batch_size)
    data = []
    for i in batch_index:
        x = np.array([tr_x[j] for j in i])
        y = np.array([tr_y[j] for j in i])
        data.append((x, y))
    return data


train_data = get_batch(len(tr_x), 128, tr_x, tr_y)
train_data[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = nn.Sequential(
    nn.Linear(digit_size,100),
    nn.ReLU(),
    nn.Linear(100,4)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)

def train(data,epochs,model,criterion,optimizer):
    for epoch in range(epochs):
        epoch_loss = 0
        for (it,(x,y)) in enumerate(data):
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).long().to(device)
            preds = model(x)

            loss = criterion(preds,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss
        print(f'Epoch:{epoch},loss:{epoch_loss/len(y)}')

train(train_data,4000,model,criterion,optimizer)

x = torch.tensor([binary_number(i) for i in range(1,101)]).float().to(device)
with torch.no_grad():
    y = model(x)
y = y.max(1)[1].tolist()
print([helper(i) for i in y])
pred = zip(range(1,101),y)
print([fizzbuzz_decode(i,p) for i,p in pred])

a = torch.randn(6,3)
a.max(1)



testX = torch.tensor([binary_number(i)for i in range(1,101)],dtype=torch.float).cuda()

with torch.no_grad():
    testY = model(testX)
