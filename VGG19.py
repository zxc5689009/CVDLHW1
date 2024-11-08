import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_model_summary import summary
from torchvision.models import vgg
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
# hyper-parameter
BATCH_SIZE = 32
EPOCH = 100
LR = 1e-3

# Dataset from torchvision
to_tensor = transforms.ToTensor()
train_set = CIFAR10(root="E:\\Nick\\Downloads\\cvdl_hw1\\cvdl_hw1\\Dataset_CvDl_Hw1\\Q5_image\\Q5_1", train=True, transform=to_tensor, download=True)
test_set = CIFAR10(root="E:\\Nick\\Downloads\\cvdl_hw1\\cvdl_hw1\\Dataset_CvDl_Hw1\\Q5_image\\Q5_1\\Q5_4", train=False, transform=to_tensor, download=True)


# data loader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE * 2)

# model from torchvision
class VGG_w_cls(nn.Module):
    def __init__(self):
        super(VGG_w_cls, self).__init__()
        self.m = nn.Sequential(
            vgg.vgg19_bn(),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        return self.m(x)

model = VGG_w_cls()
print(model)
best_acc = 0.0
best_epoch = -1
best_model_path = "model/best_model.pth"

# optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
loss_func = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(device)

# train
train_loss_list = []
train_acc_list = []
model.train()

if not os.path.exists('model'):
    os.makedirs('model')

for epoch in trange(EPOCH):
    train_total_loss = 0.0
    correct = 0
    for data in train_loader:
        imgs, labels = [t.to(device) for t in data]

        optimizer.zero_grad()

        outputs = model(imgs)

        outputs_max = torch.argmax(outputs, axis = 1)
        correct += (outputs_max == labels).sum().item()

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()
    
    train_acc = correct / len(train_loader.dataset)
    train_acc_list += [train_acc]
    train_total_loss /= len(train_loader.dataset)
    train_loss_list += [train_total_loss]
    print(train_total_loss,train_acc)
    torch.save(model.state_dict(), f"model/e_{epoch}")

# test
model = VGG_w_cls()
test_loss_list = []
test_acc_list = []
for epoch in trange(EPOCH):
    test_total_loss = 0.0
    model.load_state_dict(torch.load(f"model/e_{epoch}", map_location=torch.device(device)))
    model = model.to(device)

    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = [t.to(device) for t in data]

            outputs = model(imgs)
            loss = loss_func(outputs, labels)
            outputs_max = torch.argmax(outputs, axis = 1)
            correct += (outputs_max == labels).sum().item()
            test_total_loss += loss.item()

        test_acc = correct / len(test_loader.dataset)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
        test_acc_list += [test_acc]
        test_total_loss /= len(test_loader.dataset)
        test_loss_list += [test_total_loss]      
        print(test_acc)

print(f"epoch {best_epoch} is best its accuracy is {best_acc}")
torch.save(model.state_dict(),best_model_path)

import pickle
with open('train_loss.pkl', 'wb') as f:
    pickle.dump(train_loss_list, f)
with open('train_acc.pkl', 'wb') as f:
    pickle.dump(train_acc_list, f)
with open('test_loss.pkl', 'wb') as f:
    pickle.dump(test_loss_list, f)
with open('test_acc.pkl', 'wb') as f:
    pickle.dump(test_acc_list, f)