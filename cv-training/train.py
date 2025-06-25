# import torch
# print(torch.cuda.is_available())
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import os
import torch.utils.data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np

class CIFAR10Albumentations(CIFAR10):
    def __init__(self, train=True, transform=None):
        super().__init__(root='./', train=train, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

# hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# # data augmentation and normalization
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, ), (0.5, ))
# ])

# update
train_transform = A.Compose([
    A.PadIfNeeded(min_height=32, min_width=32, border_mode=0, p=1.0),  # border_mode=0 对应 cv2.BORDER_CONSTANT = 常数填充
    A.RandomCrop(height=32, width=32, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Normalize((0.5,)*3, (0.5,)*3),
    ToTensorV2()
])

# # train data need to augmented (rotate, move) to make the model more generalized
# # but test data, no need to do such augmentation, so just normalize
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, ), (0.5, ))
# ])

# update
test_transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# # load cifar dataset
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=False)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

train_dataset = CIFAR10Albumentations(train=True, transform=train_transform)
test_dataset = CIFAR10Albumentations(train=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# load pre-defined ResNet18, and adjust output layer
model = torchvision.models.resnet18(pretrained=False, num_classes=10)
model = model.to(DEVICE)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# training loop
for epoch in range(EPOCHS):
    model.train() # set the model to train mode
    total_loss = 0 # set total loss, the TP and total to 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}') # make the bar, visualize the progress
    for inputs, targets in pbar: # for each bach
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE) # put current batch into tensor?

        optimizer.zero_grad() # why is this?
        outputs = model(inputs) # feed inputs into the model
        loss = criterion(outputs, targets) # compute loss according to labels and predicted
        loss.backward() # backward propagation
        optimizer.step() # what does this mean?

        total_loss += loss.item() # accumulate the total loss, add loss of this time in
        _, predicted = outputs.max(1) # what does this mean? max(1), this should be as a softmax
        total += targets.size(0) # this seems like count the samples in the current batch
        correct += predicted.eq(targets).sum().item() # calculate the TP ones, eq() seems like a tensor op

        pbar.set_postfix(loss=total_loss/(total/BATCH_SIZE), acc=100.*correct/total) # adding loss, acc as postfix of the bar
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/resnet18_cifar10.pth')



























