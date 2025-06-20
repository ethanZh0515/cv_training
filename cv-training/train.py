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

# hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
# train data need to augmented (rotate, move) to make the model more generalized
# but test data, no need to do such augmentation, so just normalize
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

# load cifar dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

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

    pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{EPOCHS}') # make the bar, visualize the progress
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



























