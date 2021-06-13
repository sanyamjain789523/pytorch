import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#  Creating CNN
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1)) # these hyper-parameters gives the output as in the same shape
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride= (2,2)) # halves the data to 14
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        print("-" * 50)
        x = x.reshape(x.shape[0], -1)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)

        return x


# model = NN(784, 10)
model = CNN()
x = torch.randn(64, 1, 28, 28)

print(model(x).shape)
exit()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 784
num_classes = 10
learning_rate = 0.0001
batch_size = 64
num_epochs = 1

train_dataset = datasets.MNIST(root="dataset/", train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle=True)

model = NN(input_size = input_size, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        data = data.reshape(data.shape[0], -1)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def check_accuracy(loader, model):

    if loader.dataset.train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on test data")
    num_correct, num_samples = 0,0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got  accuracy {float(num_correct)*100/float(num_samples)}")

    model.train()
    # return acc

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)