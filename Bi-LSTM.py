import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"cuda available {torch.cuda.is_available()}")
input_size = 28
sequence_length = 28
num_classes = 10
num_layers = 2
hidden_size = 256
learning_rate = 0.001
batch_size = 64
num_epochs = 2

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*sequence_length*2, num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        # out = self.fc(out[:, -1,:])
        out = self.fc(out)

        return out

def save_checkpoint(state, filename = "initialCheckpoint.pth.tar"):
    torch.save(state, filename)

# def load_checkpoint(checkpoint):
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])

train_dataset = datasets.MNIST(root="dataset/", train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle=True)

model = BLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# load_checkpoint(torch.load("initialCheckpoint.pth.tar"))

for epoch in range(num_epochs):
    losses = []

    if epoch == 2:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device).squeeze(1)
        targets = targets.to(device = device)

        # data = data.reshape(data.shape[0], -1)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    print(f"epochs:{epoch}")

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples    # return acc

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")