import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


"""
On définit le modèle
input_size: Number of time steps (lattice size)
num_classes: nombre valeurs que peux prendre une sortie (nous ça sera bon chemin ou mauvais chemin)

model = NN(784,10)
x = torch.randn(64,784)
print(model(x).shape)

"""
class RNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 2
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#Initialize NN
model = NN(input_size,num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

#Train NN
"""
1 epochs = The dataset have seen 1 time all the images
data : images
targets : label
"""
for epoch in range(num_epochs):
    for batch_idx,(data, targets) in enumerate(train_loader):
        # Get data to Cuda
        data = data.to(device)
        targets = targets.to(device)

        # reshape in a single dimension
        data = data.reshape(data.shape[0], -1)

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent
        optimizer.step()

#Test accuracy

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
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

        print(f'Got {num_correct} / {num_samples} with accuraccy {float(num_correct)/float(num_samples)*100:.2f}')

        model.train()

check_accuracy(test_loader,model)