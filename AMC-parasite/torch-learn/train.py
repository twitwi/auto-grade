
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def evaluate_accuracy(dataloader, model, label='Test', loss_fn=None):
    if label is not None:
        print(label, end='', flush=True)
    correct, sumloss = 0, 0
    with torch.no_grad(): # do not compute gradient
        for X, y in dataloader:
            prediction = model(X)
            if loss_fn is not None:
                test_loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    sumloss /= len(dataloader)       # divide by the number of batches
    correct /= len(dataloader.dataset) # divide by the number of points
    if label is not None:
        print(f": accuracy={(100*correct):>0.1f}%", "" if loss_fn is None else "avg_loss={test_loss:>8f}")

def train_one_epoch(dataloader, model, loss_fn, optimizer, label="...", mod=100):
    if label is not None:
        print(f"Training {label}")
    for i,(X,y) in enumerate(dataloader):
        if (i+1)%mod == 0:
            print(f'... done {i+1} minibatches')
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


our_classes = "=:;.,-_()[]!?*/'+⁹"
emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
classes = emnist_classes + our_classes

BS = 64

training_emnist = datasets.EMNIST(train=True, split='balanced', root=',,data', download=True, transform=ToTensor())
test_emnist = datasets.EMNIST(train=False, split='balanced', root=',,data', download=True, transform=ToTensor())
training_emnist_dataloader = DataLoader(training_emnist, batch_size=BS)
test_emnist_dataloader = DataLoader(test_emnist, batch_size=BS)

nn_loss = nn.CrossEntropyLoss()
Act = nn.GELU
model = nn.Sequential(
    nn.BatchNorm2d(1),
    nn.Conv2d(1, 128, kernel_size=5),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(4),
    Act(),
    nn.Conv2d(128, 512, kernel_size=2),
    nn.BatchNorm2d(512),
    nn.Dropout(0.3),
    nn.MaxPool2d(2),
    Act(),
    nn.Flatten(),
    nn.Linear(
        512 * (((28-4)//4-1)//2)**2,
        len(classes)
    )
)
learning_rate = 2e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 5
for t in range(epochs):
    train_one_epoch(training_emnist_dataloader, model, nn_loss, optimizer, f"emnist epoch {t+1}")
    evaluate_accuracy(training_emnist_dataloader, model, "emnist training set")
    evaluate_accuracy(test_emnist_dataloader, model, "emnist test set")
