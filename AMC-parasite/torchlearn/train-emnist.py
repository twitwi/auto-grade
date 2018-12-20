from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import glob
import kymatio
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch EMNIST')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")


kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
_i = 0
def go(x):
    global _i
    if _i % 25 == 0:
        print(_i, x.shape)
    _i+=1
    return x

import kymatio
scattering_transform = kymatio.Scattering2D(J=2, L=8, shape=(32,32))
scatter = transforms.Lambda(lambda t: go(scattering_transform(t)[0,:,:,:]))
scatter_contiguous = transforms.Lambda(lambda t: go(scattering_transform(t.contiguous())[0,:,:,:]))

emnist_transforms = transforms.Compose([
    transforms.Pad(2),
    #transforms.Resize((28,28)),
    #transforms.RandomAffine((90,90)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.transpose(1, 2)),
    transforms.Normalize((1,), (-1,)), # Invert
    scatter_contiguous,
    #transforms.Normalize((0.1307,), (0.3081,))
])
custom_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    scatter,
    #transforms.Normalize((0.1307,), (0.3081,))
])

our_classes = "=:;.-_()[]!?*/"
classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"+our_classes

def map_target_folder(d):
    globit = glob.glob(d+'/*')
    def __sub__(t):
        folder = globit[t]
        try:
            i = classes.index(folder[-1])
        except:
            try:
                i = classes.index(folder[-1].upper())
            except:
                #print(folder, "is not a class... replacing by 0")
                i = classes.index('W') # dummy class.....
        #print(i, folder, folder[-1])
        return torch.tensor(i, dtype=torch.long)
    return __sub__

IF = lambda d: datasets.ImageFolder(d, transform=custom_transforms, target_transform=map_target_folder(d))
def cache(ds, ondisk):
    l = len(ds)
    import pickle
    import os
    if os.path.isfile(ondisk):
        with open(ondisk, "rb") as f:
            X,Y = pickle.load(f)
    else:
        X = []
        Y = []
        for o in ds:
            X.append(o[0].numpy())
            Y.append(o[1].numpy())

        X = np.array(X)
        Y = np.array(Y)
        with open(ondisk, 'wb') as out:
            pickle.dump((X,Y), out)
        
    return torch.utils.data.TensorDataset(
        torch.from_numpy(np.array(X)),
        torch.from_numpy(np.array(Y))
    )

miniset1_dataset = cache(IF('miniset1'), ',,ms1')
miniset2_dataset = cache(IF('miniset2'), ',,ms2')
train_emnist_dataset = cache(datasets.EMNIST('/tmp/REMI-data', train=True, download=True, split='balanced', transform=emnist_transforms), ',,emtrain')

def show_data():
    from matplotlib import pyplot as plt
    import random
    o = 0
    for i in range(20):
        plt.imshow(miniset1_dataset[random.randrange(len(miniset1_dataset))][0][0])
        plt.savefig('input{:02d}.png'.format(o))
        o+=1
    em = train_emnist_dataset
    for i in range(20):
        plt.imshow(em[random.randrange(len(em))][0][0])
        plt.savefig('input{:02d}.png'.format(o))
        o+=1

#show_data()

#print(datasets.EMNIST('/tmp/REMI-data', train=True, download=True, split='balanced', transform=emnist_transforms)[0])

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        #miniset2_dataset,
        train_emnist_dataset,
        #miniset1_dataset, miniset1_dataset, miniset1_dataset,
        #miniset1_dataset, miniset1_dataset, miniset1_dataset,
        #datasets.EMNIST('/tmp/REMI-data', train=True, download=True, split='balanced', transform=emnist_transforms)
    ]),
    batch_size=args.batch_size, shuffle=True, **kwargs)


miniset1_loader = torch.utils.data.DataLoader(miniset1_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
miniset2_loader = torch.utils.data.DataLoader(miniset2_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        miniset1_dataset,
        #datasets.EMNIST('/tmp/REMI-data', train=False, split='balanced', transform=emnist_transforms),
    ]),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)



#class ENet(nn.Module):
#    def __init__(self):
#        super(ENet, self).__init__()
#        self.conv1 = nn.Conv2d(1, 50, kernel_size=5)
#        self.conv2 = nn.Conv2d(50, 100, kernel_size=3)
#        self.conv2_drop = nn.Dropout2d()
#        self.fc1 = nn.Linear(5**2*100, 4096)
#        self.fc2 = nn.Linear(4096, 2048)
#        self.fc3 = nn.Linear(2048, 47)
#
#    def forward(self, x):
#        x = x.transpose(2, 3)
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#        x = x.view(-1, 5**2*100)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return F.log_softmax(x, dim=1)
#model = ENet().to(device)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
                
model = nn.Sequential()

#model.add_module('f_conv1', nn.Conv2d(81, 1, kernel_size=1))
model.add_module('f_flat',  Flatten())
model.add_module('f_fc1',   nn.Linear(81*8*8, 1000))
model.add_module('f_relu1', nn.PReLU())
model.add_module('f_fc2',   nn.Linear(1000, 1000))
model.add_module('f_relu2', nn.PReLU())
model.add_module('f_fc2b',   nn.Linear(1000, len(classes) ))
model.add_module('f_relu2b', nn.PReLU())
model.add_module('f_lsmax', nn.LogSoftmax(dim=1))

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

def train(epoch):
    #train_loader = miniset1_loader
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


for epoch in range(1, args.epochs + 1):
    train(epoch)
    train_loss = test(train_loader)
    scheduler.step(train_loss)
    test_loss = test(test_loader)
    scheduler.step(test_loss)
    #test_loss = test(miniset1_loader)
    #test_loss = test(miniset2_loader)
    #torch.save(model, "model-emnist.torch")
    print('saved')
