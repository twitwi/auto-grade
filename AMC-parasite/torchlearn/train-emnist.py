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

has_scatter = False

if has_scatter:
    import kymatio
    scattering_transform = kymatio.Scattering2D(J=2, L=8, shape=(32,32))
    scatter = transforms.Lambda(lambda t: go(scattering_transform(t)[0,:,:,:]))
    scatter_contiguous = transforms.Lambda(lambda t: go(scattering_transform(t.contiguous())[0,:,:,:]))
else:
    scatter = transforms.Normalize((0.1307,), (0.3081,))
    scatter_contiguous = scatter

emnist_transforms = transforms.Compose([
    #transforms.Pad(2), # for no augmentation
    transforms.Pad(4), transforms.RandomAffine(20, (.2, .2), (0.7, 1.1), 20), transforms.CenterCrop(32), # aug
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.transpose(1, 2)),
    transforms.Normalize((1,), (-1,)), # Invert
    scatter_contiguous,
])
custom_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    scatter,
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

if not has_scatter:
    def cache(ds, ondisk):
        return ds

# the "balanced" version
#miniset1_dataset = cache(IF('miniset1'), ',,ms1')
#miniset2_dataset = cache(IF('miniset2'), ',,ms2')
#train_emnist_dataset = cache(datasets.EMNIST('/tmp/REMI-data', train=True, download=True, split='balanced', transform=emnist_transforms), ',,emtraindigits')

miniset1_dataset = cache(IF('miniset1-digits'), ',,ms1-digits')
miniset2_dataset = cache(IF('miniset2-digits'), ',,ms2-digits')
miniset3_dataset = cache(IF('miniset3-digits'), ',,ms3-digits')
miniset4_dataset = cache(IF('miniset4-digits'), ',,ms4-digits')
train_emnist_dataset = cache(datasets.EMNIST('/tmp/REMI-data', train=True, download=True, split='digits', transform=emnist_transforms), ',,emtraindigits') # !!! 280k, do not cache this

def show_data():
    from matplotlib import pyplot as plt
    import random
    o = 0
    for i in range(20):
        plt.imshow(miniset1_dataset[random.randrange(len(miniset1_dataset))][0][0])
        plt.colorbar()
        plt.savefig('input{:02d}.png'.format(o))
        plt.close()
        o+=1
    em = train_emnist_dataset
    for i in range(20):
        plt.imshow(em[random.randrange(len(em))][0][0])
        plt.colorbar()
        plt.savefig('input{:02d}.png'.format(o))
        plt.close()
        o+=1

#show_data()

#print(datasets.EMNIST('/tmp/REMI-data', train=True, download=True, split='balanced', transform=emnist_transforms)[0])

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        #miniset1_dataset,
        #miniset2_dataset,
        #miniset3_dataset,
        train_emnist_dataset,
        #datasets.EMNIST('/tmp/REMI-data', train=True, download=True, split='balanced', transform=emnist_transforms)
    ]),
    batch_size=args.batch_size, shuffle=True, **kwargs)


# how to subsample???
#train_loader = torch.utils.data.random_split(train_emnist_dataset, [1000, len(train_emnist_dataset)-1000])[0]
#print(train_loader[0])
#train_loader = torch.utils.data.BatchSampler(
#    torch.utils.data.SequentialSampler(train_loader),
#    args.batch_size,
#    True
#)
#print(train_loader)
#print(enumerate(train_loader)[0])


miniset1_loader = torch.utils.data.DataLoader(miniset1_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
miniset2_loader = torch.utils.data.DataLoader(miniset2_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        miniset1_dataset,
        miniset2_dataset,
        miniset3_dataset,
        miniset4_dataset,
        #datasets.EMNIST('/tmp/REMI-data', train=False, split='balanced', transform=emnist_transforms),
    ]),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
                
model = nn.Sequential()
if has_scatter:
    model.add_module('f_conv1', nn.Conv2d(81, 100, kernel_size=3))
    model.add_module('f_relu1', nn.ReLU())
    model.add_module('f_bn1',   nn.BatchNorm2d(100))
    model.add_module('f_flat',  Flatten())
    model.add_module('f_fc1',   nn.Linear(100*6*6, 10))
    model.add_module('f_lsmax', nn.LogSoftmax(dim=1))
else:
    model.add_module('f_conv1', nn.Conv2d(1, 16, kernel_size=2))
    model.add_module('f_bn1',   nn.BatchNorm2d(16))
    model.add_module('f_relu1', nn.ReLU())
    model.add_module('f_conv2', nn.Conv2d(16, 16, kernel_size=2))
    model.add_module('f_bn2',   nn.BatchNorm2d(16))
    model.add_module('f_relu2', nn.ReLU())
    model.add_module('f_flat',  Flatten())
    model.add_module('f_fc1',   nn.Linear(16*30*30, 10))
    model.add_module('f_lsmax', nn.LogSoftmax(dim=1))

if True:
    model = nn.Sequential()
    model.add_module('f_bninput',   nn.BatchNorm2d(1))
    model.add_module('f_conv1', nn.Conv2d(1, 16, kernel_size=5))
    model.add_module('f_bn1',   nn.BatchNorm2d(16))
    model.add_module('f_pool1', nn.MaxPool2d(4))
    model.add_module('f_relu1', nn.ReLU())
    model.add_module('f_conv2', nn.Conv2d(16, 50, kernel_size=2))
    model.add_module('f_bn2',   nn.BatchNorm2d(50))
    model.add_module('f_drop1', nn.Dropout2d())
    model.add_module('f_pool2', nn.MaxPool2d(2))
    model.add_module('f_relu2', nn.ReLU())
    model.add_module('f_flat',  Flatten())
    model.add_module('f_fc1',   nn.Linear(50*3*3, 10))
    model.add_module('f_lsmax', nn.LogSoftmax(dim=1))
    
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

def train(epoch):
    #train_loader = miniset1_loader
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > len(train_loader)//20: break
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
    #train_loss = test(train_loader)
    #scheduler.step(train_loss)
    test_loss = test(test_loader)
    scheduler.step(test_loss)
    #test_loss = test(miniset1_loader)
    #test_loss = test(miniset2_loader)
    torch.save(model, "model-emnist.torch")
    print('saved')
