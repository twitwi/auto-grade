
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def evaluate_accuracy(dataloader, model, label='Test', loss_fn=None):
    if label is not None:
        print(label, end='', flush=True)
    correct, sumloss, count = 0, 0, 0
    with torch.no_grad(): # do not compute gradient
        for X, y in dataloader:
            count += X.shape[0]
            prediction = model(X)
            if loss_fn is not None:
                sumloss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    sumloss /= len(dataloader) # divide by the number of batches
    correct /= count           # divide by the number of points
    if label is not None:
        print(f": accuracy={(100*correct):>0.1f}%", "" if loss_fn is None else f"avg_loss={sumloss:>8f}", flush=True)

def train_one_epoch(dataloader, model, loss_fn, optimizer, label="...", mod=100):
    if label is not None:
        print(f"Training {label}", flush=True)
    for i,(X,y) in enumerate(dataloader):
        if (i+1)%mod == 0:
            print(f'... done {i+1} minibatches', flush=True)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


our_classes = "=:;.,-_()[]!?*/'+⁹"
emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
classes = emnist_classes + our_classes
#def folder_to_class(fold):
#    print(fold)
#    ncl = fold.replace(r'class-', '')
#    if ncl == '':
#        ncl = '/'
#    try:
#        icl = classes.index(ncl)
#    except:
#        try:
#            icl = classes.index(ncl.upper())
#        except:
#            print(ncl, "is not a class... replacing by ⁹")
#            ncl = '⁹'
#            icl = classes.index(ncl)  # dummy class.....
#    return icl, ncl

transform_training_emnist = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.9, 1)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.transpose(1, 2)),
    transforms.Normalize((1,), (-1,)), # Invert
])
transform_test_emnist = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.95, 0.95)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.transpose(1, 2)),
    transforms.Normalize((1,), (-1,)), # Invert
])
transform_custom = transforms.Compose([
    transforms.Grayscale(),
    #amc_resizer,
    ####transforms.RandomResizedCrop(32, scale=(0.95, 1.05), ratio=(1., 1.)),
    transforms.RandomResizedCrop(32, scale=(1, 1), ratio=(1., 1.)),
    transforms.ToTensor(),
])

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        datasets.ImageFolder.__init__(self, *args, **kwargs)
    def find_classes(self, directory):
        from pathlib import Path
        folders = Path(directory).glob('*/')
        folders = [d.name for d in folders if d.is_dir()]
        class_to_idx = {
            f'class-{k}': i
            for i,k in enumerate(classes)
            if f'class-{k}' in folders
        }
        print(f'Loaded {len(class_to_idx)} classes')
        return list(class_to_idx.keys()), class_to_idx

#######
BS = 64
QUIET = True
EPOCHS = 100
preview_dataset = False
reload_autosave = True
use_gpu = torch.cuda.is_available()
dev = torch.device('cuda' if use_gpu else 'cpu')
#######

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def to_dev(dataloader, dev=dev): return WrappedDataLoader(dataloader, lambda x,y: (x.to(dev), y.to(dev)))


training_emnist = datasets.EMNIST(train=True, split='balanced', root=',,data', download=True, transform=transform_training_emnist)
test_emnist = datasets.EMNIST(train=False, split='balanced', root=',,data', download=True, transform=transform_test_emnist)
training_emnist_dataloader = to_dev(DataLoader(training_emnist, batch_size=BS))
test_emnist_dataloader = to_dev(DataLoader(test_emnist, batch_size=BS))

custom_dataset = CustomImageFolder(root=',,customdataset', transform=transform_custom)
custom_training_size = int(0.8 * len(custom_dataset))
custom_test_size = len(custom_dataset) - custom_training_size
training_custom, test_custom = torch.utils.data.random_split(custom_dataset, [custom_training_size, custom_test_size])
training_custom_dataloader = to_dev(DataLoader(training_custom, batch_size=BS))
test_custom_dataloader = to_dev(DataLoader(test_custom, batch_size=BS))

###################
if preview_dataset:
    from matplotlib import pyplot as plt
    K = 10
    sets = [dl.__iter__().__next__() for dl in [
        training_emnist_dataloader, test_emnist_dataloader,
        training_custom_dataloader, test_custom_dataloader
    ]]
    for i in range(K):
        for r,(s, t) in enumerate(sets):
            plt.subplot(4, K, 1+i+r*K)
            plt.imshow(s[i,0,:,:])
            plt.title(f'{t[i]}: {classes[t[i]]}')
    plt.show()




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
        512 * (((32-4)//4-1)//2)**2,
        len(classes)
    )
)
learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if use_gpu:
    print("Using GPU")
    model.to(dev)
else:
    print("Using cpu")

if reload_autosave:
    print("Reloading model")
    savepoint = torch.load(',,autosave')
    model.load_state_dict(savepoint['model'])
    optimizer.load_state_dict(savepoint['optim'])
    model.train()
    
    ###? model.load_state_dict(torch.load(PATH, map_location="cuda:0")) 

for t in range(EPOCHS):
    if QUIET:
        more1 = dict(loss_fn=nn_loss)
        more2 = dict(mod=10000000)
    else:
        more1 = more2 = dict()
    #train_one_epoch(test_emnist_dataloader, model, nn_loss, optimizer, f"emnist ****TEST**** epoch {t+1}", **more2)
    train_one_epoch(training_emnist_dataloader, model, nn_loss, optimizer, f"emnist epoch {t+1}", **more2)
    evaluate_accuracy(training_emnist_dataloader, model, "emnist training set", **more1)
    evaluate_accuracy(test_emnist_dataloader, model, "emnist test set", **more1)
    evaluate_accuracy(training_custom_dataloader, model, "custom training set", **more1)
    evaluate_accuracy(test_custom_dataloader, model, "custom test set", **more1)
    
    train_one_epoch(training_custom_dataloader, model, nn_loss, optimizer, f"custom epoch {t+1}", **more2)
    evaluate_accuracy(training_emnist_dataloader, model, "emnist training set", **more1)
    evaluate_accuracy(test_emnist_dataloader, model, "emnist test set", **more1)
    evaluate_accuracy(training_custom_dataloader, model, "custom training set", **more1)
    evaluate_accuracy(test_custom_dataloader, model, "custom test set", **more1)

    torch.save({
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'model_str': str(model),
            'loss_str': str(nn_loss),
            }, ',,autosave')
    
    model_scripted = torch.jit.script(model)
    model_scripted.save(',,model_scripted.pt')
    '''
    model = torch.jit.load(',,model_scripted.pt')
    model.eval()
    '''



# LOAD
'''
savepoint = torch.load(',,autosave')
model.load_state_dict(savepoint['model'])

model.eval()
# - or -
optimizer.load_state_dict(savepoint['optim'])
model.train()
'''

# LOAD ON GPU
'''
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
'''