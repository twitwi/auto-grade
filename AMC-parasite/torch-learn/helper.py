
import ast
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Python helpers

## Command line parse PARAM1=2 PARAM2=True PARAM42=42000 command line arguments, saving to globals()
def digest_command_line(argv, target, check_exists=True, check_type=True, ignore_no_equal=False):
    no_equal = []
    for p in argv:
        parts = p.split('=', 1)
        if len(parts) == 1:
            no_equal.append(p)
            if not ignore_no_equal:
                raise Exception(f"No '=' found in {p}")
            continue
        varname, varval = parts
        if check_exists and not varname in target:
            raise Exception(f"No '{varname}' in target")
        v = ast.literal_eval(varval)
        if check_type and varname in target and type(v) != type(target[varname]):
            raise Exception(f"Types differ: passed a '{type(v)}' literal while the existing is a '{type(target[varname])}'")
        target[varname] = v
    return no_equal       
        
## Timer class
class SimpleTimer():
    def __init__(self):
        self.reset()
    def reset(self):
        self.start = time.perf_counter_ns()
    def delta(self):
        return (time.perf_counter_ns() - self.start) / 1e9
    def format(self, f):
        s = self.delta()
        ms = int(1000*s)
        return f.format(start=self.start, s=s, ms=ms, d=s, delta=s)
    def log(self, f):
        return lambda: self.format(f)

## Utility function to allow uniform processing of constant, delayed or None
def maybe(f, ifnone=''):
    if f is None:   return ifnone
    if callable(f): return f()
    return f


# Dataset-related helpers

## Classes for our use case

our_classes = "=:;.,-_()[]!?*/'+⁹"
emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
classes = emnist_classes + our_classes

## Default transform (forces grayscale, rescales to 32×32, and converts to tensor)

transform_custom = transforms.Compose([
    transforms.Grayscale(),
    ####transforms.RandomResizedCrop(32, scale=(0.95, 1.05), ratio=(1., 1.)),
    transforms.RandomResizedCrop(32, scale=(1, 1), ratio=(1., 1.)),
    transforms.ToTensor(),
])

## Loading from some image folders, from a closed class list.
## Ensures repeatable class and sample order.

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
        return sorted(list(class_to_idx.keys())), class_to_idx



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

def to_device(dataloader, dev):
    return WrappedDataLoader(dataloader, lambda x,y: (x.to(dev), y.to(dev)))

## Our

def load_custom_dataset(BS, dev, tiny=False, TEST_BS=None):
    if TEST_BS is None:
        TEST_BS = BS
    custom_dataset = CustomImageFolder(root='DATA', transform=transform_custom)
    custom_training_size = int(0.8 * len(custom_dataset))
    custom_test_size = len(custom_dataset) - custom_training_size
    fixed_random = dict(generator=torch.Generator().manual_seed(42000)) # fix the train/eval seed, never change this
    if tiny:
        training_custom, _, test_custom, _ = torch.utils.data.random_split(custom_dataset, [
            custom_training_size//10, custom_training_size - custom_training_size//10,
            custom_test_size//10, custom_test_size - custom_test_size//10], **fixed_random)
    else:
        training_custom, test_custom = torch.utils.data.random_split(custom_dataset, [custom_training_size, custom_test_size], **fixed_random)
    training_custom_dataloader = to_device(DataLoader(training_custom, batch_size=BS), dev)
    test_custom_dataloader = to_device(DataLoader(test_custom, batch_size=TEST_BS), dev)
    return training_custom_dataloader, test_custom_dataloader


# Generic evaluation loop
def evaluate_accuracy(dataloader, model, label='Test', loss_fn=None, label_end=None):
    model.eval()
    print(maybe(label), end='', flush=True)
    correct, sumloss, count = 0, 0, 0
    with torch.no_grad(): # do not compute gradient
        for X, y in dataloader:
            count += X.shape[0]
            prediction = model(X)
            if loss_fn is not None:
                sumloss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    avgloss = sumloss / len(dataloader)
    # ^ divide by the number of batches (losses like the CE do a batch mean by default)
    
    if label is not None or label_end is not None:
        print(f": accuracy={(100*correct/count):>0.1f}%",
              "" if loss_fn is None else f"avg_loss={avgloss:>8f}",
              maybe(label_end),
              flush=True)
        
    return correct, count, avgloss

## Generic training loop
def train_one_epoch(dataloader, model, loss_fn, optimizer, label="...", mod=100):
    model.train()
    print('Training', maybe(label), flush=True)
        
    rolling_loss = 0
    for i,(X,y) in enumerate(dataloader):
        if (i+1)%mod == 0:
            print(f'... done {i+1} minibatches', flush=True)
        pred = model(X)
        loss = loss_fn(pred, y)
        rolling_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return rolling_loss / len(dataloader)


