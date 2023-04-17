
print("Importing dependencies (preloading)")

import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("Starting: ", sys.argv)
from helper import (
    load_custom_dataset,
    evaluate_accuracy, train_one_epoch, SimpleTimer,
    classes,
    digest_command_line,
    to_device,
)

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


#######
DATASET_TINY = False # use a small fraction of the dataset (can be used for faster debugging)
DATASET_PREVIEW_IMAGES = 0 # image count (per set), use 0 to disable preview
EVAL_MOD = 1 # evaluate every n epochs (modulus)
EVAL_INIT = True # whether to evaluate at initialization
QUIET = False # whether to reduce the printing during training loops
BATCH_SIZE = 64 # the batch size (used for the dataloaders)
EPOCHS = 100 # number of epochs before (automatically) stopping
LR = 1e-4 # Learning rate
SEED = 0 # force random seed by setting to non-zero
FORCE_CPU = False # do not use GPU even if available

# feel free to add any PARAMETER here instead of hardcoded values (these parameters will be automatically accepted at the command line)

# we accept command line arguments like EVAL_INIT=False
digest_command_line(sys.argv[1:], globals())


reload_autosave = True
use_gpu = False if FORCE_CPU else torch.cuda.is_available()
dev = torch.device('cuda' if use_gpu else 'cpu')
#######

print("Loading Dataset")

training_emnist = datasets.EMNIST(train=True, split='balanced', root=',,data', download=True, transform=transform_training_emnist)
test_emnist = datasets.EMNIST(train=False, split='balanced', root=',,data', download=True, transform=transform_test_emnist)
training_emnist_dataloader = to_device(DataLoader(training_emnist, batch_size=BATCH_SIZE), dev)
test_emnist_dataloader = to_device(DataLoader(test_emnist, batch_size=BATCH_SIZE), dev)

training_custom_dataloader, test_custom_dataloader = load_custom_dataset(BATCH_SIZE, dev, DATASET_TINY)
print(f"Batch size is {BATCH_SIZE} with {len(training_custom_dataloader)} training batches, and {len(test_custom_dataloader)} test batches.")


actual_seed = SEED
if SEED != 0:
    torch.manual_seed(SEED)
else:
    actual_seed = torch.seed()

print("Using seed", actual_seed)
# if you use random or np.random, fix their seeds too


###################
if DATASET_PREVIEW_IMAGES > 0:
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
    plt.colorbar()
    plt.show()



print("Defining the Model (and loss)")

nn_loss = nn.CrossEntropyLoss()
Act = nn.GELU
model = nn.Sequential(
    # the input is a 1 channel image
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
        512 * (((32-4)//4-1)//2)**2, # computed from the conv and pool above
        len(classes)
    )
    # the output is len(classes)-sized logits
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
    
timer = SimpleTimer()
for t in range(EPOCHS):
    timer.reset()
    if QUIET:
        more1 = dict()
        more2 = dict(mod=10000000)
    else:
        more1 = more2 = dict()
    
    if t==0 and EVAL_INIT or t > 0 and t%EVAL_MOD == 0:
        #train_one_epoch(test_emnist_dataloader, model, nn_loss, optimizer, f"emnist ****TEST**** epoch {t+1}", **more2)
        #train_one_epoch(training_emnist_dataloader, model, nn_loss, optimizer, f"emnist epoch {t+1}", **more2)
        evaluate_accuracy(training_emnist_dataloader, model, "emnist training set", **more1)
        timer.reset()
        evaluate_accuracy(test_emnist_dataloader, model, "emnist test set", **more1)
        timer.reset()
        evaluate_accuracy(training_custom_dataloader, model, "custom training set", **more1)
        timer.reset()
        evaluate_accuracy(test_custom_dataloader, model, "custom test set", **more1)
        timer.reset()
    
    #train_one_epoch(training_custom_dataloader, model, nn_loss, optimizer, f"custom epoch {t+1}", **more2)
    #evaluate_accuracy(training_emnist_dataloader, model, "emnist training set", **more1)
    #evaluate_accuracy(test_emnist_dataloader, model, "emnist test set", **more1)
    #evaluate_accuracy(training_custom_dataloader, model, "custom training set", **more1)
    #evaluate_accuracy(test_custom_dataloader, model, "custom test set", **more1)

    rolling_loss = train_one_epoch(training_emnist_dataloader, model, nn_loss, optimizer, f"emnist epoch {t+1}", **more2)
    print(f"... average forward loss {rolling_loss:>8f}", timer.format('(epoch done in {s}sec)'))
    rolling_loss = train_one_epoch(training_custom_dataloader, model, nn_loss, optimizer, f"epoch {t+1}", **more2)
    print(f"... average forward loss {rolling_loss:>8f}", timer.format('(epoch done in {s}sec)'))

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
NOT SURE THERE IS NOT AN EASIER MANNER (SEE ACTUAL CODE)
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
'''
