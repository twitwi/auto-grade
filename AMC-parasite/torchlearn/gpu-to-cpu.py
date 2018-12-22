from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import sys

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


f = sys.argv[1]
try:
    o= sys.argv[2]
except:
    o = re.sub(r'([.][^.]*)$', r'-cpu\1', f)
    
m = torch.load(f)
mcpu = m.cpu()
torch.save(mcpu, o)




