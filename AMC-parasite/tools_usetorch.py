
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torchvision
from torchvision import transforms

from tools_querysqlite import imread, io

#import pretrainedmnist
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((60,60)),
    #transforms.Resize((42,28)),
    #transforms.CenterCrop((28,28)),
    #transforms.Grayscale(),
    transforms.ToTensor(),
    ])


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

def save_image(im, t=''):
    from matplotlib import pyplot as plt
    try:
        save_image.counter += 1
    except:
        save_image.counter = 0
    plt.imshow(im)
    plt.colorbar()
    try:
        plt.savefig('input-local{:05d}-{}.png'.format(save_image.counter, t))
    except:
        pass
    plt.close()


custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

def bytes2im(byts):
    im = imread(io.BytesIO(byts))
    im = custom_transforms(im)
    im = im.numpy()
    #save_image(im[0])
    return im
