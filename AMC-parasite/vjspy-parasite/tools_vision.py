
import io
import numpy as np

import torch
from torchvision import transforms
from PIL import Image

legacy_mxnet = False

our_classes = "=:;.,-_()[]!?*/'+⁹"
emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
classes = emnist_classes + our_classes

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


transform_custom = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomResizedCrop(32, scale=(1, 1), ratio=(1., 1.)),
    transforms.ToTensor(),
])
def bytes2im(byts):
    im = Image.open(io.BytesIO(byts)) #torchvision.io.read_image(io.BytesIO(byts))
    im = transform_custom(im)
    im = im.detach().numpy()
    #save_image(im[0])
    return im

torch_model = None
def load_model(parameter_file_name):
    global torch_model
    print("Loading pytorch model", parameter_file_name)
    torch_model = torch.jit.load(parameter_file_name, map_location='cpu')
    torch_model.eval()
    
def apply_model(ims):
    print("Using pytorch to process", len(ims), "images")
    data = np.array(ims)
    data = torch.Tensor(data)
    with torch.no_grad():
        pred = torch_model(data)
        pred = torch.softmax(pred, axis=-1)
    icl = torch.argmax(pred, axis=-1)
    # need i-1 but I'm not sure why...
    ncl = [classes[i-1] for i in icl.detach().numpy().tolist()]
    return icl, ncl, pred

if legacy_mxnet:

    from imageio import imread
    import mxnet as mx
    from mxnet.gluon.data.vision import transforms as mxtransforms
    from mxnet.gluon import nn

    our_net_mxnet = None
    ctx = None
    def load_model(parameter_file_name):
        parameter_file_name = 'mxnet6.model'
        print("Loading MXNet model", parameter_file_name)
        global our_net_mxnet
        global ctx
        ctx = mx.cpu()
        our_net = nn.Sequential()
        with our_net.name_scope():
            our_net.add(nn.BatchNorm(in_channels=1))
            our_net.add(nn.Conv2D(channels=128, kernel_size=5))
            our_net.add(nn.BatchNorm())
            our_net.add(nn.MaxPool2D(4))
            our_net.add(nn.Activation(activation='relu'))
            our_net.add(nn.Conv2D(channels=512, kernel_size=2))
            our_net.add(nn.BatchNorm())
            our_net.add(nn.Dropout(rate=0.5))
            our_net.add(nn.MaxPool2D(2))
            our_net.add(nn.Activation(activation='relu'))
            our_net.add(nn.Dense(len(classes)))
        
        our_net.collect_params().initialize(mx.init.Normal(1), ctx=ctx)
        our_net.forward(mx.nd.zeros(shape=(1,1,32,32), ctx=ctx))
        our_net.load_parameters(parameter_file_name)
        our_net_mxnet = our_net


    def apply_model(ims):
        print("Using MXNet to process", len(ims), "images")
        data = np.array(ims)
        data = mx.nd.array(data)
        data = data.as_in_context(ctx)
        pred = our_net_mxnet(data)
        icl = mx.nd.argmax(pred, axis=1)
        icl = icl
        ncl = [classes[int(i)] for i in icl.asnumpy().tolist()]
        return icl, ncl, pred

    def amc_resizer(im):
        δ = im.shape[0] - im.shape[1]
        im = mx.image.copyMakeBorder(im, top=0, bot=0, left=δ//2, right=(δ-δ//2), values=255.)
        return im

    transform = mxtransforms.Compose([
        amc_resizer,
        mxtransforms.RandomResizedCrop(32, scale=(1, 1), ratio=(1., 1.)),
        # Transpose the image from height*width*num_channels to num_channels*height*width
        # and map values from [0, 255] to [0,1]
        mxtransforms.ToTensor(),
    ])

    def bytes2im(byts):
        im = imread(io.BytesIO(byts))
        im = mx.nd.array(im).mean(axis=2, keepdims=True)
        im = transform(im)
        im = im.asnumpy()
        #save_image(im[0])
        return im


