
import io
import numpy as np
from imageio import imread
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import nn

our_classes = "=:;.,-_()[]!?*/'+⁹" # 
emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
classes = emnist_classes + our_classes

def amc_resizer(im):
    δ = im.shape[0] - im.shape[1]
    im = mx.image.copyMakeBorder(im, top=0, bot=0, left=δ//2, right=(δ-δ//2), values=255.)
    return im

transform = transforms.Compose([
    amc_resizer,
    transforms.RandomResizedCrop(32, scale=(1, 1), ratio=(1., 1.)),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
])

def bytes2im(byts):
    im = imread(io.BytesIO(byts))
    im = mx.nd.array(im).mean(axis=2, keepdims=True)
    im = transform(im)
    im = im.asnumpy()
    #save_image(im[0])
    return im

our_net = None
ctx = None
def load_model(parameter_file_name):
    global our_net
    global ctx
    ctx = mx.cpu()
    our_net = nn.Sequential()
#    with our_net.name_scope():
#        our_net.add(nn.BatchNorm())
#        our_net.add(nn.Conv2D(channels=16, kernel_size=5))
#        our_net.add(nn.BatchNorm())
#        our_net.add(nn.MaxPool2D(4))
#        our_net.add(nn.Activation(activation='relu'))
#        our_net.add(nn.Conv2D(channels=50, kernel_size=2))
#        our_net.add(nn.BatchNorm())
#        our_net.add(nn.Dropout(rate=0.5))
#        our_net.add(nn.MaxPool2D(2))
#        our_net.add(nn.Activation(activation='relu'))
#        our_net.add(nn.Dense(len(classes)))
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

def apply_model(ims):
    print("Using MXNet to process", len(ims), "images")
    data = np.array(ims)
    data = mx.nd.array(data)
    data = data.as_in_context(ctx)
    pred = our_net(data)
    icl = mx.nd.argmax(pred, axis=1)
    icl = icl
    ncl = [classes[int(i)] for i in icl.asnumpy().tolist()]
    return icl, ncl, pred


