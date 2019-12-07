
# %%
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet import nd, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from mxnet.gluon.data import DataLoader

from emnist import extract_training_samples
from emnist import extract_test_samples

from mxboard import SummaryWriter

#from gluoncv.data import transforms as gcv_transforms


ctx = mx.cpu()

# %%

# %%


our_classes = "=:;.,-_()[]!?*/'⁹"
classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"+our_classes


# %%

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.9, 1)),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(1, 1), ratio=(1, 1)),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
])

# %%

emnist_train_data, emnist_train_labels = extract_training_samples('balanced')
emnist_test_data, emnist_test_labels = extract_test_samples('balanced')

emnist_train_data = nd.array(emnist_train_data[:,:,:,None])
emnist_test_data = nd.array(emnist_test_data[:,:,:,None])


# %%

emnist_train_dataset = ArrayDataset(SimpleDataset(emnist_train_data).transform(transform_train), emnist_train_labels)
emnist_train_loader = DataLoader(emnist_train_dataset, shuffle=True, batch_size=32)

emnist_test_dataset = ArrayDataset(SimpleDataset(emnist_test_data).transform(transform_test), emnist_test_labels)
emnist_test_loader = DataLoader(emnist_test_dataset, batch_size=32)

# %%


our_net = gluon.nn.Sequential()
if False:
    with our_net.name_scope():
        our_net.add(nn.Conv2D(channels=16, kernel_size=2))
        our_net.add(nn.BatchNorm())
        our_net.add(nn.Activation(activation='relu'))
        our_net.add(nn.Conv2D(channels=16, kernel_size=2))
        our_net.add(nn.BatchNorm())
        our_net.add(nn.Activation(activation='relu'))
        our_net.add(nn.Dense(len(classes)))
        ##our_net.add(nn.LogSoftmax(dim=1))

if True:
    with our_net.name_scope():
        our_net.add(nn.BatchNorm())
        our_net.add(nn.Conv2D(channels=16, kernel_size=5))
        our_net.add(nn.BatchNorm())
        our_net.add(nn.MaxPool2D(4))
        our_net.add(nn.Activation(activation='relu'))
        our_net.add(nn.Conv2D(channels=50, kernel_size=2))
        our_net.add(nn.BatchNorm())
        our_net.add(nn.Dropout(rate=0.5))
        our_net.add(nn.MaxPool2D(2))
        our_net.add(nn.Activation(activation='relu'))
        our_net.add(nn.Dense(len(classes)))

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# %%
#our_net.collect_params().initialize(mx.init.Normal(1), ctx=ctx)
our_net.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2), force_reinit=True)

# %%

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


# %%

#trainer = gluon.Trainer(our_net.collect_params(), 'sgd', {'learning_rate': .1})
trainer = gluon.Trainer(our_net.collect_params(), 'adam', {'learning_rate': .001})

epo = 0

# %%
smoothing_constant = .01
n_epochs = 10
for e in range(n_epochs):
    epo += 1
    with SummaryWriter(logdir='./logs', flush_secs=5) as sw:
        for i, (data, label) in enumerate(emnist_train_loader):

            if i == 0:
                sw.add_image('first_minibatch', data, global_step=epo)
            if i % 300 == 0:
                print(" - minibatch", i)
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = our_net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            sw.add_scalar(tag='cross_entropy', value=loss.mean().asscalar(), global_step=epo)
            trainer.step(data.shape[0])

            ########################################
            #  Keep a moving average of the losses #
            ########################################
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                        else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

        test_accuracy = evaluate_accuracy(emnist_test_loader, our_net)
        train_accuracy = evaluate_accuracy(emnist_train_loader, our_net)
        sw.add_scalar(tag='train_acc', value=train_accuracy, global_step=epo)
        sw.add_scalar(tag='test_acc', value=test_accuracy, global_step=epo)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (epo, moving_loss, train_accuracy, test_accuracy))


# %%


# %%

from datetime import datetime
fname = ',,,,backup-'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M')
our_net.save_parameters(fname)

# %%

our_net.load_parameters(',,,,backup-2019-12-07_00:46')
print('saved')

# %%

# %%

# %%
import os
print(os.getcwd())

# %%

def miniset(path):
    ifd = ImageFolderDataset(path)

    for ii, fold in enumerate(ifd.synsets):
        folder = fold.replace(r'class-', '')
        try:
            i = classes.index(folder)
        except:
            try:
                i = classes.index(folder.upper())
            except:
                print(folder, "is not a class... replacing by ⁹")
                folder = '⁹'
                i = classes.index(folder) # dummy class.....
        ifd.synsets[ii] = folder
    
    return ifd

#train_set = miniset('mxnet-learn/2018-12-20-1545320427510')
#test_set = miniset('mxnet-learn/2018-12-20-1545320427510')

# %%




# %%
