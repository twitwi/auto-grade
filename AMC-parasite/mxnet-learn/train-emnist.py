
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
#from gluoncv.data.transforms.image import resize_short_within


from emnist import extract_training_samples
from emnist import extract_test_samples

from mxboard import SummaryWriter

from datetime import datetime

def save_now(net):
    fname = ',,,,backup-'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M')
    net.save_parameters(fname)
    print('saved as', fname)

#from gluoncv.data import transforms as gcv_transforms


ctx = mx.gpu()

# %%

# %%

our_classes = "=:;.,-_()[]!?*/'+⁹"
emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
classes = emnist_classes + our_classes


# %%

transform_train_emnist = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.9, 1)),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
])


def amc_resizer(im):
    δ = im.shape[0] - im.shape[1]
    im = mx.image.copyMakeBorder(im, top=0, bot=0, left=δ//2, right=(δ-δ//2), values=255.)
    return im


transform_train = transforms.Compose([
    amc_resizer,
    transforms.RandomResizedCrop(32, scale=(0.95, 1.05), ratio=(1., 1.)),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    amc_resizer,
    transforms.RandomResizedCrop(32, scale=(1, 1), ratio=(1, 1)),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
])

# %%

emnist_train_data, emnist_train_labels = extract_training_samples('balanced')
emnist_test_data, emnist_test_labels = extract_test_samples('balanced')

emnist_train_data = nd.array(255 - emnist_train_data[:,:,:,None])
emnist_test_data = nd.array(255 - emnist_test_data[:,:,:,None])


# %%
BS = 64

emnist_train_dataset = ArrayDataset(SimpleDataset(emnist_train_data).transform(transform_train_emnist), emnist_train_labels)
emnist_train_loader = DataLoader(emnist_train_dataset, shuffle=True, batch_size=BS)

emnist_test_dataset = ArrayDataset(SimpleDataset(emnist_test_data).transform(transform_test), emnist_test_labels)
emnist_test_loader = DataLoader(emnist_test_dataset, batch_size=BS)

#with SummaryWriter(logdir='./logs') as sw:
#    sw.add_histogram('emnist_classes', mx.nd.array([c for (f,c) in emnist_train_dataset]), bins=np.arange(-0.5, len(classes)+1))
#    sw.add_histogram('emnist_classes', mx.nd.array([c for (f,c) in emnist_test_dataset]), bins=np.arange(-0.5, len(classes)+1))

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

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# %%
#our_net.collect_params().initialize(mx.init.Normal(1), ctx=ctx)
our_net.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2), force_reinit=True, ctx=ctx)

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

#trainer = gluon.Trainer(our_net.collect_params(), 'adam', {'learning_rate': .001})
trainer = gluon.Trainer(our_net.collect_params(), 'adam', {'learning_rate': .0001})
#trainer = gluon.Trainer(our_net.collect_params(), 'ftml', {})


epo = 0

# %%

def train_it(net, trainer, train_loader, test_loader, n_epochs=1, n_noise = 0):
    global epo
    smoothing_constant = .01
    for e in range(n_epochs):
        epo += 1
        with SummaryWriter(logdir='./logs', flush_secs=5) as sw:
            for i, (data, label) in enumerate(train_loader):

                if i == 0:
                    sw.add_image('first_minibatch', data, global_step=epo)
                if i % 500 == 0:
                    print(" - minibatch", i)
                data = data.copy()
                # as we have unseen classes, ensure noise is predicted as noise
                if n_noise > 0:
                    data[:n_noise, :, :, :] = np.random.uniform(size=(n_noise, 1, 32, 32))
                    label[:n_noise] = np.random.randint(len(classes), size=(n_noise,))
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                with autograd.record():
                    output = net(data)
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

            test_accuracy = evaluate_accuracy(test_loader, net)
            train_accuracy = evaluate_accuracy(train_loader, net)
            sw.add_scalar(tag='train_acc', value=train_accuracy, global_step=epo)
            sw.add_scalar(tag='test_acc', value=test_accuracy, global_step=epo)
            print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (epo, moving_loss, train_accuracy, test_accuracy))

# %%
train_it(our_net, trainer, emnist_train_loader, emnist_test_loader, n_epochs=1, n_noise=4)

# %%


# %%

save_now(our_net)
# %%

#our_net.load_parameters(',,,,backup-2019-12-07_12:37')
our_net.load_parameters(',,,,backup-2019-12-19_15:29')
print('loaded')

# %%

# %%

# %%
import os
print(os.getcwd())

# %%
glo = {}

def folder_to_class(fold):
    ncl = fold.replace(r'class-', '')
    if ncl == '': ncl = '/'
    try:
        icl = classes.index(ncl)
    except:
        try:
            icl = classes.index(ncl.upper())
        except:
            print(ncl, "is not a class... replacing by ⁹")
            ncl = '⁹'
            icl = classes.index(ncl) # dummy class.....
    return icl, ncl

def miniset(paths, train=True, batch_size=BS):
    if type(paths) == str:
        paths = [paths]

    transf = transform_train if train else transform_test

    ifd_all = None
    for ipath, path in enumerate(paths):
        ifd = ImageFolderDataset(path, flag=0)
        # use folder names ("synsets") to map the loader class to our class index (in "classes") 
        for ii, (f, cl) in enumerate(ifd.items):
            icl, ncl = folder_to_class(ifd.synsets[cl])
            ifd.items[ii] = (f, icl)

        ifd.synsets = list(classes)
        #ifd = ifd.transform_first(transf)
        if ipath == 0:
            ifd_all = ifd
        else:
            ifd_all.items += ifd.items

    ifd_all = ifd_all.transform_first(transf)
    loader = DataLoader(ifd_all, shuffle=True, batch_size=batch_size)

    return ifd_all, loader

# %%
#mini2 = '2018-12-19-1545245917071'
#mini3 = '2018-12-20-1545320427510'
#mini4 = '2018-12-20-1545326369644'
#mini5 = '2018-12-23-1545524873090'
#mini6 = '2018-12-23-1545577163244'
#minisets = [mini2, mini3, mini4, mini5, mini6]
# they are actually all very correlated as is, no new set since then

mini2018 = '2018-clean-dataset'
minisets = [
'miniset-2019-12-19--2018-infospichi-4',
'miniset-2019-12-19--2018-poo-2',
'miniset-2019-12-19--2018-pwa-1',
'miniset-2019-12-19--2019-dw2-1',
'miniset-2019-12-19--2019-infospichi-2',
'miniset-2019-12-19--2019-network-1',
]

# %%
for mi, mns in enumerate(minisets):
    print(mi, '))', mns)
    dataset, loader = miniset('mxnet-learn/'+mns)
    with SummaryWriter(logdir='./logs') as sw:
        sw.add_histogram('all_minisets', mx.nd.array([c for (f,c) in dataset]), bins=np.arange(-0.5, len(classes)+1), global_step=mi)

# %%

def make_dataset_viewer(reldir):
    import urllib.parse
    dir = 'mxnet-learn/'+reldir
    class ListDict(dict):
        def __missing__(self, key):
            self[key] = []
            return self[key]

    ifd = ImageFolderDataset(dir, flag=0)
    class_images = ListDict()

    for ii, (f, cl) in enumerate(ifd.items):
        icl, ncl = folder_to_class(ifd.synsets[cl])
        class_images[icl].append(f)

    def relpath(im):
        return urllib.parse.quote(im[im.index(reldir):])
    def imgs(l):
        return '\n'.join((f'<img src="{relpath(im)}"/>' for im in l))
        
    body = ''.join((
        f"""
        <div>Class {icl} <span>{classes[icl]}</span> ({len(class_images[icl])})</div>
        {imgs(class_images[icl])}
        """
        for icl in sorted(class_images.keys())
    ))

    with open(dir+'.html', 'w') as f:
        print("""
        <html>
        <head>
            <style>
            div span { border: 1px solid black; background: gray; font-size: 150%; padding: 0.2em; }
            </style>
        </head>
        <body>
           <div>All """+str(sum([len(l) for l in class_images.values()]))+"""</div>
        """ + body + """
        </body>
        </html>
        """, file=f)

make_dataset_viewer(minisets[0])

# %%

train_set, train_loader = miniset(['mxnet-learn/'+m for m in minisets[:-1]])
acc = evaluate_accuracy(train_loader, our_net)
print(acc)

test_set, test_loader = miniset('mxnet-learn/'+minisets[-1], False)
acc = evaluate_accuracy(test_loader, our_net)
print(acc)


# %%
train_it(our_net, trainer, train_loader, test_loader, n_epochs=50)


# %%
acc = evaluate_accuracy(emnist_train_loader, our_net)
print(acc)

# %%

save_now(our_net)

# %%

import matplotlib.pyplot as plt
i = 550
dset = emnist_test_dataset
print(i, dset[i][1], classes[dset[i][1]])
plt.imshow(dset[i][0][0,:,:].asnumpy())

# %%

i = 275
dset = test_set
print(i, dset[i][1], classes[dset[i][1]])
plt.imshow(dset[i][0][0,:,:].asnumpy())
# %%


# %%
