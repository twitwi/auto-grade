import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave

from glob import glob
import re
from random import shuffle

from matplotlib import pyplot as plt

def load_uji2():
    res = []
    #for f in sorted(glob('all-jpg/*-[a-z0-9]-*.jpg')):
    for f in sorted(glob('all-jpg/*-*-*.jpg')):
        g = re.match(r'.*/*-0*(\d+)-(.)-(\w*).jpg$', f)
        target = g.group(2)
        
        #if target not in list('abc012'):
        #    continue

        res.append({
            'id': int(g.group(1)),
            'target': target,
            'is_train': g.group(3)=='train',
            'im': imread(f)/255 - 0.5,
        })
    return res

_dataset = load_uji2()

dataset_train = list(filter(lambda o: o['is_train'], _dataset))
N_train = len(dataset_train)
classes = sorted(list(set(map(lambda o: o['target'], dataset_train))))

dataset_test = list(filter(lambda o: not o['is_train'], _dataset))
N_test = len(dataset_test)

print(len(_dataset), N_train, N_test, classes)



NC = len(classes)
W = 80
H = 90

x = tf.placeholder(tf.float32, [None, H, W])
x_image = tf.reshape(x, [-1, H, W, 1])

def var_with_decay(shape, stddev, wd=.05):
    res = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.norm(res), wd)
        tf.add_to_collection('losses', weight_decay)
    return res

def convrelu(X, fr, to, no_pool=False, no_bn=False):
    w = var_with_decay([3,3, fr, to], stddev=0.05)
    b = tf.Variable(tf.truncated_normal([to], stddev=0.05))
    c = tf.nn.conv2d(X, w, [1, 1,1, 1], "VALID")
    if no_pool:
        o = tf.nn.relu(c + b)
    else:
        o = tf.nn.elu(tf.nn.max_pool(c + b, [1,2,2,1], [1,2,2,1], "VALID"))
    if not no_bn:
        o = tf.contrib.layers.batch_norm(o)
    return w,b,o

def fc(X, to, relu=True):
    fr = int(X.shape[1])
    w = var_with_decay([fr, to], stddev=0.05)
    b = tf.Variable(tf.zeros([to]))
    o = tf.matmul(X, w) + b
    if relu:
        o = tf.nn.relu(o)
    return w,b,o


# model à deux couches
NH = 100
#NF = (1, 8, 16, 16, 32, 32)
#NF = (1, 8, 16, 16, 32)
#NF = (1, 16, 32, 64, 64) # mod1 (no_res, 36) 33s full round,  28s full round (tensor rebuilt)
                          # reaches, 95.5 / 83, then overfits a little, then, reaches, 98.3 / 84.3

#NF = (1, 16,16, 32,32, 63,63, 64) # (res) test, ~100 ex per seconds

#NH, NF = 50, (1, 8, 16, 16, 64) # mod2 (no_res, 36)

#NH, NF = 50, (1, 8, 16, 32, 64) # mod3 (95%, 73% lucky)
#NH, NF = 50, (1, 8, 16, 32, 64) # mod4 same as mod3 but wd=.1 (decortiqué85.5%, 75.5%)
#NH, NF = 150, (1, 8, 32, 64, 64)  # mod5 wd=.1 (86.7%, 76.8%)
NH, NF = 200, (1, 6, 6, 16, 16, 64)  # mod6 wd=.1 lenet5-like (84.9%, 75%)


o = x_image
for i, (nf1, nf2) in enumerate(zip(NF[:-1], NF[1:])):
    print(i, nf1, nf2)
    _,_, o = convrelu(o, nf1, nf2, no_pool=i==(len(NF)-2))

# converge avec un tout-connecté
NPRE = int(np.prod(o.shape[1:]))
print(o.shape, NPRE)

_,_,fc1 = fc(tf.reshape(o, [-1, NPRE]), NH)
_,_,fc2 = fc(fc1, NC, relu=False)
#_,_,fc3 = fc(fc2, NC, relu=False)

y = fc2

# étiquette cible
y_ = tf.placeholder(tf.int64, [None])

# fonction de perte (sorte de maximisation de la probabilité prédite par le réseau)
cross_entropy = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))

# optimiseur
#train_step = tf.train.GradientDescentOptimizer(0.25).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(
    cross_entropy  + tf.add_n(tf.get_collection('losses'))
)


with tf.Session() as sess:
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
  
    for t in range(10000):
        # Entrainement
        bs = 20
        for iteration in range(100):
            #istart = (iteration*bs) % N_train
            #iend = min(istart+bs, N_train)
            #batch_xs = np.array(list(map(lambda o: o['im'], dataset_train[istart:iend])))
            #batch_ys = np.array(list(map(lambda o: classes.index(o['target']), dataset_train[istart:iend])))
            #print(batch_ys, iteration*bs, N_train, istart, iend)
            p = list(range(N_train))
            shuffle(p)
            p = list(map(dataset_train.__getitem__, p[:bs]))
            batch_xs = np.array(list(map(lambda o: o['im'], p)))
            batch_ys = np.array(list(map(lambda o: classes.index(o['target']), p)))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
        # Tester le model appris
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        bs = 200
        acc = 0
        for istart in range(0, N_train, bs):
            iend = min(istart+bs, N_train)
            acc += (iend-istart)/N_train * sess.run(accuracy, feed_dict={
                x: np.array(list(map(lambda o: o['im'], dataset_train[istart:iend]))),
                y_: np.array(list(map(lambda o: classes.index(o['target']), dataset_train[istart:iend])))
            })
        print("Precision entrainement :", acc)
        acc = 0
        for istart in range(0, N_test, bs):
            iend = min(istart+bs, N_test)
            acc += (iend-istart)/N_test * sess.run(accuracy, feed_dict={
                x: np.array(list(map(lambda o: o['im'], dataset_test[istart:iend]))),
                y_: np.array(list(map(lambda o: classes.index(o['target']), dataset_test[istart:iend])))
            })
        print("Precision test :", acc)
            
        # print("Precision entrainement :", sess.run(accuracy, feed_dict={
        #     x: np.array(list(map(lambda o: o['im'], dataset_train))),
        #     y_: np.array(list(map(lambda o: classes.index(o['target']), dataset_train)))
        # }))
        # print("Precision test :", sess.run(accuracy, feed_dict={
        #     x: np.array(list(map(lambda o: o['im'], dataset_test))),
        #     y_: np.array(list(map(lambda o: classes.index(o['target']), dataset_test)))
        # }))
        outfile = 'model-%d'%(t%10,)
        saver.save(sess, outfile)
        print("Saved as", outfile)

