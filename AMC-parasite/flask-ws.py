


from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, join_room, emit

import time
import os
import shutil


# pip3 install flask flask-socketio eventlet
# https://secdevops.ai/weekend-project-part-2-turning-flask-into-a-real-time-websocket-server-using-flask-socketio-ab6b45f1d896


# initialize Flask
#app = Flask(__name__, template_folder='flask-ws')
app = Flask(__name__)
socketio = SocketIO(app)

local_MC = './,,test/'

# Host the mc projects
@app.route('/MC/<path:path>')
def send_MC(path):
    return send_from_directory(local_MC, path)
# Host the generate files
@app.route('/gen/<path:path>')
def send_publiccustom(path):
    return send_from_directory('./generated/', path)

@app.route('/')
def index():
    """Serve the index HTML"""
    return render_template('plain-index.html')

@socketio.on('test1')
def on_create(data):
    now = int(time.time())
    emit('test1rep', {"date": now, "original": data})

from scipy.misc import imread, imsave
import io
import sqlite3
def make_connection(p):
    return sqlite3.connect(p)
def asbase64(im):
    import base64
    return base64.b64encode(im).decode('ascii')
def assuch(im): return im
def preload_all_queries(conn, more='', improcess=asbase64):
    c = conn.cursor()
    c.execute('''SELECT id_a,student,* FROM capture_zone WHERE type=4 '''+more+''' ORDER BY student,id_a,id_b ASC''')
    res = {}
    for r in c.fetchall():
        k = r[1]
        if not (k in res):
            res[k] = []
        #r = r[:-1] + (imread(io.BytesIO(r[-1])),)
        #import pytesseract
        #print(pytesseract.image_to_string(imread(io.BytesIO(r[-1])), config='--psm 10'))
        # ^ doesn't seem to work too well on this data
        r = r[:-1] + (improcess(r[-1]),)
        res[k].append(r)
    return res
@socketio.on('test2_load_all')
def on_test2(data):
    if 'only' in data:
        info = preload_all_queries(make_connection(data['file']), more=' AND student='+str(data['only']))
    else:
        info = preload_all_queries(make_connection(data['file']))
    info['_id'] = data['_id']
    emit('test2rep', info)


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torchvision
from torchvision import transforms
#import pretrainedmnist
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((60,60)),
    #transforms.Resize((42,28)),
    #transforms.CenterCrop((28,28)),
    #transforms.Grayscale(),
    transforms.ToTensor(),
    ])

def bytes2im(byts, i=-1):
    im = imread(io.BytesIO(byts))
    im = np.transpose(transform(im).numpy(), [1,2,0])

    if False:
        N = 78
        if i==N: print(im.shape)
        R = im[:,:,0]
        G = im[:,:,1]
        B = im[:,:,2]
        im = np.sum(im, axis=2) / 3 #, np.array(im[:,:,1:2])# - np.sum(im, axis=2, keepdims=True))
        if np.max(im) - np.min(im) == 0:
            im = np.array([im])
            print("-------------", np.shape(im))
            return im
        im = (im - np.min(im)) / (np.max(im) - np.min(im))

        im[((B*2-G-R)>.05)] = np.max(im)
        im[((R*2-G-B)>.05)] = np.max(im)
    else:
        im = np.sum(im, axis=2) / 3 #, np.array(im[:,:,1:2])# - np.sum(im, axis=2, keepdims=True))
        #im = (im - np.min(im)) / (np.max(im) - np.min(im))

    im = 1 - im**4
    s = np.sum(im)
    if s == 0: s = 1
    cx = np.sum(im * np.arange(im.shape[1]).reshape((1, -1))) / s
    cy = np.sum(im * np.arange(im.shape[0]).reshape((-1, 1))) / s
    cxx = np.sum(im * (np.arange(im.shape[1])**2).reshape((1, -1))) / s
    cyy = np.sum(im * (np.arange(im.shape[0])**2).reshape((-1, 1))) / s
    cxx = (cxx - cx**2)**0.5
    cyy = (cyy - cy**2)**0.5
    print(cx, cy, cxx, cyy, s)
    #im /= np.max(im)
    #im[im>0.5] *= 3
    im = np.array((im[...,np.newaxis]*np.ones((1,1,3)))*255, dtype=np.uint8)
    print((max(0, int(im.shape[1]/2-cx)), max(0, int(im.shape[0]/2-cy)),
                    max(0, int(cx-im.shape[1]/2)), max(0, int(cy-im.shape[0]/2))))
    im = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad((max(0, int(im.shape[1]-2*cx)), max(0, int(im.shape[0]-2*cy)),
                        max(0, int(cx*2-im.shape[1])), max(0, int(cy*2-im.shape[0])))),
        transforms.CenterCrop((int(4*cxx),int(4*cyy))),
        transforms.Resize((28,28)),
        #transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])(im)

    im = im[1:2, ::, ::].numpy()
    #print(np.shape(im))
    return im

@socketio.on('test3_show')
def on_test3_show(data):
    info = preload_all_queries(make_connection(data['file']), more=' AND zoneid='+str(data['rowId']), improcess=assuch)
    from matplotlib import pyplot as plt
    for k,v in info.items():
        pairs = list(enumerate(v))
        for i,r in pairs:
            im = bytes2im(r[-1], i)
            plt.imshow(im[0,:,:], cmap='gray')
            plt.show()

@socketio.on('test3_load_all')
def on_test3(data):
    if 'only' in data:
        info = preload_all_queries(make_connection(data['file']), more=' AND student='+str(data['only']), improcess=assuch)
    else:
        info = preload_all_queries(make_connection(data['file']), improcess=assuch)
    net = torch.load('resources/model-emnist.torch').to(torch.device('cpu'))
    #net = pretrainedmnist.mnist(pretrained=True)
    for k,v in info.items():
        ims = []
        pairs = list(enumerate(v))
        for i,r in pairs:
            ims.append(bytes2im(r[-1], i))
        with torch.no_grad():
            pred = net(torch.from_numpy(np.array(ims)))
        amax = np.argmax(pred.numpy(), axis=1)
        print(amax, pred.numpy())
        our_classes = "=:;.,-_()[]!?*/'"
        classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"+our_classes
        for i,r in pairs:
            char = classes[int(amax[i])]
            if r[10] == 0: char = '_'
            v[i] = v[i][:-1] + (asbase64(v[i][-1]), char)

    info['_id'] = data['_id']
    emit('test3rep', info)

@socketio.on('test3_log')
def on_test3log(data):
    import codecs
    with codecs.open("all-logs.jstream", "a", "utf-8") as f:
        f.write(data)
    print("saved")


@socketio.on('manual-load-images')
def on_manual_load_images(data):
    print('SQL QUERY')
    if 'only' in data:
        info = preload_all_queries(make_connection(local_MC+data['pro']+'/data/capture.sqlite'), more=' AND student='+str(data['only']), improcess=assuch)
    else:
        info = preload_all_queries(make_connection(local_MC+data['pro']+'/data/capture.sqlite'), improcess=assuch)
    print('...DONE')
    sub = 'pyannotate'
    directory = './generated/'+sub
    if os.path.exists(directory):
        shutil.rmtree(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for k,v in info.items():
        pairs = list(enumerate(v))
        for i,r in pairs:
            n = "/im-" + str(i) + ".png"
            print('IMAGE', n)
            #print(info[k][:-2])
            #info[k].append(sub+n)
            #print(i, n)
            with open(directory+n, "wb") as out_file:
                out_file.write(r[-1])
                v[i] = r[:-1] + ('gen/'+sub+n,)
    info['_id'] = data['_id']
    emit('manual-loaded-images', info)

@socketio.on('manual-log') # TODO should probably scope the logs (into the project dir)
def on_manual_log(data):
    import codecs
    with codecs.open("all-logs-manual.jstream", "a", "utf-8") as f:
        f.write(data)
    print("saved")

@socketio.on('miniset-get-logs')
def on_miniset_getlogs(data):
    import codecs
    with codecs.open("all-logs-manual.jstream", "r", "utf-8") as f:
        emit('miniset-got-logs', list(map(lambda l: l[:-1], f.readlines())))

@socketio.on('miniset-export')
def on_miniset_export(data):
    name = data['name']
    print("Exporting miniset", name)
    sub = 'miniset/'+name
    directory = 'generated/'+sub
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print("WILL NOT OVERWRITE", directory)
        return
    for d in data['annotations']:
        student = d[0]
        ann = d[1]
        info = preload_all_queries(make_connection(local_MC+data['pro']+'/data/capture.sqlite'), more=' AND student='+str(student), improcess=assuch)
        info = info[int(student)]
        for i, r in list(enumerate(info)):
            if not str(i) in ann:
                print("SKIP", i)
                continue
            di = directory + '/class-' + ann[str(i)]
            if not os.path.exists(di):
                os.makedirs(di)
            n = '/im-%05d.png' % (i,)
            with open(di+n, "wb") as out_file:
                out_file.write(r[-1])
            #n = '/im-%05d.txt' % (i,)
            #with open(directory+n, "w") as out_file:
            #    out_file.write(ann[str(i)])
    print("WROTE", directory)

if __name__ == '__main__':
    socketio.run(app, debug=True)
