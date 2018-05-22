


from flask import Flask, render_template
from flask_socketio import SocketIO, join_room, emit

import time

# pip3 install flask flask-socketio eventlet
# https://secdevops.ai/weekend-project-part-2-turning-flask-into-a-real-time-websocket-server-using-flask-socketio-ab6b45f1d896


# initialize Flask
app = Flask(__name__, template_folder='flask-ws')
socketio = SocketIO(app)

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



@socketio.on('test3_load_all')
def on_test3(data):
    if 'only' in data:
        info = preload_all_queries(make_connection(data['file']), more=' AND student='+str(data['only']), improcess=assuch)
    else:
        info = preload_all_queries(make_connection(data['file']), improcess=assuch)
    import numpy as np
    import torch
    import torchvision
    from torchvision import transforms
    import pretrainedmnist
    net = pretrainedmnist.mnist(pretrained=True)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Resize((28,28)),
        transforms.Resize((42,28)),
        transforms.CenterCrop((28,28)),
        #transforms.Grayscale(),
        transforms.ToTensor()
        ])
    for k,v in info.items():
        ims = []
        pairs = list(enumerate(v))
        for i,r in pairs:
            im = imread(io.BytesIO(r[-1]))
            im = np.transpose(transform(im).numpy(), [1,2,0])
            N = 79
            if i==N: print(im.shape)
            R = im[:,:,0]
            G = im[:,:,1]
            B = im[:,:,2]
            im = np.sum(im, axis=2) / 3 #, np.array(im[:,:,1:2])# - np.sum(im, axis=2, keepdims=True))
            im = (im - np.min(im)) / (np.max(im) - np.min(im))

            im[((B*2-G-R)>.05)] = np.max(im)
            im[((R*2-G-B)>.05)] = np.max(im)
            im = 1 - im
            #im /= np.max(im)
            #im[im>0.5] *= 3
            ims.append(im)
        from matplotlib import pyplot as plt
        plt.imshow(ims[N], cmap='gray')
        plt.show()
        with torch.no_grad():
            pred = net(torch.from_numpy(np.array(ims)))
        amax = np.argmax(pred.numpy(), axis=1)
        for i,r in pairs:
            v[i] = v[i][:-1] + (asbase64(v[i][-1]), int(amax[i]))

    info['_id'] = data['_id']
    emit('test3rep', info)


if __name__ == '__main__':
    socketio.run(app, debug=True)
