


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
def preload_all_queries(conn, more=''):
    c = conn.cursor()
    c.execute('''SELECT id_a,student,* FROM capture_zone WHERE type=4 '''+more+''' ORDER BY student,id_a,id_b ASC''')
    res = {}
    for r in c.fetchall():
        k = r[1]
        if not (k in res):
            res[k] = []
        #r = r[:-1] + (imread(io.BytesIO(r[-1])),)
        import base64
        r = r[:-1] + (base64.b64encode(r[-1]).decode('ascii'),)
        res[k].append(r)
    return res
@socketio.on('test2_load_all')
def on_create(data):
    if 'only' in data:
        info = preload_all_queries(make_connection(data['file']), more=' AND student='+str(data['only']))
    else:
        info = preload_all_queries(make_connection(data['file']))
    info['_id'] = data['_id']
    emit('test2rep', info)


if __name__ == '__main__':
    socketio.run(app, debug=True)
