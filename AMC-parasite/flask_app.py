
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, join_room, emit

import time
import os
import shutil
import codecs
import openpyxl
from itertools import islice

from tools_querysqlite import *
from tools_usetorch import *
from tools_generic import *
from importlib import import_module

# pip3 install flask flask-socketio eventlet
# https://secdevops.ai/weekend-project-part-2-turning-flask-into-a-real-time-websocket-server-using-flask-socketio-ab6b45f1d896
# pip3 install openpyxl yaml

# initialize Flask
#app = Flask(__name__, template_folder='flask-ws')
app = Flask(__name__)
socketio = SocketIO(app)

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
