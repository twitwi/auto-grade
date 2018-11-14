

A new start from from-MC-projects.

Will do a UI that helps around AMC, and maybe replace the UI completely at some point.
For now it is mostly to have the OCR feature.

## Setup

### AMC

Around AMC, to generate the capture file that serves as a basis for OCR labeling.

Building the docker version of AMC, with a custom rendering (no boxes), with ~/app/docker-build/....-custom
~~~
cd ~/app/docker-build/AMC
docker--build -t amc --network host   -f Dockerfile-1404-custom .
~~~

Then it can be run with `@amc` and scan etc.


### Annotator (inspired by ../from-MC-Projects)

It uses a Python+Flask web server, interfaced using socket.io with the vue.js UI in the browser.
It allows to browse the answers of each student, use OCR (a bad one, together with suggestions and a matching algo) to pre-fill the corresponding answer, allow the user to change it, and to save it.

To run, in two processes:

~~~
pip3 install flask flask_socketio
python3 flask-ws.py

# and

cd vue-view
with_node
yarn install
yarn run dev
~~~

The "LOG IT" button will append to `all-logs.jstream`, which can be converted to TSV using:

~~~
python3 pytools/jstream-to-tsv.py  all-logs.jstream

less all-logs.jstream.tsv
~~~


### Creating the model on a GPU machine

(here GPU2 but need to adapt titanxk to get another one, or manually ssh, ...)

~~~
srun -p GPU --gres=gpu:titanxk:1 -I -N 1 -c 1 --pty -t 0-01:05 /bin/bash

source /home_expes/tools/python/python3_gpu
with_proxy
#NOT SURE IT IS NECESSARY AS THE CLUSTER HAS THE LAST VERSION pip3 install --target=/tmp/REMI-PIP  http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-linux_x86_64.whl torchvision

#PYTHONPATH=/tmp/REMI-PIP
python3 train-emnist2.py

#PYTHONPATH=/tmp/REMI-PIP python3 /tmp/REMI-PIP/ptpython/entry_points/run_ptpython.py

# conv to cpu
def the class
ncpu = torch.load(....).cpu()
torch.save(ncpu, ....)
~~~
