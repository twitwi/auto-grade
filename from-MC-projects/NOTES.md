
See also ../README.md for details about hacking amc.




# About the annotator

It uses a Python+Flask web server, interfaced using socket.io with the vue.js UI in the browser.
It allows to browse the answers of each student, use OCR (a bad one, together with suggestions and a matching algo) to pre-fill the corresponding answer, allow the user to change it, and to save it.

Currently, some things are strongly hard-coded, this includes:
- the fields (OCR boxes) to extract, and which type they have (see Test3.vue#created)
- the matching algorithm, with the auto-complete list that is a per category list of words (suggestions.js)

To run, in two processes:

~~~
python3 flask-ws.py

# and

cd vue-view
with_node
yarn run dev
~~~

"LOG IT" will append to `all-logs.jstream`, which can be converted to TSV using:

~~~
python3 jstream-to-tsv.py  ../all-logs.jstream

less ../all-logs.jstream.tsv
~~~


# On GPU2

~~~
srun -p GPU --gres=gpu:titanxk:1 -I -N 1 -c 1 --pty -t 0-01:05 /bin/bash

source /home_expes/tools/python/python3_gpu
with_proxy
pip3 install --target=/tmp/REMI-PIP  http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-linux_x86_64.whl torchvision

PYTHONPATH=/tmp/REMI-PIP python3 train-emnist2.py

PYTHONPATH=/tmp/REMI-PIP python3 /tmp/REMI-PIP/ptpython/entry_points/run_ptpython.py

# conv to cpu
def the class
ncpu = torch.load(....).cpu()
torch.save(ncpu, ....)
~~~
