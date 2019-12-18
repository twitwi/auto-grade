

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


### Re-refresh

~~~
pip install vuejspython
pip install pyyaml
pip install openpyxl
pip install imageio
~~~

### Annotator (inspired by ../from-MC-Projects)

It uses a Python+Flask web server, interfaced using socket.io with the vue.js UI in the browser.
It allows to browse the answers of each student, use OCR (a bad one, together with suggestions and a matching algo) to pre-fill the corresponding answer, allow the user to change it, and to save it.

To run, in two processes:

~~~
pip3 install -U setuptools
pip3 install flask flask_socketio openpyxl scipy Pillow pyaml
pip3 install torch torchvision
python3 flask-ws.py

# and

cd vview
with_node
yarn install
yarn run serve
# or vue ui
~~~

### Working on an ocr project

**make your latex subject**

In the latex, use \ocr{12} (for 12 boxes), using this in header

~~~
%%%%% TODO UPDATE WITH THE MOST RECENT + HOW TO GET A TEMPLATE .yaml from the logs
%%%%%%%%% OCR %%%%%%%%%
\usepackage{pgffor}
\newcounter{ocrfieldcount}[AMCquestionaff]
\newcommand{\ocrfield}[1]{
\AMCboxDimensions{shape=square,height=2em,width=1.5em,down=.5em,rule=.5pt}
  \foreach \n in {1,...,#1}{%
    \hspace{-\fboxrule}%
    \csname AMC@answerBox@\endcsname{}{}{1}{case:\csname AMCid@name\endcsname:\the\csname AMCid@quest\endcsname,\theocrfieldcount}\stepcounter{ocrfieldcount}%
  }
}
~~~

**correct the sheets using AMC, as usual, not carring about the proper check/uncheck location of the ocr boxes**

**prepare yaml descriptor**

Prepare a parasite.yml file that you should put in the project.
Question numbers are to be found in DOC-corrige.pdf.

Propositions can end in "±0.3" to have a value of 0.3 (by default the ok/correct is of value 1, the other 0).


**prepare xlsx table**

Prepare a parasite.xlsx file that should be put in the project.
Have a "lastname" and a "firstname" column, with the student pre-filled. Also have an "examid" collumn (empty).

**run the OCR helper**

Term 1

~~~
cd vview
with_node
# maybe: yarn install
yarn run serve
# or vue ui
~~~

Term 2

~~~
. VENV/bin/activate
# pip3 install flask flask_socketio openpyxl scipy Pillow pyaml imageio
# pip3 install torch torchvision
python3 flask-ws.py
~~~

**use the OCR helper**

Visit http://localhost:8080/#/

Open the console to set the project name using the helper given, i.e., with something like

~~~
    localStorage.setItem('cfg--defaultProjectDir', '2018-infospichi-3-exam-2')
~~~

Refresh for simplicity.

In ExamId

- "load parasite"
- "do guess"
- wait... (at some point it will also start loading all images, in Term 2)
- if there are unguessed, try again "do guess" (it will affect based only on the remaining ones)
- if there are still unguessed after several "do guess" you can manually click the unguessed, look at it and then click the corresponding id in the list below
- "...save" (that will open the parasite.xlsx, create a collumn and save it back)
- check that the "examid" collumn that you created is now filled with ids

In OCRQuestions (possibly multiple times, if you want to correct by subset of questions)

- click on "focus"
- use tab/shift+tab to select the question index (0 is the first OCR question from parasite.yaml, etc)
- press key Esc
- use arrow keys to sort all answers in the proper collumns
- do the next question with "tab" (and backspace to go back to line 0)
- click "save" when you are done with all questions (you want to correct in this wave)... this should popup a message when it saved














### Annotating and exporting a miniset

Then you can annotate by clicking on FOCUS and typing what you see, using "enter" to save to the logs when you are done with a student.
Then you can select from the logs what to export as a miniset.

### Learning a model with miniset(s)

~~~
# make and send miniset
cd vview
(cd ../generated/miniset/ && zip -r ../../vview/miniset-1.zip 2018-12-17-1545081044263)
scp miniset-1.zip labslurm:/home_expes/er49873h/




srun -p LONG -t 120:00:00 -c 8 --mem=48G --pty bash -i
source /home_expes/tools/python/Python-3.7.1-debian_gpu/bin/activate
with_proxy

cd ~/auto-grade/AMC-parasite/torchlearn
cd
unzip miniset-1.zip
cd -
ln -s ~/2018-12-17-..... miniset







srun -p GPU --gres=gpu:gtxp:1 -I -N 1 -c 1 --pty -t 0-01:05 /bin/bash
with_proxy
source /home_expes/tools/python/Python-3.7.1-debian_gpu/bin/activate


# term 1
ssh labslurm
srun -p GPU --gres=gpu:gtxp:1 -I -N 1 -c 1 --mem=12000 --pty -t 0-4:05 /bin/bash
source VENV-AUTOGRADE/bin/activate
cd auto-grade/AMC-parasite/torchlearn/
python3 train-emnist.py
# and to convert to cpu
mv model-emnist.torch model-emnist-gpu.torch
python3 gpu-to-cpu.py model-emnist-gpu.torch model-emnist.torch

# term 2
ssh labslurm
emacs -nw auto-grade/AMC-parasite/torchlearn/train-emnist.py
# M-x normal-erase-is-backspace-mode

# term 3
scp labslurm:auto-grade/AMC-parasite/torchlearn/input*.png /tmp/ ; gthumb /tmp/input*.png
cd projects/auto-grade/AMC-parasite/resources/
scp labslurm:auto-grade/AMC-parasite/torchlearn/model-emnist.torch .

~~~


Sets:
1. 1-5
2. 6-10
3. 11-15
4. 16-20
5. 21-34
6. wip 35-56 (remove ' ')







                        The "LOG IT" button will append to `all-logs.jstream`, which can be converted to TSV using:

                        ~~~
                        python3 pytools/jstream-to-tsv.py  all-logs.jstream

                        less all-logs.jstream.tsv
                        ~~~


### Creating the model on a GPU machine

(here GPU2 but need to adapt titanxk to get another one, or manually ssh, ...)

~~~
ssh calcul-gpu-lahc-6

cd auto-grade/AMC-parasite/torchlearn
nvidia-smi
source /home_expes/tools/python/python3_gpu
with_proxy



srun -p GPU --gres=gpu:titanxk:1 -I -N 1 -c 1 --pty -t 0-01:05 /bin/bash

source /home_expes/tools/python/python371_gpu
with_proxy
CUDA_VISIBLE_DEVICES=2 python3 train-emnist.py





cd torchlearn
python3 train-emnist.py

# conv to cpu
def the class
ncpu = torch.load(....).cpu()
torch.save(ncpu, ....)
    ~~~
