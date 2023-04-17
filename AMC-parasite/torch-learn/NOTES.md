

# Re/update (from DL1) 2023-04-14


ssh labslurm
tmux a
cd torch-learn/
rm -rf ,,*

rsync -avzu torch-learn/,,customdataset/ labslurm:torch-learn/,,customdataset
rsync -avzu --exclude ',,*' torch-learn/ labslurm:torch-learn

with_proxy
python3 train.py EVAL_INIT=False | tee -a ,,y


mkdir ,,SAVE-13epochs
cp -t ,,SAVE-13epochs/ train.py  ,,model_scripted.pt ,,y ,,autosave

d=,,SAVE-13epochs ; rsync -avzu labslurm:torch-learn/$d torch-learn/$d
cat torch-learn/$d/$d/,,y |grep ' test'|tr '=%' '  '|awk '$1=="emnist" {s=$5; next} {print s " " $5}'|pyplot sep=" "

cat torch-learn/$d/$d/,,y |grep ' set:'|tr '=%' '  '|awk '$1$2=="emnisttraining" {etr=$5; next} $1$2=="emnisttest" {ete=$5; next} $1$2=="customtraining" {ctr=$5; next} {print etr " " ete " " ctr " " $5}'|pyplot sep=" "



# The version recreated in 2023-02

```python
python3 train.py
```

### Helpers for the cluster

```bash
rsync -avzu torch-learn/ labslurm:torch-learn
rsync -avzu --exclude ',,*' torch-learn/ labslurm:torch-learn

tmux
srun -p GPU-DEPINFO -t 8:00:00 -c 2 --gres=gpu:1 --mem=10G --pty bash -i
cd torch-learn/
. /home_expes/tools/python/torch/gpu/1.7.1
with_proxy
python3 train.py | tee -a ,,

#####
python3 train.py  | tee -a log-long-train-wave3
mkdir ,,SAVE-37epochs+4epochs+5epochs-alt-202302
cp -t ,,SAVE-37epochs+4epochs+5epochs-alt-202302/ train.py  ,,model_scripted.pt log-long-train* ,,autosave

d=,,SAVE-37epochs+4epochs+5epochs-alt-202302 ; rsync -avzu labslurm:torch-learn/$d torch-learn/$d
cat torch-learn/$d/$d/log-long-train* |grep ' test'|tr '=%' '  '|awk '$1=="emnist" {s=$5; next} {print s " " $5}'|pyplot sep=" "

cat torch-learn/$d/$d/log-long-train* |grep ' set:'|tr '=%' '  '|awk '$1$2=="emnisttraining" {etr=$5; next} $1$2=="emnisttest" {ete=$5; next} $1$2=="customtraining" {ctr=$5; next} {print etr " " ete " " ctr " " $5}'|pyplot sep=" "



```

