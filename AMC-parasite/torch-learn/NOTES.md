
# The version recreated in 2023-02

```python
python3 train.py
```

### Helpers for the cluster

```bash
rsync -avzu torch-learn/ labslurm:torch-learn
rsync -avzu --exclude ',,*' torch-learn/ labslurm:torch-learn

srun -p GPU-DEPINFO -t 3:00:00 -c 2 --gres=gpu:1 --mem=10G --pty bash -i
cd torch-learn/
. /home_expes/tools/python/torch/gpu/1.7.1
with_proxy
python3 train.py

#####
python3 train.py  | tee -a log-long-train-wave3
mkdir ,,SAVE-37epochs+4epochs+5epochs-alt-202302
cp -t ,,SAVE-37epochs+4epochs+5epochs-alt-202302/ train.py  ,,model_scripted.pt log-long-train* ,,autosave

d=,,SAVE-37epochs+4epochs+5epochs-alt-202302 ; rsync -avzu labslurm:torch-learn/$d torch-learn/$d
cat torch-learn/,,SAVE-37epochs+4epochs+5epochs-alt-202302/,,SAVE-37epochs+4epochs+5epochs-alt-202302/log-long-train* |grep ' test'|tr '=%' '  '|awk '$1=="emnist" {s=$5; next} {print s " " $5}'|pyplot sep=" "

```

