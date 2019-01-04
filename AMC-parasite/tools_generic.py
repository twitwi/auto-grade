
import yaml

row_elements = 'username firstname lastname group examid'.split(' ')

def noop(*args, **kwargs): pass
def lmap(*args, **kwargs): return list(map(*args, **kwargs))
def at(k):                 return lambda o: o[k]
def attr_value():          return lambda o: o.value
def of(o):                 return lambda k: o[k]

def read_config(pro, filename='parasite.yaml', print=noop):
    with open(pro + '/' + filename) as f:
        cfg = yaml.load(f)
    def withdef(path, v, o=cfg, prev=''):
        if '/' in path:
            splt = path.split('/', 1)
            withdef(splt[1], v, o[splt[0]], prev+'→'+splt[0])
        else:
            if path not in o:
                print("At", prev+'→'+path, "--- Setting default", v)
                o[path] = v
            else:
                print("At", prev+'→'+path, "--- Keeping", o[path], "vs default", v)
    for h in row_elements:
        withdef('headers/'+h, h)
    return cfg
