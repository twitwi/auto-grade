
import os
import shutil
from tools_querysqlite import *
from tools_generic import *
from tools_vision import bytes2im, load_model, apply_model


def load_images(project_full_path, data):
    predict = 'predict' in data and data['predict'] == True
    print('SQL QUERY')
    more=''
    if 'onlyq' in data:
        more += ' AND id_a=' + str(data['onlyq'])
    if 'only' in data:
        more += ' AND student=' + str(data['only'])
    if 'TMP' in data:
        more += ' AND student > 5 AND student < 15'
    info = preload_all_queries(make_connection(project_full_path + '/data/capture.sqlite'), more=more, improcess=assuch)
	# 0id_answer 1student
    # 2`zoneid`	INTEGER, 3`student`	INTEGER, 4`page`	INTEGER, 5`copy`	INTEGER, 6`type`	INTEGER, 7`id_a`	INTEGER, 8`id_b`	INTEGER,
    # 9`total`	INTEGER DEFAULT -1, 10`black`	INTEGER DEFAULT -1, 11`manual`	REAL DEFAULT -1, 12`image`	TEXT, 13`imagedata`	BLOB,
    prefix = 'im'
    if 'prefix' in data:
        prefix = data['prefix']
    if predict:
        modelname = 'mxnet3.model'
        print("Loading MXNet model: "+modelname)
        net = load_model(modelname)
    print('...DONE')
    sub = 'images'
    directory = './generated/'+sub
    #if os.path.exists(directory) and not 'keepImages' in data:
    #    shutil.rmtree(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    ###j = 0
    for k,v in info.items():
        pairs = list(enumerate(v))
        ims = []
        for i,r in pairs:
            n = "/" + prefix + "-" + str(r[2]) + ".png"
            ###j += 1
            with open(directory+n, "wb") as out_file:
                out_file.write(r[-1])
                v[i] = r[:-1] + (directory+n,) #('gen/'+sub+n,)
            if predict:
                ims.append(bytes2im(r[-1]))
        if predict:
            icls, ncls, preds = apply_model(ims)
            for i,r in pairs:
                ch = ncls[i]
                if r[10] == 0: ch = '_'
                v[i] = v[i][:-2] + (ch,) + v[i][-1:]
                #print(v[i])

    return info
