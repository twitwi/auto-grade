
from jinja2 import Template as JJ2
import numpy as np
from pathlib import Path

def load_uji2(imshape=()):
    with open('ujipenchars2.txt', 'r') as f:
        g_minx = 999999
        g_maxx = -999999
        g_miny = 999999
        g_maxy = -999999
        
        data_index = -1
        strokes = []
        remaining_strokes = -1
        char = '«'
        is_train = None
        
        for line in f:
            if line.startswith('//'): continue
            e = line.lstrip(' ').split(' ')
            print(len(e))
            print(line)
            if e[0] == 'WORD':
                data_index += 1
                strokes = []
                remaining_strokes = -1
                char = e[1]
                is_train = e[2].startswith('trn_')
            elif e[0] == 'NUMSTROKES':
                remaining_strokes = int(e[1])
            elif e[0] == 'POINTS':
                strokes.append([
                    (int(e[i]), int(e[i+1]))
                    for i in range(3, len(e)-1, 2)
                    ])
                remaining_strokes -= 1
                print(strokes)
            if remaining_strokes == 0:
                # process
                # svg template bounds (manual, 0 111 0 155)
                # dataset bounds -25 2221 -193 3099
                print("GO")
                r = 155/3099 * .8
                svg = JJ2(Path('template.svg').read_text()).render({
                    'strokes': [
                        {
                            'path': 'M '+ ' L '.join([ str(x*r)+','+str(y*r) for x,y in st ])
                        }
                        for st in strokes
                    ]
                })

                fname = 'test-%06d-%s-%s.svg' % (data_index, char, 'train' if is_train else 'test')
                Path(fname).write_text(svg)
                #print(svg)
                
                g_minx = min(g_minx, min([ np.array(st)[:,0].min() for st in strokes]))
                g_maxx = max(g_maxx, min([ np.array(st)[:,0].max() for st in strokes]))
                g_miny = min(g_miny, min([ np.array(st)[:,1].min() for st in strokes]))
                g_maxy = max(g_maxy, min([ np.array(st)[:,1].max() for st in strokes]))
                print("BOUNDS:", g_minx, g_maxx, g_miny, g_maxy)
                            
load_uji2()
