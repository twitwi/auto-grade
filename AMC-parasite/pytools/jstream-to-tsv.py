import json
import sys

blobs = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        l = json.loads(line)
        for o in l:
            blob = o['blob']
            if blob not in blobs:
                blobs.append(blob)
        print(blobs)
    #for element in content:
    #    results.append(','.join([str(y[1]) for y in element['points']]))

results = ['### ' + '\t'.join(blobs) + '\tMORE']
with open(sys.argv[1], 'r') as f:
    for line in f:
        l = json.loads(line)
        row = ['']*(len(blobs)+1)
        for o in l:
            row[blobs.index(o['blob'])] = o['value']
            if 'NB' in o:
                row[-1] += '[[' + o['blob'] + ': ' + o['NB'] + ']]'
        #print(row)
        results.append('"' + '"\t"'.join(map(lambda s: s.replace('"', '""'), row)) + '"')

with open(sys.argv[1]+'.tsv', 'w') as f:
    f.write('\n'.join(results))
