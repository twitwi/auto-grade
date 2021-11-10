
import sys
from mxnet.gluon.data.vision.datasets import ImageFolderDataset # might replace by glob...

our_classes = "=:;.,-_()[]!?*/'+⁹"
emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
classes = emnist_classes + our_classes

def folder_to_class(fold):
    ncl = fold.replace(r'class-', '')
    if ncl == '': ncl = '/'
    try:
        icl = classes.index(ncl)
    except:
        try:
            icl = classes.index(ncl.upper())
        except:
            print(ncl, "is not a class... replacing by ⁹")
            ncl = '⁹'
            icl = classes.index(ncl) # dummy class.....
    return icl, ncl


def make_dataset_viewer(reldir):
    import urllib.parse
    dir = reldir[:-1] if reldir.endswith('/') else reldir
    class ListDict(dict):
        def __missing__(self, key):
            self[key] = []
            return self[key]

    ifd = ImageFolderDataset(dir, flag=0)
    class_images = ListDict()

    for ii, (f, cl) in enumerate(ifd.items):
        icl, ncl = folder_to_class(ifd.synsets[cl])
        class_images[icl].append(f)

    base = dir.split('/')[-1]
    def relpath(im):
        return urllib.parse.quote(im[im.index(base):])
    def imgs(l):
        return '\n'.join((f'<img src="{relpath(im)}"/>' for im in l))
        
    body = ''.join((
        f"""
        <div>Class {icl} <span>{classes[icl]}</span> ({len(class_images[icl])})</div>
        {imgs(class_images[icl])}
        """
        for icl in sorted(class_images.keys())
    ))

    ofname = dir+'.html'
    with open(ofname, 'w') as f:
        print("""
        <html>
        <head>
            <style>
            div span { border: 1px solid black; background: gray; font-size: 150%; padding: 0.2em; }
            </style>
        </head>
        <body>
           <div>All """+str(sum([len(l) for l in class_images.values()]))+"""</div>
        """ + body + """
        </body>
        </html>
        """, file=f)
    print("Generated:", ofname)

        
for p in sys.argv[1:]:
    make_dataset_viewer(p)
