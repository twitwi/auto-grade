
from scipy.misc import imread, imsave
import io
import sqlite3

def show(a):
    from matplotlib import pyplot as plt
    plt.imshow(a, cmap='gray', interpolation='nearest')
    plt.show()

def show_auto_grid(l):
    from matplotlib import pyplot as plt
    import math
    N = math.ceil(len(l)**0.5)
    M = math.ceil(len(l)/N)
    f, axarr = plt.subplots(M, N)
    f.subplots_adjust(hspace=0)
    for i,a in enumerate(l):
        axarr[i//N, i%N].imshow(a, cmap='gray', interpolation='nearest')
        axarr[i//N, i%N].axis('off')
        imsave(",,test-"+str(i)+".jpg", a)
    for i in range(len(l), N*M):
        axarr[i//N, i%N].axis('off')
    plt.show()


conn = sqlite3.connect('capture.sqlite')
c = conn.cursor()

user = 1
c.execute('''SELECT *
             FROM capture_zone
             WHERE student=? AND type=?
             LIMIT 100
          ''',
#             ORDER BY black DESC
          (user, 4))


boxes = c.fetchall()
#print(darkest)
ims = [imread(io.BytesIO(b[-1])) for b in boxes]
show_auto_grid(ims)
