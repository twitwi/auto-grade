
#python3 script_histo_of_total_box_sizes.py ~/MC-Projects/*/data/capture.sqlite


import sqlite3
import sys
import numpy as np
import matplotlib.pyplot as plt


for i,f in enumerate(sys.argv[1:]):
    conn = sqlite3.connect(f)

    c = conn.cursor()
    c.execute('''SELECT total, count(*) FROM capture_zone WHERE type=4 group by total''')
    arr = np.array(c.fetchall())
    if sum(np.shape(arr))==0:
        print(f)
        continue
    print(np.min(arr, axis=0), np.max(arr, axis=0))
    plt.scatter(
            arr[:,0] / np.min(arr[:,0]), # normalize wrt to min
        i + arr[:,1] / np.sum(arr[:,1]), # show as a distribution
        marker='.')
    conn.close()

plt.plot([1.5]*2, [0, len(sys.argv)]) # draw the selected separation
plt.show()

