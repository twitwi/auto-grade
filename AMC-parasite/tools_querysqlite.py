
from imageio import imread
import io
import sqlite3

def make_connection(p):
    return sqlite3.connect(p)
def asbase64(im):
    import base64
    return base64.b64encode(im).decode('ascii')
def assuch(im): return im
def preload_all_queries(conn, more='', improcess=asbase64):
    c = conn.cursor()
    c.execute('''SELECT id_a,student,* FROM capture_zone WHERE type=4 AND total > 1.5 * (SELECT MIN(total) FROM capture_zone WHERE total>0) '''+more+''' ORDER BY student,id_a,id_b ASC''')
    # criteria on total is a tentative to remove non-OCR checkboxes
    res = {}
    for r in c.fetchall():
        k = r[1]
        if not (k in res):
            res[k] = []
        #r = r[:-1] + (imread(io.BytesIO(r[-1])),)
        #import pytesseract
        #print(pytesseract.image_to_string(imread(io.BytesIO(r[-1])), config='--psm 10'))
        # ^ doesn't seem to work too well on this data
        r = r[:-1] + (improcess(r[-1]),)
        res[k].append(r)
    return res
