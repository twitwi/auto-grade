
from vuejspython import model, start, atomic
import config
from tools_generic import read_config, parse_xlsx_first_page, parse_xlsx_generic, add_xlsx_sheet_keyed_by_examid
from tools_images import load_images
from tools_querysqlite import make_connection, get_scanzone_position

import os
import shutil
from datetime import datetime


@model
class ExamGrade:
    props = ['project_path']
    project_path = ''

    # reactive properties
    cfg = {}
    raw_boxes = {}

    def load_yaml_config(self):
        self.cfg = read_config(self.project_path)
        fields = self.cfg['fields']

    def load_xlsx(self):
        headers = self.cfg['headers']
        xlfile = self.project_path + '/parasite.xlsx'
        dr = parse_xlsx_generic(xlfile, sheet_index=-1)
        print(dr)
        if dr[0][0] == 'examid':
            return dr
        return []

    def save_xlsx(self, data):
        headers = self.cfg['headers']
        xlfile = self.project_path + '/parasite.xlsx'
        add_xlsx_sheet_keyed_by_examid(xlfile, data)

    def load_boxes(self, for_q):
        fields = self.cfg['fields']
        if for_q in self.raw_boxes: return
        # avoid writing to the default shared {} by making a copy
        self.raw_boxes = {**self.raw_boxes, for_q: []}
        options = {'predict': True, 'onlyq': for_q }
        self.raw_boxes = {
            **self.raw_boxes,
            for_q: load_images(self.project_path, options),
        }
        #del self.raw_boxes[for_q]
        print(self.raw_boxes)

    def load_scanzone_position(self, zoneid):
        conn = make_connection(self.project_path + '/data/capture.sqlite')
        bbox = get_scanzone_position(conn, zoneid)
        return bbox

@model
class ExamIdentify:
    props = ['project_path']
    project_path = ''

    # reactive properties
    cfg = {}
    data_rows = []
    guess = {}
    raw_boxes = {}

    def load_info(self):
        self.load_yaml_config()
        self.load_xlsx({})
        self.load_boxes()

    def load_yaml_config(self):
        self.cfg = read_config(self.project_path)

    def load_xlsx(self, data):
        headers = self.cfg['headers']
        xlfile = self.project_path + '/parasite.xlsx'
        self.data_rows = parse_xlsx_first_page(xlfile, data, headers, lambda a,b: self.log(a, b))
        #
        guess = {}
        for i,r in enumerate(self.data_rows):
            if r[4] != None:
                guess[i] = str(r[4]) # We use string so it is ready for indexing boxes[]
        self.guess = guess

    def save_xlsx(self):
        annotated_rows = self.data_rows.copy()
        for ind in self.guess.keys():
            annotated_rows[ind][4] = int(self.guess[ind])
        headers = self.cfg['headers']
        xlfile = self.project_path + '/parasite.xlsx'
        parse_xlsx_first_page(xlfile, {'write': annotated_rows}, headers, lambda a,b: self.log(a, b))

    def load_boxes(self):
        fields = self.cfg['fields']
        if fields['lastname'][0] != fields['firstname'][0]:
            self.log('error', 'Unimplemented: firstname and lastname in different questions')
            return
        options = {'predict': True, 'onlyq': fields['lastname'][0] }
        self.raw_boxes = load_images(self.project_path, options)

    def affect_user_to_row(self, user, row):
        guess = self.guess.copy()
        guess[row] = str(user)
        self.guess = guess

@model
class AMCParasite:
    # reactive properties
    local_MC = config.local_MC
    project_path = config.default_project_dir
    debug_logs = []

    def computed_project_full_path(self):
        if self.project_path.startswith('/'):
            return self.project_path
        p = self.local_MC
        if p[-1] != '/':
            p += '/'
        return p + self.project_path

    def __init__(self, argv):
        if len(argv) > 1:
            self.project_path = argv[1]
            if self.project_path.startswith(self.local_MC):
                self.project_path = self.project_path[len(self.local_MC):]
    
    def log(self, type, msg):
        self.debug_logs.append((type, msg))

    def save_miniset(self, l):
        dir = 'miniset-'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M')
        os.mkdir(dir)
        for c, impath in l:
            cdir = dir + '/class-' + c
            if not os.path.exists(cdir):
                os.mkdir(cdir)
            shutil.copy2(impath, cdir)


import sys
start(AMCParasite(sys.argv))
