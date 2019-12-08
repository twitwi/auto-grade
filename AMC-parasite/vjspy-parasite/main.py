
from vuejspython import model, start, atomic
import config
from tools_generic import read_config, parse_xlsx
from tools_images import load_images

@model
class AMCParasite:
    # reactive properties
    local_MC = config.local_MC
    project_path = config.default_project_dir
    cfg = {}
    data_rows = []
    guess = {}
    raw_boxes = {}
    debug_logs = []


    def __init__(self, argv):
        if len(argv) > 1:
            self.project_path = argv[1]
    

    def log(self, type, msg):
        self.debug_logs.append((type, msg))

    def load_info(self):
        self.load_yaml_config()
        self.load_xlsx({})
        self.load_boxes()

    def load_yaml_config(self):
        self.cfg = read_config(self.local_MC + self.project_path)

    def load_xlsx(self, data):
        headers = self.cfg['headers']
        xlfile = self.local_MC + self.project_path + '/parasite.xlsx'
        self.data_rows = parse_xlsx(xlfile, data, headers, lambda a,b: self.log(a, b))
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
        xlfile = self.local_MC + self.project_path + '/parasite.xlsx'
        parse_xlsx(xlfile, {'write': annotated_rows}, headers, lambda a,b: self.log(a, b))

    def load_boxes(self):
        fields = self.cfg['fields']
        if fields['lastname'][0] != fields['firstname'][0]:
            self.log('error', 'Unimplemented: firstname and lastname in different questions')
            return
        project_full_path = self.local_MC + self.project_path
        options = {'predict': True, 'onlyq': fields['lastname'][0] }
        self.raw_boxes = load_images(project_full_path, options)

    def affect_user_to_row(self, user, row):
        guess = self.guess.copy()
        guess[row] = str(user)
        self.guess = guess


import sys
start(AMCParasite(sys.argv))
