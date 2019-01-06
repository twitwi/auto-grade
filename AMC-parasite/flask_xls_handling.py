
from flask import Flask, Blueprint, render_template, send_from_directory
from flask_socketio import SocketIO, join_room, emit

import time
import os
import shutil
import codecs
import openpyxl
from itertools import islice

from tools_querysqlite import *
from tools_usetorch import *
from tools_generic import *

from flask_app import socketio

@socketio.on('xlsx-structured-rows')
def on_xlsx_read_rows(data):
    cfg = read_config(local_MC + data['pro'])
    xlfile = local_MC + data['pro'] + '/parasite.xlsx'
    wb = openpyxl.load_workbook(filename = xlfile)
    ws = wb[wb.sheetnames[0]]
    def digest_header_into_index_access():
        headers = lmap(attr_value(), ws['1'])
        headers_lookup = { v: i for i,v in enumerate(headers) }
        cfg_mapped = lmap(lambda h: cfg['headers'][h], row_elements)
        return lmap(of(headers_lookup), cfg_mapped)

    row_indices = digest_header_into_index_access()
    res = []
    for i, row in islice(enumerate(ws.rows), 1, None):
        cells = lmap(of(row), row_indices)
        res.append(lmap(attr_value(), cells))
    emit('got-xlsx-structured-rows', res)

@socketio.on('yaml-config')
def on_get_yaml_config(data):
    cfg = read_config(local_MC + data['pro'])
    emit('got-yaml-config', cfg)
