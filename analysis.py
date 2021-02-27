'''
Will analyze alpr log files
'''
import pandas as pd
import json
import os
from parse import compile as pcompile, parse
import glob
import csv


def parse_info(fname):
    '''
    given file name, open it, parse total # of frames per vehicle, only for ones with detected plate from the file and
    put it into df
    '''
    data = {"car_id":[], "total_frame":[]}
    with open(fname, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            #params = p.parse(line)
            print("Line:", len(row))
            #print("params:", params)
            #try:
            #    print(params.named["detected"])
            #except AttributeError:
            #    print("No match found")

file_dir_path = 'alpr_logs'
file_name_pat  = 'alpr.log*'

log_file_list = glob.glob(os.path.join(file_dir_path, file_name_pat))
ha = 'vehicle_{car_num:d}.pickle'
log_msg_fmt ='''{datetime:w} - cam_id: {cam_id:w}, v_id: {vehicle_id:d}, try_num / total_frames: {num_try:d}/{total_frames:d}, plate?: {detected:w}, credit_used: {credit_used}/{total_credit}, detected_vehicle_info: {vehicle_dic}, tot: {motion_score}, lp: {blur_score}, max_retry: {max_retry}, plate_visibility_check: {visibility_coord_dic}, instance: {instance_id}, frame_id: {frame_id}'''
print(log_msg_fmt)
p = pcompile(log_msg_fmt)

for fname in log_file_list:
    print(fname)
    parse_info(fname)


