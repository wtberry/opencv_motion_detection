'''
utility files
'''
import pytz
from datetime import datetime
import cv2
import os
import pickle


def convert_ts(ts, config):
    '''Converts a timestamp to the configured timezone. Returns a localized datetime object.'''
    #lambda_tz = timezone('US/Pacific')
    tz = pytz.timezone(config['timezone'])
    utc = pytz.utc

    utc_dt = utc.localize(datetime.utcfromtimestamp(ts))

    localized_dt = utc_dt.astimezone(tz)

    return localized_dt

def current_utc():
    '''
    return unix timestamp at utc
    '''
    utc_dt = pytz.utc.localize(datetime.now())
    now_ts_utc = (utc_dt - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
    return now_ts_utc


class Frame(object):
    '''
    Frame class for frame instances
    '''

    def __init__(self, frame_arr, approx_captured_timestamp, frame_cnt, camera_id, camera_name):
        self.frame_arr = frame_arr
        self.processed_timestamp = None
        self.approx_captured_timestamp = approx_captured_timestamp
        self.frame_cnt = frame_cnt
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.vehicle_id = None
        self.motion_score = None
        self.plate = None
        self.blur_score = None
        self._file_name_fmt_str = 'time_{time}_frame-cnt_{frame_cnt}_car_{car_id}_blur_{blur_score}_motion_{motion_score}.jpg'

    
    #def save(self, file_dir, car_id='None', file_name='None'):
    #    '''
    #    save image to the given file_dir, which if does not exist create one
    #    '''
    #    if file_name == "None":
    #        file_name = self._file_name_fmt_str.format(time=self.approx_capture_timestamp, frame_cnt=self.frame_cnt, car_id=car_id, blur_score=self.blur_score, motion_score=self.motion_score)
    #    os.makedirs(file_dir, exist_ok=True)
    #    cv2.imwrite(os.path.join(file_dir, file_name), frame)
    #    print("Saved image in {}".format(os.path.join(self.img_dir, file_name)))


    def to_img_bytes(self):
        '''convert frame arr to image byte and return'''
        ret, jpg = cv2.imencode('.jpg', self.frame_arr)
        return bytearray(jpg)


class Vehicle(object):
    '''
    Represent each detected vehicle in video stream
    '''
    def __init__(self, id, frame_obj):
        self.id = id
        self.plate = {}
        self.detected = False
        self.frame_list = []
        self.first_timestamp = frame_obj.approx_captured_timestamp
        self.last_timestamp = 0
        self.camera_id = frame_obj.camera_id
        self._best_img_idx = 0 # private

        self.add_frame(frame_obj)


    def add_frame(self, frame):
        '''
        add new frame to the vehicle list and sort by blur score
        '''
        frame.vehicle_id = self.id
        self.frame_list.append(frame)
        self.frame_list = sorted(self.frame_list, key=lambda x: x.blur_score, reverse=True)

    def get_frame(self):
        '''
        return the best frame at the moment (1st, 2nd or 3rd..) using index.
        Keep all the frames tho
        '''
        try:
            best_img = self.frame_list[self._best_img_idx]
        except IndexError as IE:
            print("No More Best frames in car list")
            return None
        else:
            self._best_img_idx += 1
            return best_img

    def save_all_frames(self, path):
        '''
        save all of the frames in the vehicle under specified path with
        img_file_format = 'time_{time}_frame-cnt_{frame_count}_car_{car_num:d}_blur_{blur_score:g}_motion_{motion_score:d}.jpg' fmt.
        '''
        for frame in self.frame_list:
            frame.save(path, self.id)
   
    def pickle(self, file_root, config):
        '''pickle and save the vehicle obj under file_root'''

        first_datetime = convert_ts(self.first_timestamp, config)
        path = os.path.join(file_root, str(first_datetime.year), str(first_datetime.month), str(first_datetime.day), self.camera_id)
        print("path:", path)
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, 'vehicle_'+str(self.id)+'.pickle')
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

