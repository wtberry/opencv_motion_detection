'''
cascade and yolo detector model wrapper
'''
#camera.py
# import the necessary packages
import cv2
import numpy as np
#from scipy.stats import mode
import os
import time
from io import BytesIO
from base64 import b64decode
import utils


class MotionDetectorAbs(object):
    '''
    opencv's motion detector.
    '''
    def __init__(self,history=10, varThreshold=30, detectShadows=False, img_threshold=15000000, coordinates=(0.0625,0,0.6,0.6), image_shape=(720, 1280, 3)):
        '''
        args:
            history: int, how many histories to compare images up to
            varThreshold: int, ...
            img_threshold: int, motion value's threshold. Anything bigger than this value will be flagged as vehicle
            detectShadows: boolean, detect shadow when motion sencing...
            coordinates: tuple, (x, y, h, w) to look at in image, in portion.
            image shape: tuple, (height, width, rgb)
        '''

        #self.fgbg =  cv2.createBackgroundSubtractorMOG2(history =history, varThreshold =varThreshold, detectShadows =detectShadows)
        self.fgbg =  cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=30)
        self.avg = None
        self.prev_x_cen = 0
        self.coordinates = coordinates
        self.x = int(coordinates[0]*image_shape[1])
        self.y = int(coordinates[1]*image_shape[0])
        self.h = int(coordinates[2]*image_shape[0])
        self.w = int(coordinates[3]*image_shape[1])
        self.img_threshold=img_threshold
        self.previous = 0
        self.img_dir = 'car_img'
        print("Created motion detector!!")

    
    def detect_motion(self, frame, frame_count, new_car_threshold=15):
        '''detect motion, add the detected info to frame obj, finally
        return frame_obj and whether new car was detected or not'''
        # if the 'q' key is pressed then break from the loop
        img = frame[self.y:self.y+self.h, self.x:self.x+self.w].copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        is_new_car = False
        if self.avg is None:
            self.avg = gray.copy().astype("float")
            #rawCapture.truncate(0)     
            return None, None, None

        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, self.avg, 0.15) # larger the alpha (third) value, less previous frames to compare to
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))
        # coonvert the difference into binary & dilate the result to fill in small holes
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, np.ones((10,10),np.uint8),iterations=2)
        contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt=contours[max_index]   
            # draw a bounding box/rectangle around the largest contour
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, "score: {}".format(np.sum(frameDelta)), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            area = cv2.contourArea(cnt)
            if area > 40000:
                # print area to the terminal
                print(area)
                x_cen = (x+w)/2
                # add text to the frame
                if x_cen - self.prev_x_cen < -1:
                    text =  "Area: {} : {}".format(area, "Left")
                elif x_cen - self.prev_x_cen > 1:
                    text =  "Area: {} : {}".format(area, "Right")
                else:
                    text =  "Area: {} : {}".format(area, x_cen - self.prev_x_cen)
                self.prev_x_cen = x_cen
                #cv2.putText(frame, "Couter Area: {}".format(area), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                cv2.putText(frame, text, (x, y + (h+50)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        return frameDelta, thresh, frame


class HaarCascadeCar(object):

    def __init__(self, cls_file="cars.xml", objects=[],ds_factor=1.0):
        cls_file = cls_file
        self.model=cv2.CascadeClassifier(cls_file)
        self.objects = objects
        self.ds_factor=ds_factor

    def detect(self,frame):
            '''given input image byte string, decode it into img and 
            do some processing with cv2
            '''
            frame=cv2.resize(frame,None,fx=self.ds_factor,fy=self.ds_factor,
            interpolation=cv2.INTER_AREA)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            obj_rects=self.model.detectMultiScale(gray,1.1,1)
            objs = []
            for (x,y,w,h) in obj_rects:
                obj = {}
                bounding_box = {}
                obj["Name"] = 'Car'
                bounding_box['x'] = int(x/self.ds_factor)
                bounding_box['y'] = int(y/self.ds_factor)
                bounding_box['w'] = int(w/self.ds_factor)
                bounding_box['h'] = int(h/self.ds_factor)
                obj["BoundingBox"] = bounding_box
                objs.append(obj)
            return {"Labels":objs}



class MotionDetector(object):
    '''
    opencv's motion detector.
    '''
    def __init__(self,history=10, varThreshold=30, detectShadows=False, img_threshold=15000000, coordinates=(0.0625,0,0.6,0.6), image_shape=(720, 1280, 3)):
        '''
        args:
            history: int, how many histories to compare images up to
            varThreshold: int, ...
            img_threshold: int, motion value's threshold. Anything bigger than this value will be flagged as vehicle
            detectShadows: boolean, detect shadow when motion sencing...
            coordinates: tuple, (x, y, h, w) to look at in image, in portion.
            image shape: tuple, (height, width, rgb)
        '''

        self.fgbg =  cv2.createBackgroundSubtractorMOG2(history =history, varThreshold =varThreshold, detectShadows =detectShadows)
        self.x = int(coordinates[0]*image_shape[1])
        self.y = int(coordinates[1]*image_shape[0])
        self.h = int(coordinates[2]*image_shape[0])
        self.w = int(coordinates[3]*image_shape[1])
        self.x_cord = coordinates[0]
        self.y_cord = coordinates[1]
        self.h_cord = coordinates[2]
        self.w_cord = coordinates[3]
        self.img_threshold=img_threshold
        self.car_id = 0
        self.previous = 0
        self.img_dir = 'car_img'
        print("Created motion detector!!")

    
    def detect_motion(self, frame_obj, new_car_threshold=15, draw=False):
        '''detect motion, add the detected info to frame obj, finally
        return frame_obj and whether new car was detected or not'''

        frame_count = frame_obj.frame_cnt
        frame = frame_obj.frame_arr
        cam_id = frame_obj.camera_id
        curr_time = frame_obj.approx_captured_timestamp
        img = frame[self.y:self.y+self.h, self.x:self.x+self.w].copy()
        is_new_car = False
        fgmask = self.fgbg.apply(img)
        tot = np.sum(fgmask)
        lp = "None"
        if tot > self.img_threshold:
            lp = cv2.Laplacian(img, cv2.CV_64F).var()
            # save info in frame
            if frame_count - self.previous > new_car_threshold:
                is_new_car = True
                self.car_id += 1
            self.previous = frame_count
        if draw:
            text = "tot: {}, Lap: {}".format(tot, lp)
            text2 = "Car: {}".format(self.car_id)
            cv2.putText(frame, text, (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            cv2.putText(frame, text2, (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            cv2.putText(frame, 'x: {}, h: {}'.format(self.x_cord, self.h_cord), (self.x, self.y+self.h+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            cv2.putText(frame, 'w: {}'.format(self.w_cord), (self.x+self.w, self.y+self.h+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            cv2.rectangle(frame,(self.x,self.y),(self.x+self.w,self.y+self.h),(0,255,0),2)
        frame_obj.processed_timestamp = utils.current_utc()
        frame_obj.motion_score = tot
        frame_obj.blur_score = lp
        return is_new_car, frame_obj