'''
process given video with model
'''
import cv2
import os
import pandas as pd

import utils
from models import MotionDetector
#from motion_detection_avg import Motion



video = 'adjusted_gatecam_20min.mp4'

capture_rate = 300
frame_cnt = 0
writer = None
output = 'output.mp4'
motion_detector = MotionDetector(img_threshold=0, coordinates=(0.33,0.35,0.6,0.43))
video = cv2.VideoCapture(video)
save_video = False
save_img = True
img_dir = 'exp_img'
save_data = False

data = []

while True:
    if frame_cnt % 900 == 0:
        # slack notification here
        if frame_cnt != 0:
            break
        print("processing {}th frmae for video: {}".format(frame_cnt, video))

    grabbed, frame = video.read()
    # check if it is the end of the video
    if not grabbed:
        # end of the clip
        if writer != None:
            writer.release()
        cv2.destroyAllWindows()
        break
    if frame_cnt % capture_rate == 0:
        now_ts_utc = utils.current_utc() # unix timestamp
        frame_obj = utils.Frame(frame, now_ts_utc, frame_cnt, 'stored-video', video)
        is_new_car, frame_obj = motion_detector.detect_motion(frame_obj, draw=True)
        frame = frame_obj.frame_arr
        print("Frame shape:", frame.shape)

        # check if the video writer is None
        if writer is None and save_video:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output, fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)
        if save_video:
            writer.write(frame)

        if save_img:
            os.makedirs(img_dir, exist_ok=True)
            cv2.imwrite(os.path.join(img_dir, 'frame_{}.jpg'.format(frame_cnt)), frame)

        if save_data:
            datum = {'frame_cnt': frame_cnt, 'tot':frame_obj.motion_score, 'lp':frame_obj.blur_score, 'x':motion_detector.x_cord, 'y':motion_detector.y_cord, \
                        'h':motion_detector.h_cord, 'w':motion_detector.w_cord}
            data.append(datum)

    frame_cnt += 1

if save_data:
    df = pd.DataFrame(data, columns=['frame_cnt','tot', 'lp', 'x', 'y', 'h', 'w'])
    df.to_csv('base_data.csv', index=False)

    #cv2.imshow("Video", frame)   
    # if the 'q' key is pressed then break from the loop
    #key = cv2.waitKey(1) & 0xFF
    #if key == ord('q'):
    #    if writer:
    #        writer.release()
    #    cv2.destroyAllWindows()
    #    break