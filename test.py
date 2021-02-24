'''
test out motion detection
'''
import models 
import cv2

md = models.MotionDetector()

video = 'ゲートカメラ11_28.mp4'
# initialize the camera and grab a reference to the raw camera capture
avg = None
previous_x_cen = 0

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video)
frame_cnt = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    mask, thresh, result_frame = md.detect_motion(frame, frame_cnt)
    if mask is None:
        continue

    cv2.imshow('thresh', thresh)
    cv2.imshow('frameDelta', mask)
    cv2.imshow('video', result_frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

    frame_cnt += 1

cap.release()
cv2.destroyAllWindows()