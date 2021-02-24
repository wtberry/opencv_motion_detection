'''
process given video with model
'''
import cv2
import os
import glob

from models import MotionDetector, HaarCascadeCar

img_dir = '/home/wataru/Projects/object-detection-deep-learning/images'
img_list = glob.glob(os.path.join(img_dir, 'camera_ゲートカメラ_car_*.jpg'))
print("IMG LIST:", img_list)
coordinates =(0.33,0.35,0.6,0.43)

motion_detector = HaarCascadeCar()

for img_file in img_list:
    img = cv2.imread(img_file)
    labels = motion_detector.detect(img)

    # draw on img
    labels = labels["Labels"]
    print("labels:", labels)
    img_name = "_".join(img_file.split("_")[-2:])
    for label in labels:
        name = label['Name']
        box = label["BoundingBox"]
        cv2.rectangle(img,(box['x'],box['y']),(box['x']+box['w'],box['y']+box['h']),(0,255,0),2)
    cv2.imshow(img_name, img)   
    # if the 'q' key is pressed then break from the loop
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break