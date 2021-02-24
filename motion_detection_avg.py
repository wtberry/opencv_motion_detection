# import the necessary packages
import time
import cv2
import numpy as np

#video = '../gate2020-09-23_11-55-00_01min.mp4'
video = 'gatecam_test.mp4'
# initialize the camera and grab a reference to the raw camera capture
avg = None
previous_x_cen = 0

image_shape = (720, 1280)
coordinates = (0.35,0.05,0.8,0.45)


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video)

while True:
    ret, frame = cap.read()
    imgage_shape = frame.shape
    x = int(coordinates[0]*image_shape[1])
    y = int(coordinates[1]*image_shape[0])
    h = int(coordinates[2]*image_shape[0])
    w = int(coordinates[3]*image_shape[1])

    if not ret:
        break

    # convert imags to grayscale &  blur the result
    img = frame[y:y+h, x:x+w].copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # inittialize avg if it hasn't been done
    if avg is None:
        avg = gray.copy().astype("float")
        #rawCapture.truncate(0)
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.05)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    cv2.imshow('frameDelta', frameDelta)
    # coonvert the difference into binary & dilate the result to fill in small holes
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh,None, iterations=2)
    print("Tresh:", thresh.shape)
    # show the result
    cv2.imshow("Delta + Thresh", thresh)
    # find contours or continuous white blobs in the image
    #contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # find the index of the largest contour
    #if len(contours) > 0:
    #    areas = [cv2.contourArea(c) for c in contours]
    #    max_index = np.argmax(areas)
    #    cnt=contours[max_index]   
    #    # draw a bounding box/rectangle around the largest contour
    #    x,y,w,h = cv2.boundingRect(cnt)
    #    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #    area = cv2.contourArea(cnt)
    #    if area > 35000:
    #        # print area to the terminal
    #        print(area)
    #        x_cen = (x+w)/2
    #        # add text to the frame
    #        if x_cen - previous_x_cen < -1:
    #            text =  "Area: {} : {}".format(area, "Left")
    #        elif x_cen - previous_x_cen > 1:
    #            text =  "Area: {} : {}".format(area, "Right")
    #        else:
    #            text =  "Area: {} : {}".format(area, x_cen - previous_x_cen)
    #        previous_x_cen = x_cen
    #        #cv2.putText(frame, "Couter Area: {}".format(area), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    #        cv2.putText(frame, text, (x, y + (h+50)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    # show the frame
    cv2.putText(frame, "sum: {}".format(thresh.sum()), (x, y + (h+50)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.imshow("Video", frame)
    # if the 'q' key is pressed then break from the loop
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
        
cv2.destroyAllWindows()