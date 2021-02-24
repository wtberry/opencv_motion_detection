import cv2

'''open image and draw bounding box
'''
images = ['camera_stored-video_car_35.jpg', 'camera_stored-video_car_37.jpg','camera_stored-video_car_38.jpg']
coords = [{"y": 266, "x": 31, "height": 197, "width": 282}, {"y": 35, "x": 67, "height": 673, "width": 1111},{"y": 6, "x": 107, "height": 701, "width": 984}]



def draw_show(i, image, coord):

    img = cv2.imread(image)
    
    x= coord['x']
    y= coord['y']
    h= coord['height']
    w= coord['width']
    
    
    
    x_box=int(0.33*1280)
    y_box=int(0.35*720)
    h_box=int(0.6*720)
    w_box=int(0.43*1280)
    
    x_min = x_box
    y_min = y_box
    x_max = x_min+w_box
    y_max = y_min+h_box
    
    print("x_overlap:", x_max, x)
    print("x_overlap:", x+w, x_min)
    
    cv2.rectangle(img,(x_box,y_box),(x_box+w_box,y_box+h_box),(0,0,255),2)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img, 'y: {}'.format(y), (x, y+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.putText(img, 'x: {}, h: {}'.format(x, h), (x, y+h-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.putText(img, 'w: {}'.format(w), (x+w, y+h-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.imshow("vehicle: {}".format(i), img)

i = 0
for image, coord in zip(images, coords):
    draw_show(i, image, coord)
    i+=1

key = cv2.waitKey(0)
if key == ord('q'):
    cv2.destroyAllWindows()


