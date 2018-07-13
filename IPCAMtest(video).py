import cv2
import numpy as np

cam = cv2.VideoCapture()
cam.open('http://192.168.0.5:8080/video')

while(True):
    ret,img = cam.read()
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cam.release()
cv2.destroyAllWindows()
