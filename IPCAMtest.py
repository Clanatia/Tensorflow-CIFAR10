'''import sys
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen
import cv2
import numpy as np
url = 'http://192.168.0.5:8080/shot.jpg’
while(True):
    imgResp =urlib.urlopen(url)
    imgNp1=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img1=cv2.imdecode(imgNp1,-1)
    cv2.imshow('img', img1)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
cv2.imwrite(img1,'img.jpg')
cv2.destroyAllWindows()'''

import cv2
import numpy as np

cam = cv2.VideoCapture()
cam.open('http://192.168.0.5:8080/shot.jpg')
ret,img = cam.read()
cv2.imshow('img',img)
    
if cv2.waitKey(1

cv2.imwrite(img1,'img.jpg')
cam.release()
cv2.destroyAllWindows()
