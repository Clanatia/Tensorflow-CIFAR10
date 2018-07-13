import sys
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen
import cv2
import numpy as np
url = 'http://192.168.0.2:8080/shot.jpg'
'''
while(True):
    imgResp = urlopen(url)
    imgNp1=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img1=cv2.imdecode(imgNp1,-1)
    cv2.imshow('img', img1)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break      
'''      
imgResp = urlopen(url)
imgNp1=np.array(bytearray(imgResp.read()),dtype=np.uint8)
img1=cv2.imdecode(imgNp1,-1)
cv2.imshow('img', img1)    

cv2.imwrite('img.jpg',img1)
cam.release()
cv2.destroyAllWindows()
