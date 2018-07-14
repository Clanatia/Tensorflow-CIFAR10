import sys
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen
import cv2
import numpy as np
url = 'http://192.168.0.2:8080/shot.jpg'   
imgResp = urlopen(url)
imgNp1=np.array(bytearray(imgResp.read()),dtype=np.uint8)
img1=cv2.imdecode(imgNp1,-1)
cv2.imshow('img', img1)    

cv2.waitKey(1)

cv2.destroyAllWindows()

import tensorflow as tf

from include.data import get_data_set
from include.model import model
x, y, output, y_pred_cls, global_step, learning_rate = model()
Label = ['ariplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
saver = tf.train.Saver()
sess = tf.Session()

try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())
    
img = cv2.resize(img1,(32,32))
img = img.reshape([1, 32*32*3])    
img = np.array(img, dtype=float) / 255.0
print(img.shape)

result = sess.run(output, feed_dict={x:img})

print(result)
def findnum(array):
    maxn = max(array)
    maxnumb = 0
    for epoch in array:
        if epoch == maxn:
            break;
        maxnumb+=1
    return maxnumb

num = findnum(result[0])
#print("%s" % Label[num])
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1,Label[num],(10,10), font, 4,(255,255,255),2,cv2.LINE_AA)
cv2.imshow('img', img1)   

    

