import numpy as np
import tensorflow as tf

from include.data import get_data_set
from include.model import model


test_x, test_y = get_data_set("test")
x, y, output, y_pred_cls, global_step, learning_rate = model()


_BATCH_SIZE = 128
_CLASS_SIZE = 10
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"

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


def main():
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

    import cv2
    img = cv2.imread("cat.png")
    img = cv2.resize(img,(32,32))
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
    print("%s" % Label[num])


if __name__ == "__main__":
    main()


sess.close()
