import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import cv2

mnist = tf.keras.datasets.mnist
(image_train, label_train), (image_test, label_test) = mnist.load_data()
model = tf.keras.models.load_model('ANN_mnist.h5')

NUM = 5
predict = model.predict(image_test[0:NUM])
print(predict)

print(" * Prediction,", np.argmax(predict, axis = 1))

plt.figure(figsize=(15, 15))
for idx in range(NUM):
    sp = plt.subplot(1, 5, idx+1)
    plt.imshow(image_test[idx])
plt.show()