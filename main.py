import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

import models

(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.mnist.load_data()

train_images = tf.keras.utils.normalize(train_images, axis=1)
test_images = tf.keras.utils.normalize(test_images, axis=1)

model = models.simplest()
models.load_model(model, 'simplest')
models.train_simplest(model, train_images, train_labels, 5, save=True)

test = model.predict(test_images)

for i in range(9):
    img = np.array(test_images[i], dtype='float').reshape((28, 28))
    print(f'Label: {test_labels[i]}\tPredicted: {np.argmax(test[i])}')
    plt.imshow(img, cmap='gray_r')
    plt.show()

print('\nTest images:')

for filepath in glob.iglob(r'./img/*.png'):
    img = cv.imread(filepath)[:, :, 0]
    img = np.invert([img])
    prediction = model.predict(img)
    print(f'Label: {filepath[6:7]}\tPredicted: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap='gray_r')
    plt.show()
