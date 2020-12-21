import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot
import glob
import os

from tensorflow.keras import callbacks

(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.mnist.load_data()

train_images = tf.keras.utils.normalize(train_images, axis=1)
test_images = tf.keras.utils.normalize(test_images, axis=1)

checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.load_weights(checkpoint_path)
model.fit(train_images,
          train_labels,
          epochs=40,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

test = model.predict(test_images)

for i in range(9):
    img = np.array(test_images[i], dtype='float').reshape((28, 28))
    print(f'Label: {test_labels[i]}\tPredicted: {np.argmax(test[i])}')
    plot.imshow(img, cmap='gray_r')
    plot.show()

labels = [1, 2, 4, 5, 5, 6, 7, 7, 8]
i = 0
print('\nTest images:')

for filepath in glob.iglob(r'./img/*.png'):
    img = cv.imread(filepath)[:, :, 0]
    img = np.invert([img])
    prediction = model.predict(img)
    print(f'Label: {labels[i]}\tPredicted: {np.argmax(prediction)}')
    i = i + 1
    plot.imshow(img[0], cmap='gray_r')
    plot.show()
