import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

# model.save('digits.model')

test = model.predict(x_test)

for i in range(9):
  img = np.array(x_test[i], dtype='float').reshape((28,28))
  print(f'Label: {y_test[i]}\tPredicted: {np.argmax(test[i])}')
  plot.imshow(img, cmap='gray_r')
  plot.show()

img = cv.imread('7_1.png')[:,:,0]
img = np.invert([img])
prediction = model.predict(img)
print(np.argmax(prediction))
plot.imshow(img[0], cmap='gray_r')
plot.show()