import glob
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from emnist import extract_test_samples, extract_training_samples
import random

import models
from util import printPrediction

# train_images_digits, train_labels_digits = extract_training_samples(
#     'digits')
train_images_letters, train_labels_letters = extract_training_samples(
    'letters')
# test_images_digits, test_labels_digits = extract_test_samples('digits')
test_images_letters, test_labels_letters = extract_test_samples('letters')

train_labels_letters = [x - 1 for x in train_labels_letters]
train_labels_letters = np.array(train_labels_letters)

test_labels_letters = [x - 1 + ord("a") for x in test_labels_letters]
test_labels_letters = np.array(test_labels_letters)

# train_images_digits = tf.keras.utils.normalize(train_images_digits, axis=1)
train_images_letters = tf.keras.utils.normalize(train_images_letters, axis=1)
# test_images_digits = tf.keras.utils.normalize(test_images_digits, axis=1)
test_images_letters = tf.keras.utils.normalize(test_images_letters, axis=1)

model = models.letters(load=True)

# print('Training on digits:')
# models.train(model, train_images_digits,
#              train_labels_digits, epochs=3, save=True)
# print('Training on letters:')
# models.train(model, train_images_letters,
#              train_labels_letters, epochs=3, save=True)

# test_digits = model.predict(test_images_digits)
test_letters = model.predict(test_images_letters)

# print('Digits:')

# for i in range(9):
#     img = np.array(test_images_digits[i], dtype='float').reshape((28, 28))
#     print(
#         f'Label: {label(test_labels_digits[i])}\tPredicted: {label(np.argmax(test_digits[i]))}')
#     plt.imshow(img, cmap='gray_r')
#     plt.show()

print('Letters:')

indices = [random.randint(0, 10000) for i in range(9)]
print(indices)

for i in indices:
    img = np.array(test_images_letters[i], dtype='float').reshape((28, 28))
    printPrediction(test_labels_letters[i],
                    np.argmax(test_letters[i]) + ord("a"))
    plt.imshow(img, cmap='gray_r')
    plt.show()

# print('\nTest images:')

# for filepath in glob.iglob(r'./img/*.png'):
#     img = cv.imread(filepath)[:, :, 0]
#     img = np.invert([img])
#     prediction = model.predict(img)
#     print(f'Label: {filepath[6:7]}\tPredicted: {label(np.argmax(prediction))}')
#     plt.imshow(img[0], cmap='gray_r')
#     plt.show()
