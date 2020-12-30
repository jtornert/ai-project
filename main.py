import glob
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import emnist
import random

import models
from util import printPrediction, invlabel

# import training and test samples
train_images_digits, train_labels_digits = emnist.extract_training_samples(
    'digits')
train_images_letters, train_labels_letters = emnist.extract_training_samples(
    'letters')
test_images_digits, test_labels_digits = emnist.extract_test_samples('digits')
test_images_letters, test_labels_letters = emnist.extract_test_samples(
    'letters')

# update labels for letters to be compatible with neural network model
train_labels_letters = [x + 9 for x in train_labels_letters]
train_labels_letters = np.array(train_labels_letters)
test_labels_letters = [x + 9 for x in test_labels_letters]
test_labels_letters = np.array(test_labels_letters)

# normalize sample arrays
train_images_digits = tf.keras.utils.normalize(train_images_digits, axis=1)
train_images_letters = tf.keras.utils.normalize(train_images_letters, axis=1)
test_images_digits = tf.keras.utils.normalize(test_images_digits, axis=1)
test_images_letters = tf.keras.utils.normalize(test_images_letters, axis=1)

# create neural network model
model = models.dual(load=True)

# train model
# models.train(model,
#              np.concatenate((train_images_digits, train_images_letters)),
#              np.concatenate((train_labels_digits, train_labels_letters)),
#              epochs=3,
#              save=True)

# testing

print('\nDigits:')

indices = [random.randint(0, 10000) for i in range(9)]

for i in indices:
    img = np.array(test_images_digits[i], dtype='float').reshape((28, 28))
    prediction = model.predict(np.array([img]))
    printPrediction(test_labels_digits[i], np.argmax(prediction))
    plt.imshow(img, cmap='gray_r')
    plt.show()

print('\nLetters:')

indices = [random.randint(0, 10000) for i in range(9)]

for i in indices:
    img = np.array(test_images_letters[i], dtype='float').reshape((28, 28))
    prediction = model.predict(np.array([img]))
    printPrediction(test_labels_letters[i], np.argmax(prediction))
    plt.imshow(img, cmap='gray_r')
    plt.show()

print('\nTest images:')

for filepath in glob.iglob(r'./img/*.png'):
    img = cv.imread(filepath)[:, :, 0]
    img = np.invert([img])
    prediction = model.predict(img)
    printPrediction(ord(filepath[6:7]) - ord('0'), np.argmax(prediction))
    plt.imshow(img[0], cmap='gray_r')
    plt.show()

print('\nNIST images:')

NIST_labels = [['0', 'O', 'o'],
               ['1', '7', 'I', 'i', 'l'],
               ['6', '8', 'S', 's']]

i = 0
for filepath in glob.iglob(r'./nist/*'):
    # print(filepath)
    folders = glob.glob(filepath + r'/*')
    symbols = [x for x in range(len(folders))]
    random.shuffle(symbols)
    for symbol in symbols:
        # print(folders[symbol])
        testfolders = glob.glob(folders[symbol] + r'/*')
        # print(testfolders)
        testfolder = testfolders[random.randint(0, len(testfolders) - 1)]
        images = glob.glob(testfolder + r'/*')
        file = images[random.randint(0, len(images) - 1)]
        img = cv.imread(file)[:, :, 0]
        img = img[28:100, 28:100]
        img = cv.resize(img, (28, 28), interpolation=cv.INTER_LANCZOS4)
        img = np.invert([img])
        img = tf.keras.utils.normalize(np.array(img))
        prediction = model.predict(img)
        printPrediction(invlabel(NIST_labels[i][symbol]),
                        np.argmax(prediction))
        plt.imshow(img[0], cmap='gray_r')
        plt.show()
    i = i + 1

# NIST images:
# ./nist\0Oo
# ./nist\0Oo\30
# ./nist\0Oo\4f
# ./nist\0Oo\6f
# ./nist\17Ili
# ./nist\17Ili\31
# ./nist\17Ili\37
# ./nist\17Ili\49
# ./nist\17Ili\69
# ./nist\17Ili\6c
# ./nist\8Ss6
# ./nist\8Ss6\36
# ./nist\8Ss6\38
# ./nist\8Ss6\53
# ./nist\8Ss6\73

print('Done')
