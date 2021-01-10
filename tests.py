import numpy as np
import matplotlib.pyplot as plt
import emnist
import util
import random
import glob
import cv2 as cv
import tensorflow as tf

test_images_all, test_labels_all = emnist.extract_test_samples('byclass')
test_images_digits, test_labels_digits = emnist.extract_test_samples('digits')
test_images_letters, test_labels_letters = emnist.extract_test_samples(
    'letters')

test_images_all = tf.keras.utils.normalize(test_images_all, axis=1)
test_images_digits = tf.keras.utils.normalize(test_images_digits, axis=1)
test_images_letters = tf.keras.utils.normalize(test_images_letters, axis=1)

show_all = False


def rand(model, iter=None):
    iterations = 0
    if iter == None:
        print('\nRandom test, enter a value between 1 and 20:', end=" ")
        iterations = int(input())
    else:
        print('\nRandom test:')
        iterations = iter
    if iterations < 1:
        iterations = 1
    elif iterations > 20:
        iterations = 20
    indices = [random.randint(0, 116323 - 1) for i in range(iterations)]

    for index in indices:
        img = np.array(test_images_all[index], dtype='float').reshape((28, 28))
        prediction = model.predict(np.array([img]))
        util.printPrediction(test_labels_all[index], np.argmax(prediction))
        if util.label(test_labels_all[index]) == util.label(np.argmax(prediction)):
            if show_all == False:
                continue
        plt.imshow(img, cmap='gray_r')
        plt.show()


def test(model):
    print('\nTest images:')

    for filepath in glob.iglob(r'./img/*.png'):
        img = cv.imread(filepath)[:, :, 0]
        img = np.invert([img])
        prediction = model.predict(img)
        util.printPrediction(
            ord(filepath[6:7]) - ord('0'), np.argmax(prediction))
        if util.label(ord(filepath[6:7]) - ord('0')) == util.label(np.argmax(prediction)):
            if show_all == False:
                continue
        plt.imshow(img[0], cmap='gray_r')
        plt.show()


def paint(model):
    print('\nCustom images:')

    for filepath in glob.iglob(r'./test_images/*.png'):
        print(ord(filepath[14:15]))
        img = cv.imread(filepath)[:, :, 0]
        img = np.invert([img])
        prediction = model.predict(img)
        util.printPrediction(
            util.invlabel(filepath[14:15]), np.argmax(prediction))
        if util.label(util.invlabel(filepath[14:15])) == util.label(np.argmax(prediction)):
            if show_all == False:
                continue
        plt.imshow(img[0], cmap='gray_r')
        plt.show()


def nist(model):
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
            # img = img[28:100, 28:100]
            img = cv.resize(img, (28, 28), interpolation=cv.INTER_LANCZOS4)
            img = np.invert([img])
            img = tf.keras.utils.normalize(np.array(img))
            prediction = model.predict(img)
            util.printPrediction(util.invlabel(NIST_labels[i][symbol]),
                                 np.argmax(prediction))
            if util.label(util.invlabel(NIST_labels[i][symbol])) == util.label(np.argmax(prediction)):
                if show_all == False:
                    continue
            plt.imshow(img[0], cmap='gray_r')
            plt.show()
        i = i + 1
