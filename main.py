import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import tensorflow as tf

from effnet import *


hgt, wdt, chn = 512, 512, 1


# -------------------------------------------------------------------------------------------------

training_images = "/home/nsid/Documents/Projects/Data/cells_isbi2012/train/images/"
training_labels = "/home/nsid/Documents/Projects/Data/cells_isbi2012/train/labels/"
testing_images = "/home/nsid/Documents/Projects/Data/cells_isbi2012/test/images/"

train_img = next(os.walk(training_images))[2]
train_lbs = next(os.walk(training_labels))[2]
test_img = next(os.walk(testing_images))[2]

X_train = np.zeros((len(train_img), hgt, wdt, chn), dtype=np.float32)
Y_train = np.zeros((len(train_lbs), hgt, wdt, 1), dtype=np.bool)
X_test = np.zeros((len(test_img), hgt, wdt, chn), dtype=np.float32)

for n, file_id in enumerate(train_img):
    img = io.imread(training_images + file_id)
    img = np.reshape(img, (hgt, wdt, chn)) / 255
    X_train[n] = img

    mask = io.imread(training_labels + file_id)
    mask = np.reshape(mask, (hgt, wdt, chn))
    Y_train[n] = mask

for n, file_id in enumerate(test_img):
    img = io.imread(testing_images + file_id)
    img = np.reshape(img, (hgt, wdt, chn)) / 255
    X_test[n] = img

# -------------------------------------------------------------------------------------------------


box = efficientUnet()
box.summary()
print("Completed")
