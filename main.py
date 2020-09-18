import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import tensorflow as tf

from effnet import *

np.random.seed = 42

h, w, c = 584, 565, 3


# -------------------------------------------------------------------------------------------------

# training_images = "/home/nsid/Documents/Projects/Data/DRIVE/training/images/"
# training_labels = "/home/nsid/Documents/Projects/Data/DRIVE/training/1st_manual/"
# training_masks = "/home/nsid/Documents/Projects/Data/DRIVE/training/mask/"
# testing_images = "/home/nsid/Documents/Projects/Data/DRIVE/test/images/"
# testing_masks = "/home/nsid/Documents/Projects/Data/DRIVE/test/mask/"

# train_img = next(os.walk(training_images))[2]
# train_lbs = next(os.walk(training_labels))[2]
# train_msk = next(os.walk(training_masks))[2]
# test_img = next(os.walk(testing_images))[2]
# test_msk = next(os.walk(testing_masks))[2]

# X_train = np.zeros((len(train_img), h, w, c), dtype=np.float32)
# Y_train = np.zeros((len(train_lbs), h, w, 1), dtype=np.bool)
# X_test = np.zeros((len(test_img), h, w, c), dtype=np.float32)

# print(train_img)

# for n, file_id in enumerate(train_img):
#     img = io.imread(training_images + file_id)
#     img = np.reshape(img, (h, w, c)) / 255
#     X_train[n] = img

#     mask = io.imread(training_labels + file_id)
#     mask = np.reshape(mask, (h, w, c))
#     Y_train[n] = mask

# for n, file_id in enumerate(test_img):
#     img = io.imread(testing_images + file_id)
#     img = np.reshape(img, (h, w, c)) / 255
#     X_test[n] = img

# -------------------------------------------------------------------------------------------------


box = efficientUnet(input_shape=(h, w, c))
box.summary()
print("Completed")
