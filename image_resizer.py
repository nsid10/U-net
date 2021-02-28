import os
import numpy as np

from PIL import Image


target = "labels"
main_path = f"/home/nsid/Documents/Projects/Data/ISIC 2018/{target}/"

pics = next(os.walk(main_path))[2]
pics.sort()

dim = 256

train_box = []
test_box = []

for i in range(len(pics)):
    if i % 5 == 0:
        im = Image.open(main_path + pics[i])
        im2 = im.resize((dim, dim))
        im2 = np.array(im2)
        im2 = im2.reshape((1, dim, dim))
        test_box.append(im2)
    else:
        im = Image.open(main_path + pics[i])
        im2 = im.resize((dim, dim))
        im2 = np.array(im2)
        im2 = im2.reshape((1, dim, dim))
        train_box.append(im2)

    # if i % 100 == 0:
    #     print(i)

super_train_box = np.concatenate(train_box, axis=0)
super_test_box = np.concatenate(test_box, axis=0)

np.savez_compressed(f"/home/nsid/Documents/Projects/Data/ISIC 2018/train_{target}.npz", super_train_box)
np.savez_compressed(f"/home/nsid/Documents/Projects/Data/ISIC 2018/test_{target}.npz", super_test_box)
