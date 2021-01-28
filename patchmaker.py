import os
import numpy as np

from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d

main_path = "/home/nsid/Documents/Projects/Data/DRIVE"

training_images = f"{main_path}/training/images/"
training_labels = f"{main_path}/training/1st_manual/"
training_masks = f"{main_path}/training/mask/"
# testing_images = f"{main_path}/test/images/"
# testing_masks = f"{main_path}/test/mask/"

train_img = next(os.walk(training_images))[2]
train_lbl = next(os.walk(training_labels))[2]
train_msk = next(os.walk(training_masks))[2]
# test_img = next(os.walk(testing_images))[2]
# test_msk = next(os.walk(testing_masks))[2]

train_img.sort()
train_lbl.sort()
train_msk.sort()
# test_img.sort()
# test_msk.sort()


def patcherize(img):
    pic = np.array(img)
    patches = extract_patches_2d(pic, patch_size=(64, 64), max_patches=5950, random_state=42)

    return patches


n = 0

for i in range(16):
    img = Image.open(training_images + train_img[i])
    lbl = Image.open(training_labels + train_lbl[i])
    msk = Image.open(training_masks + train_msk[i])

    img_box = patcherize(img)
    lbl_box = patcherize(lbl)
    msk_box = patcherize(msk)

    for img_patch, lbl_patch, msk_patch in zip(img_box, lbl_box, msk_box):
        pic = Image.fromarray(img_patch)
        gt = Image.fromarray(lbl_patch)

        # if np.mean(msk_patch) > 127:
        #     n += 1
        #     pic.save(f"{main_path}/patches/training/images/{n:06}.png")
        #     gt.save(f"{main_path}/patches/training/labels/{n:06}.png")
        #     if n % 100 == 0:
        #         print(f"{n} operations done")

        if np.mean(msk_patch) > 127:
            n += 1
            # pic.save(f"{main_path}/patches/test/images/{n:06}.png")
            # gt.save(f"{main_path}/patches/test/labels/{n:06}.png")
            # if n % 100 == 0:
            #     print(f"{n} operations done")


print(n)
