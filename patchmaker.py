import os
import numpy as np

from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d

# -------------------------------------------------------------------------------------------------
dim = 96

main_path = "/home/nsid/Documents/Projects/Data/DRIVE"
images = f"{main_path}/training/images/"
labels = f"{main_path}/training/1st_manual/"
# -------------------------------------------------------------------------------------------------

img = next(os.walk(images))[2]
lbl = next(os.walk(labels))[2]

img.sort()
lbl.sort()


def patcherize_simple(img):
    pic = np.array(img)
    patches = extract_patches_2d(pic, patch_size=(dim, dim), max_patches=100, random_state=42)

    return patches


def patcherize(im):
    pics = [im, im.transpose(Image.FLIP_LEFT_RIGHT)]

    for an in (90, 180, 270):
        imr = im.rotate(an, expand=1)
        pics.append(imr)
        imt = imr.transpose(Image.FLIP_LEFT_RIGHT)
        pics.append(imt)

    patches = []
    arp = np.array(pics[0])
    patches.append(extract_patches_2d(arp, patch_size=(dim, dim), max_patches=75, random_state=42))

    for p in range(1, 8):
        arp = np.array(pics[p])
        patches.append(extract_patches_2d(arp, patch_size=(dim, dim), max_patches=25, random_state=42))

    all_patches = np.concatenate(patches, axis=0)
    return all_patches


for i in range(len(img)):
    img = Image.open(images + img[i])
    lbl = Image.open(labels + lbl[i])

    img_box = patcherize(img)
    lbl_box = patcherize(lbl)

    np.savez_compressed(f"{main_path}/patches_{dim}/images/img_patches_{i+1:02}.npz", img_box)
    np.savez_compressed(f"{main_path}/patches_{dim}/labels/lbl_patches_{i+1:02}.npz", lbl_box)
else:
    print(img_box.shape)
    print(lbl_box.shape)
