import os
import numpy as np

from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d


main_path = ""
images = f"{main_path}"
labels = f"{main_path}"


def make_patches(images_path, labels_path, patch_dim, max_patches=100):
    def patcherize(im, patch_dim, max_patches):
        pics = [im, im.transpose(Image.FLIP_LEFT_RIGHT)]

        for an in (90, 180, 270):
            imr = im.rotate(an, expand=1)
            pics.append(imr)
            imt = imr.transpose(Image.FLIP_LEFT_RIGHT)
            pics.append(imt)

        patches_per = max_patches // 8
        patches = []
        for p in pics:
            arp = np.array(p)
            patches.append(extract_patches_2d(arp, patch_size=(patch_dim, patch_dim), max_patches=patches_per, random_state=42))

        all_patches = np.concatenate(patches, axis=0)
        return all_patches

    image_list = next(os.walk(images_path))[2]
    label_list = next(os.walk(labels_path))[2]

    image_list.sort()
    label_list.sort()

    for i, l in zip(image_list, label_list):
        img = Image.open(images_path + i)
        lbl = Image.open(labels_path + l)

        img_box = patcherize(img, patch_dim, max_patches)
        lbl_box = patcherize(lbl, patch_dim, max_patches)

        np.savez_compressed(f"{images_path}/image_patches_{i+1:02}.npz", img_box)
        np.savez_compressed(f"{labels_path}/label_patches_{i+1:02}.npz", lbl_box)


def image_resize(images_path, labels_path, target_dim):
    target = "images"
    for path in (images_path, labels_path):
        pics = next(os.walk(path))[2]
        pics.sort()

        box = []

        for i in range(len(pics)):
            im = Image.open(path + pics[i])
            im2 = im.resize((target_dim, target_dim))
            im2 = np.array(im2)
            im2 = im2.reshape((1, target_dim, target_dim))
            box.append(im2)

        super_box = np.concatenate(box, axis=0)

        np.savez_compressed(f"{path}/resized_{target}.npz", super_box)
        target = "labels"
