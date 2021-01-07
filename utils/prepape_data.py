import argparse
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def make_folds(images_folder, labels_folder, num_fold=5):

    save_folder = '../data/membrane/folds'
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)

    file_paths = []
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            file_paths.append(os.path.join(root, file))
    kf = KFold(num_fold, shuffle=True)

    fold_num = 1
    for train, val in kf.split(file_paths):
        tr_imgs = []
        tr_masks = []

        val_imgs = []
        val_masks = []

        for ts in train:
            img = file_paths[ts]
            tr_imgs.append(img)
            mask = img.replace(images_folder.split('/')[-1], labels_folder.split('/')[-1])
            tr_masks.append(mask)

        for vs in val:
            img = file_paths[vs]
            val_imgs.append(img)
            mask = img.replace(images_folder.split('/')[-1], labels_folder.split('/')[-1])
            val_masks.append(mask)

        np.save(f'{save_folder}/fold_{fold_num}_train_images', tr_imgs)
        np.save(f'{save_folder}/fold_{fold_num}_train_masks', tr_masks)
        np.save(f'{save_folder}/fold_{fold_num}_val_images', val_imgs)
        np.save(f'{save_folder}/fold_{fold_num}_val_masks', val_masks)

        fold_num += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--images_folder', default='/content/unet_pytorch/data/membrane/train/image')
    parser.add_argument('--labels_folder', default='/content/unet_pytorch/data/membrane/train/label')

    args = parser.parse_args()

    make_folds(args.images_folder, args.labels_folder)
