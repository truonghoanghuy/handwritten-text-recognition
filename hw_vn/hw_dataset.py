import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import string_utils, augmentation

PADDING_CONSTANT = 0


def collate(batch):
    batch = [b for b in batch if b is not None]
    # These all should be the same size or error
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i, :, :b_img.shape[1], :] = b_img

        L = batch[i]['gt_label']
        all_labels.append(L)
        label_lengths.append(len(L))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0, 3, 1, 2])
    line_imgs = torch.from_numpy(line_imgs)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch]
    }


class HwDataset(Dataset):
    def __init__(self, set_list, char_to_idx, augment=False, img_height=32, random_subset_size=None):

        self.img_height = img_height

        self.ids = set_list
        self.ids.sort()

        if random_subset_size is not None:
            self.ids = random.sample(self.ids, min(random_subset_size, len(self.ids)))

        self.char_to_idx = char_to_idx
        self.augmentation = augment
        self.warning = False

        if self.augmentation:
            self.augmenter = augmentation.HwAugmenter()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path, gt_path = self.ids[idx]
        gt = open(gt_path, encoding='utf8')
        if gt is None:
            return None

        img = cv2.imread(img_path)

        if img is None:
            return None

        if img.shape[0] != self.img_height:
            if img.shape[0] < self.img_height and not self.warning:
                self.warning = True
                print("WARNING: upsampling image to fit size")
            percent = float(self.img_height) / img.shape[0]
            img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)

        if img is None:
            return None

        if self.augmentation:
            img = self.augmenter(img)

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        gt = gt.read()
        if len(gt) == 0:
            return None
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)

        return {
            "line_img": img,
            "gt": gt,
            "gt_label": gt_label
        }
