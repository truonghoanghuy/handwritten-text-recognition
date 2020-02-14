import os
from typing import Dict, List, Union

import torch.utils.data
from PIL import Image


def read_img_gt(data_path: Union[str, os.PathLike]):
    """
    Read the ground-truth for img folders that contain image in ``.png`` format and the corresponding ground-truth is in
    ``.txt`` files

    :param data_path: path to the img folder
    :return: the python dictionary which keys are the image filename and values are their label.
    """

    path = os.path.join
    normpath = os.path.normpath

    partition: Dict[str, List[str]] = {'train': [], 'validation': []}
    gt_dict: Dict[str, str] = {}
    train_files: List[str] = []
    validation_files: List[str] = []
    train_path: str = ''
    validation_path: str = ''

    # get file list
    for (dir_path, _, filenames) in os.walk(data_path):
        if normpath(dir_path) == normpath(path(data_path, 'train')):
            train_path, train_files = dir_path, filenames
        if normpath(dir_path) == normpath(path(data_path, 'validation')):
            validation_path, validation_files = dir_path, filenames

    # each TXT contains a label and the corresponding PNG is the index
    for filename in train_files:
        if filename.endswith('.txt'):
            index = filename[:-len('.txt')] + '.png'
            with open(path(train_path, filename), 'r', encoding='utf-8') as label_file:
                partition['train'].append(index)
                gt_dict[index] = label_file.readline()
    for filename in validation_files:
        if filename.endswith('.txt'):
            index = filename[:-len('.txt')] + '.png'
            with open(path(validation_path, filename), 'r', encoding='utf-8') as label_file:
                partition['validation'].append(index)
                gt_dict[index] = label_file.readline()
    return partition, gt_dict


class EndToEndDataset(torch.utils.data.Dataset):
    """
    Customized for reading VNOnDB dataset after processed by vndee/vnondb-extractor
    """

    def __init__(self, data_path: Union[str, os.PathLike]):
        super().__init__()
        self.data_path = data_path
        self.partition, self.image_gt = read_img_gt(data_path)
        self.filenames = self.partition['train'] + self.partition['validation']

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.data_path, image_name)
        image = Image.open(image_path)
        label = self.image_gt[image_name]
        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.image_gt)
