import os
import sys
import cv2
from shutil import copy2

from utils import augmentation

input_dir = sys.argv[1]
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

augmenter = augmentation.HwAugmenter()

for root, _, files in os.walk(input_dir):
    for file in files:
        file_name, file_extension = file.split('.')
        if file_extension == 'txt':
            continue
        img = cv2.imread(os.path.join(root, file))
        img = augmenter(img)
        cv2.imwrite(os.path.join(output_dir, file), img)
        copy2(os.path.join(root, file_name + '.txt'), output_dir)
