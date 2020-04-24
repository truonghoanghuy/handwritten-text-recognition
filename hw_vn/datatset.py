from os import listdir, mkdir
from os.path import isfile, join, exists
from sklearn import model_selection
from shutil import copy2
from pathlib import Path
import sys
import json


def split_datatset_hw():
    original_dataset_path = 'data/InkData_line_processed'
    files_name_lst = []

    for f in listdir(original_dataset_path):
        if not isfile(join(original_dataset_path, f)):
            continue

        f_name, f_extension = f.split('.')
        if f_extension == 'png':
            files_name_lst.append(f_name)

    training_dataset, test_dataset = model_selection.train_test_split(files_name_lst, train_size=0.8, test_size=0.2,
                                                                      shuffle=True)

    path = Path(original_dataset_path).parent
    dst_path = [join(path, 'train_hw_vn'), join(path, 'test_hw_vn')]
    for path in dst_path:
        if not exists(path):
            mkdir(path)
    for idx, ds in enumerate([training_dataset, test_dataset]):
        for f_name in ds:
            png_f = f_name + ".png"
            txt_f = f_name + ".txt"
            if not isfile(join(original_dataset_path, txt_f)):
                print("File not found", txt_f)

            copy2(join(original_dataset_path, png_f), join(dst_path[idx], png_f))
            copy2(join(original_dataset_path, txt_f), join(dst_path[idx], txt_f))


def get_json_dataset():
    train_path = 'data/train_hw_vn'
    test_path = 'data/test_hw_vn'
    lst = [train_path, test_path]
    hw_vn_training_set = []
    hw_vn_test_set = []
    set_lst = [hw_vn_training_set, hw_vn_test_set]
    parent_path = Path(train_path).parent

    for idx, path in enumerate(lst):
        for f in listdir(path):
            if not isfile(join(path, f)):
                continue
            f_name, f_extension = f.split('.')
            if f_extension == 'png':
                if not isfile(join(path, f_name + '.txt')):
                    print('File not found {}. Can not find corresponding text file, check your splitting dataset'.format(f_name + '.txt'))
                    continue
                set_lst[idx].append([join(path, f_name + '.png'), join(path, f_name + '.txt')])

    with open(join(parent_path, 'hw_vn_training_set.json'), 'w') as f:
        json.dump(set_lst[0], f)
    with open(join(parent_path, 'hw_vn_test_set.json'), 'w') as f:
        json.dump(set_lst[1], f)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'split_dataset':
        split_datatset_hw()
    elif len(sys.argv) == 2 and sys.argv[1] == 'json_dataset':
        get_json_dataset()
    else:
        print('Your parameters is incorrect. Please try again!')