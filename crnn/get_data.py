import os
from os.path import join
from shutil import copy2
import sys
import json


def vnondb():
    print('Processing for VNOBDB dataset')
    root_path = '../data'
    data_path = join(root_path, 'InkData_line_processed')
    target_data_path = 'data'

    if not os.path.exists(target_data_path):
        os.mkdir(target_data_path)

    train_file = join('data', 'train')
    train_file = open(train_file, 'w')
    test_file = join('data', 'test')
    test_file = open(test_file, 'w')

    for root, dirs, files in os.walk(data_path):
        if os.path.basename(os.path.normpath(root)) == 'train':
            print(f'Processing {root} ... ', end='')
            for file in files:
                if file.split('.')[1] != 'txt':
                    train_file.write(file + '\n')
                copy2(join(root, file), target_data_path)
            print('Done.')
        elif os.path.basename(os.path.normpath(root)) in ['test', 'validation']:
            print(f'Processing {root} ... ', end='')
            for file in files:
                if file.split('.')[1] != 'txt':
                    test_file.write(file + '\n')
                copy2(join(root, file), target_data_path)
            print('Done.')
    train_file.close()
    test_file.close()
    print('Done.')


def cinnamon():
    print('Processing for Cinnamon dataset')
    root_path = '../data'
    target_data_path = 'data'
    if not os.path.exists(target_data_path):
        os.mkdir(target_data_path)
    train_file = join(target_data_path, 'train')
    train_file = open(train_file, 'a')
    test_file = join(target_data_path, 'test')
    test_file = open(test_file, 'a')

    def process_directory(source_path, target_file):
        with open(join(source_path, 'labels.json'), encoding='utf8') as f:
            labels = json.load(f)

        for k, v in labels.items():
            name_file = 'cinnamon_' + k.split('.')[0]

            txt_file = open(join(target_data_path, name_file + '.txt'), 'w', encoding='utf8')
            txt_file.write(v)
            txt_file.close()

            name_file += '.' + k.split('.')[1]
            copy2(join(source_path, k), join(target_data_path, name_file))
            target_file.write(name_file + '\n')

    for root, _, files in os.walk(root_path):
        base_name = os.path.basename(os.path.normpath(root))
        if base_name.startswith('cinnamon'):
            print(f'Processing {root} ... ', end='')
            process_directory(root, test_file if 'test' in base_name else train_file)
            print('Done.')

    train_file.close()
    test_file.close()
    print('Done.')


if __name__ == '__main__':
    if 'vnondb' in sys.argv:
        vnondb()
    if 'cinnamon' in sys.argv:
        cinnamon()