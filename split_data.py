import argparse
import os
from typing import Dict, List


def main(args):
    path = os.path.join
    normpath = os.path.normpath

    data_path = path(args.root, 'data')
    train_split_file = path(data_path, 'train_set.txt')
    validation_split_file = path(data_path, 'validation_set.txt')
    test_split_file = path(data_path, 'test_set.txt')
    line_folder = path(data_path, 'InkData_line_processed')
    paragraph_folder = path(data_path, 'InkData_paragraph_processed')
    word_folder = path(data_path, 'InkData_word_processed')

    def strip(x):
        return x[:-len('.inkml\n')]

    partition: Dict[str, List[str]] = {}
    with open(train_split_file, 'r') as train:
        partition['train'] = [strip(x) for x in train.readlines()]
    with open(validation_split_file, 'r') as validation:
        partition['validation'] = [strip(x) for x in validation.readlines()]
    with open(test_split_file, 'r') as test:
        partition['test'] = [strip(x) for x in test.readlines()]

    def move_into(directory: str, file: str, destination: str):
        target_path = path(directory, destination)
        if not os.path.exists(target_path):
            try:
                os.makedirs(target_path)
            except OSError:
                pass
        indices: List[str] = partition[destination]
        for idx in indices:
            if file.startswith(idx):
                old_path = path(directory, file)
                new_path = path(directory, destination, file)
                os.replace(old_path, new_path)

    for (dir_path, _, filenames) in os.walk(data_path):
        if normpath(dir_path) in (normpath(paragraph_folder),
                                  normpath(line_folder),
                                  normpath(word_folder)):
            print('Processing the directory {} ... '.format(normpath(dir_path)), end='', flush=True)
            for filename in filenames:
                move_into(dir_path, filename, destination='train')
                move_into(dir_path, filename, destination='validation')
                move_into(dir_path, filename, destination='test')
            print('OK.', flush=True)
    print('Done splitting.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split the data into train, validation and test sets')
    parser.add_argument('--root', type=str, default='',
                        help='path to the folder that contains the "data" folder that has the structure as described in'
                             ' the project\'s README')
    try:
        main(parser.parse_args())
    except OSError as e:
        print(e)
        parser.print_help()
