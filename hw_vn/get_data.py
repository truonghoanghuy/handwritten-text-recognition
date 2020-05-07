from os import listdir, walk
from os.path import isfile, join, basename, normpath
import sys
import json


def process_cinnamon_dataset(output, root_path):
    with open(join(root_path, 'labels.json'), 'r', encoding='utf8') as f:
        labels = json.load(f)

    for root, _, files in walk(root_path):
        for file in files:
            f_name, f_extension = file.split('.')
            if f_extension.lower() in ['png', 'jpg', 'jpeg']:
                if file not in labels:
                    print(f'Can not find label of image \"{join(root_path, file)}\"')
                    continue

                label = labels[file]
                txt_file_path = join(root, f_name + '.txt')
                txt_file = open(txt_file_path, 'w', encoding='utf8')
                txt_file.write(label)
                txt_file.close()

                output.append([join(root, file), txt_file_path])


def get_json_dataset(output_file_path, input_directories_list):
    output = []

    for directory_path in input_directories_list:
        print(f'Processing \"{directory_path}\" ... ', end='')
        if 'cinnamon' in basename(normpath(directory_path)):
            process_cinnamon_dataset(output, directory_path)
        else:
            for root, _, files in walk(directory_path):
                for file in files:
                    f_name, f_extension = file.split('.')
                    if f_extension.lower() in ['png', 'jpg', 'jpeg']:
                        if not isfile(join(root, f_name + '.txt')):
                            print(
                                'File not found {}. Can not find corresponding text file, check your splitting '
                                'dataset'.format(f_name + '.txt'))
                            continue
                        output.append([join(root, file), join(root, f_name + '.txt')])
        print('Done.')

    with open(output_file_path, 'w') as f:
        json.dump(output, f)
    print(f'Finished saving all data to \"{output_file_path}\".')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Your parameters is incorrect. Please try again!')
    else:
        output_file = sys.argv[1]
        input_directories_lst = []
        for i in range(2, len(sys.argv)):
            input_directories_lst.append(sys.argv[i])

        get_json_dataset(output_file, input_directories_lst)