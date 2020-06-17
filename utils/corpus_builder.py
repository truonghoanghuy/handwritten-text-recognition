import sys
import os


if __name__ == '__main__':
    txt_path = sys.argv[1]
    input_directory_path = []
    for i in range(2, len(sys.argv)):
        input_directory_path.append(sys.argv[i])

    output = open(txt_path, 'a+', encoding='utf8')

    cnt = 0
    for directory in input_directory_path:
        print(f'Processing directory {directory} ...')
        for root, _, files in os.walk(directory):
            for file in files:
                if file.split('.')[1] != 'txt':
                    continue
                cnt += 1
                print(file)
                data = open(os.path.join(root, file), 'r', encoding='utf8')
                output.write('\n')
                output.write(data.read())
        print('Done.')
        print('-------------------------------')

    print(f'Total {cnt} file(s) processed')
    output.close()