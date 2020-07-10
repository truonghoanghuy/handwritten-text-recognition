import os

from utils.printer import ProgressBarPrinter

if __name__ == '__main__':
    line_dir = os.path.join('data', 'InkData_line_processed')
    para_dir = os.path.join('data', 'InkData_paragraph_processed')
    out_dir = os.path.join('data', 'line_gt_concatenated')

    line_filename_list = []
    para_filename_list = []
    for root, folders, files in os.walk(line_dir):
        for filename in files:
            if filename.endswith('.txt'):
                full = os.path.join(root, filename)
                line_filename_list.append(full)

    for root, folders, files in os.walk(para_dir):
        for filename in files:
            if filename.endswith('.txt'):
                full = os.path.join(root, filename)
                para_filename_list.append(full)

    progress = ProgressBarPrinter(len(para_filename_list))
    progress.start()
    for filename in para_filename_list:
        base = os.path.basename(filename)
        base_para_prefix = base[:base.rfind('.')]
        out_dir_full = os.path.dirname(filename).replace(para_dir, out_dir)
        os.makedirs(out_dir_full, exist_ok=True)
        refs = [x for x in line_filename_list if base_para_prefix in x]
        if len(refs) == 0:
            continue
        lines = []
        refs.sort(key=lambda x: int(x[x.rfind('_') + 1:x.rfind('.')]))
        for line_filename in refs:
            with open(line_filename, encoding='utf-8') as line_file:
                lines.append(line_file.read())

        out_filename = os.path.join(out_dir_full, base)
        with open(out_filename, 'w', encoding='utf-8') as para_file:
            para = u'\n'.join(lines)
            para_file.write(para)
            r = open(filename, encoding='utf-8').read()
            if u' '.join(lines) != r:
                print('{: <100}'.format(out_filename))
        progress.step()
