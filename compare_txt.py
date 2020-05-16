import os
import sys

from utils import error_rates

if __name__ == '__main__':
    dir1, dir2 = sys.argv[1:3]
    filename_list1 = []
    filename_list2 = []
    for root, folders, files in os.walk(dir1):
        for filename in files:
            if filename.endswith('.txt'):
                full = os.path.join(root, filename)
                filename_list1.append(full)

    for root, folders, files in os.walk(dir2):
        for filename in files:
            if filename.endswith('.txt'):
                full = os.path.join(root, filename)
                filename_list2.append(full)

    cer = []
    wer = []
    for filename in filename_list1:
        base = os.path.basename(filename)
        base_prefix = base[:base.find('.')]
        ref = [x for x in filename_list2 if base_prefix in x]
        if len(ref) == 0: continue
        ref = ref[0]
        with open(filename, encoding='utf-8') as f1:
            with open(ref, encoding='utf-8') as f2:
                str1 = u' '.join(f1.readlines())
                str2 = u' '.join(f2.readlines())
                cer.append(error_rates.cer(str1, str2))
                wer.append(error_rates.wer(str1, str2))
    cer = sum(cer) / len(cer)
    wer = sum(wer) / len(wer)
    print('CER:', cer)
    print('WER:', wer)
