import os
import sys

from utils import error_rates

if __name__ == '__main__':
    gt_dir, pr_dir = sys.argv[1:3]
    gt_filename_list = []
    pr_filename_list = []
    for root, folders, files in os.walk(gt_dir):
        for filename in files:
            if filename.endswith('.txt'):
                full = os.path.join(root, filename)
                gt_filename_list.append(full)

    for root, folders, files in os.walk(pr_dir):
        for filename in files:
            if filename.endswith('.txt'):
                full = os.path.join(root, filename)
                pr_filename_list.append(full)

    cer = []
    wer = []
    for filename in gt_filename_list:
        base = os.path.basename(filename)
        base_prefix = base[:base.find('.')]
        ref = [x for x in pr_filename_list if base_prefix in x]
        if len(ref) == 0: continue
        ref = ref[0]
        with open(filename, encoding='utf-8') as gt_file:
            with open(ref, encoding='utf-8') as pr_file:
                gt_str = u' '.join(gt_file.readlines())
                pr_str = u' '.join(pr_file.readlines())
                cer.append(error_rates.cer(gt_str, pr_str))
                wer.append(error_rates.wer(gt_str, pr_str))
    cer = sum(cer) / len(cer)
    wer = sum(wer) / len(wer)
    print('CER:', cer)
    print('WER:', wer)
