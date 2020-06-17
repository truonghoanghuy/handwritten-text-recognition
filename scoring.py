import sys
import yaml
import os
import json
import numpy as np
import time
import argparse

from e2e import e2e_postprocessing
from utils import error_rates, string_utils, printer
from utils.paragraph_processing import softmax, combine_lines_into_paragraph
from hw_vn.beam_search_with_lm import beam_search_with_lm


parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('npz_folder')
parser.add_argument('--best_path', action='store_true')
parser.add_argument('--combine', action='store_true')
args = parser.parse_args()

config_path = args.config_path
npz_folder = args.npz_folder
use_best_path = args.best_path
combine_lines = args.combine

with open(config_path) as f:
    config = yaml.load(f)

npz_paths = []
for root, folder, files in os.walk(npz_folder):
    for f in files:
        if f.lower().endswith(".npz"):
            npz_paths.append(os.path.join(root, f))
print('Finished collecting all npz files.')

char_set_path = config['network']['hw']['char_set_path']
with open(char_set_path, encoding='utf8') as f:
    char_set = json.load(f)

idx_to_char = {}
for k, v in iter(char_set['idx_to_char'].items()):
    idx_to_char[int(k)] = v

char_to_idx = char_set['char_to_idx']

paragraphs_hw = []
ground_truth_hw = []
for npz_path in sorted(npz_paths):
    out = np.load(npz_path)
    out = dict(out)

    path = str(out['image_path']).replace('\\', '/')
    name_file = os.path.basename(os.path.normpath(path)).split('.')[0]
    name_txt = name_file + '.txt'
    with open(os.path.join(os.path.dirname(os.path.normpath(path)), name_txt)) as f:
        ground_truth_hw.append(u' '.join(f.readlines()))

    # Postprocessing Steps
    out['idx'] = np.arange(out['sol'].shape[0])
    out = e2e_postprocessing.trim_ends(out)

    e2e_postprocessing.filter_on_pick(out, e2e_postprocessing.select_non_empty_string(out))
    out = e2e_postprocessing.postprocess(out,
                                         sol_threshold=config['post_processing']['sol_threshold'],
                                         lf_nms_params={
                                             "overlap_range": config['post_processing']['lf_nms_range'],
                                             "overlap_threshold": config['post_processing']['lf_nms_threshold']
                                         }
                                         )
    order = e2e_postprocessing.read_order(out)
    e2e_postprocessing.filter_on_pick(out, order)
    paragraphs_hw.append(out['hw'])
print('Finished getting all ground truth files and filtering lines in each paragraph')

alpha = 2
beta = 1

for u in range(len(paragraphs_hw)):
    paragraphs = paragraphs_hw[u]
    for v in range(len(paragraphs)):
        line = paragraphs[v]
        temp = np.copy(line[:, 0])
        for i in range(0, len(line[0]) - 1):
            line[:, i] = line[:, i + 1]
        line[:, len(line[0]) - 1] = temp
        softmax_line = softmax(line)
        paragraphs[v] = softmax_line

    if combine_lines:
        space_idx = char_to_idx[' '] - 1
        len_decoder = len(char_to_idx) + 1
        paragraphs = combine_lines_into_paragraph(paragraphs, space_idx, len_decoder)

    paragraphs_hw[u] = paragraphs
print('Finished making soft max and maybe concatenate for all lines in all image files')

print('Start scoring')
cer = []
wer = []
process_bar = printer.ProgressBarPrinter(len(paragraphs_hw))
process_bar.start()
start_time = time.time()
for i in range(len(paragraphs_hw)):
    paragraph = paragraphs_hw[i]
    script = ''

    if use_best_path:
        if combine_lines:
            pred, raw_pred = string_utils.naive_decode(paragraph)
            script = string_utils.label2str_single(pred, idx_to_char, False)
        else:
            output_strings, _ = e2e_postprocessing.decode_handwriting({'hw', np.array(paragraph)}, idx_to_char)
            script = u' '.join(output_strings)
    else:
        if combine_lines:
            script = beam_search_with_lm(paragraph)
        else:
            output_string = []
            for line in paragraph:
                res = beam_search_with_lm(line)
                output_string.append(res)
            script = u' '.join(output_string)

    cer.append(error_rates.cer(script, ground_truth_hw[i]))
    wer.append(error_rates.wer(script, ground_truth_hw[i]))
    process_bar.step()
cer = sum(cer) / len(cer)
wer = sum(wer) / len(wer)

print(f'Total time for scoring: {time.time() - start_time} second(s)')
print(f'CER: {cer}, WER: {wer}')
