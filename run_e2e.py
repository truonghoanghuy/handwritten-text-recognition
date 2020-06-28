import yaml
import os
import json
import numpy as np
import time
import argparse
import torch
import cv2

from e2e import e2e_postprocessing, visualization
from e2e.e2e_model import E2EModel
from utils import error_rates, string_utils
from utils.paragraph_processing import softmax, combine_lines_into_paragraph
from utils.printer import ProgressBarPrinter
from hw_vn.continuous_state import init_model
from line_extractor import get_lines


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='e2e_config.yaml', type=str, help='The YAML configuration file.')
parser.add_argument('--input', default='data/InkData_paragraph_processed/test', type=str,
                    help='Path to the input directory.')
parser.add_argument('--output', default='data/output', type=str, help='Path to the output directory.')
parser.add_argument('--model', default='cnn_attention_lstm', help='HWR model used')
parser.add_argument('--best_path', action='store_true',
                    help='Use best path decoding, default is using beam search decoding with language model.')
parser.add_argument('--combine', action='store_true',
                    help='Combine all lines in one paragraph before decoding. Default is decoding each line on by one.')
parser.add_argument('--cpu', action='store_true',
                    help='using CPU instead of GPU. Default is trying to use GPU, if can not go to CPU.')
parser.add_argument('--scoring', action='store_true', help='scoring prediction by CER and WER')
parser.add_argument('--no_output', action='store_true', help='no output of visualization images and predicted text')

args = parser.parse_args()
config_path = args.config
image_path_directory = args.input
output_directory = args.output
use_best_path = args.best_path
combine_lines = args.combine
use_cpu = args.cpu
scoring = args.scoring
no_output = args.no_output
hw_model = args.model
mode = 'hw_vn'

if not use_best_path:
    from hw_vn.beam_search_with_lm import beam_search_with_lm

image_paths = []
for root, folder, files in os.walk(image_path_directory):
    for f in files:
        if f.lower().endswith('.jpg') or f.lower().endswith('.png') or f.lower().endswith('.jpeg'):
            image_paths.append(os.path.join(root, f))

with open(config_path) as f:
    config = yaml.load(f)

if not no_output and not os.path.exists(output_directory):
    os.mkdir(output_directory)

char_set_path = config['network']['hw']['char_set_path']
with open(char_set_path, encoding='utf8') as f:
    char_set = json.load(f)
idx_to_char = {int(k): v for k, v in char_set['idx_to_char'].items()}
char_to_idx = char_set['char_to_idx']

model_mode = 'best_overall'
sol, lf, hw = init_model(config, sol_dir=model_mode, lf_dir=model_mode, hw_dir=model_mode,
                         use_cpu=use_cpu, hw_model=hw_model)

e2e = E2EModel(sol, lf, hw, use_cpu=use_cpu)
e2e.eval()

alpha = 2
beta = 1

if scoring:
    cer = []
    wer = []

progress_bar = ProgressBarPrinter(len(image_paths))
progress_bar.start()
start_time = time.time()
for image_path in sorted(image_paths):
    name_file = os.path.basename(os.path.normpath(image_path)).split('.')[0]
    name_txt = name_file + '.txt'
    with open(os.path.join(os.path.dirname(os.path.normpath(image_path)), name_txt), encoding='utf8') as f:
        ground_truth = u' '.join(f.readlines())

    org_img = cv2.imread(image_path)
    out = get_lines(image_path, e2e, config, mode='hw_vn')

    paragraph = out['hw']
    if not use_best_path:
        for v in range(len(paragraph)):
            line = paragraph[v]
            temp = np.copy(line[:, 0])
            for i in range(0, len(line[0]) - 1):
                line[:, i] = line[:, i + 1]
            line[:, len(line[0]) - 1] = temp
            softmax_line = softmax(line)
            paragraph[v] = softmax_line

    if combine_lines:
        space_idx = char_to_idx[' '] - 1
        len_decoder = len(char_to_idx) + 1
        paragraph = combine_lines_into_paragraph(paragraph, space_idx, len_decoder)

    output_strings = []
    if use_best_path:
        if combine_lines:
            pred, raw_pred = string_utils.naive_decode(paragraph)
            output_strings.append(string_utils.label2str_single(pred, idx_to_char, False))
        else:
            param = {'hw': np.array(paragraph)}
            output_strings, _ = e2e_postprocessing.decode_handwriting(param, idx_to_char)
    else:
        if combine_lines:
            output_strings.append(beam_search_with_lm(paragraph))
        else:
            for line in paragraph:
                res = beam_search_with_lm(line)
                output_strings.append(res)

    if not no_output:
        draw_img = visualization.draw_output(out, org_img)
        out_image_name = os.path.basename(os.path.normpath(image_path))
        cv2.imwrite(os.path.join(output_directory, out_image_name), draw_img)

        # Save results
        label_string = "_"
        if use_best_path:
            label_string += "bestpath_"
        else:
            label_string += 'beam_search_lm_'
        file_path = os.path.join(output_directory, out_image_name.split('.')[0]) + label_string + ".txt"

        with open(file_path, 'w', encoding='utf8') as f:
            f.write(u'\n'.join(output_strings))

    if scoring:
        script = u' '.join(output_strings)
        cer.append(error_rates.cer(ground_truth, script))
        wer.append(error_rates.wer(ground_truth, script))

    del paragraph, out
    torch.cuda.empty_cache()
    progress_bar.step()

print(f'Total time for running e2e: {time.time() - start_time} second(s)')
if scoring:
    cer = sum(cer) / len(cer)
    wer = sum(wer) / len(wer)
    print(f'CER: {cer}, WER: {wer}')
