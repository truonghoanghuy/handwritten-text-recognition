import argparse
import json
import os
import time
from typing import List

import cv2
import numpy as np
import torch
import yaml

from e2e import e2e_postprocessing, visualization
from e2e.e2e_model import E2EModel
from hw import cnn_lstm
from hw_vn.continuous_state import init_model
from utils import error_rates, string_utils, safe_load
from utils.paragraph_processing import softmax, combine_lines_into_paragraph
from utils.printer import ProgressBarPrinter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='e2e_config.yaml', type=str, help='The YAML configuration file.')
    parser.add_argument('--input', default='data/InkData_paragraph_processed/test', type=str,
                        help='Path to the input directory.')
    parser.add_argument('--output', default='data/output', type=str, help='Path to the output directory.')
    parser.add_argument('--model', default='cnn_attention_lstm', help='HWR model used')
    parser.add_argument('--best_path', action='store_true',
                        help='Use best path decoding. '
                             'Default is using beam search decoding with language model.')
    parser.add_argument('--combine', action='store_true',
                        help='Combine all lines in one paragraph before decoding. '
                             'Default is decoding each line one by one.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU. Default is trying to use GPU if possible.')
    parser.add_argument('--scoring', action='store_true', help='Scoring prediction by CER and WER')
    parser.add_argument('--verbose', action='store_true', help='Export line images and show padded image.')
    parser.add_argument('--no_output', action='store_true', help='No output files.')

    args = parser.parse_args()
    config_path = args.config
    input_directory = args.input
    output_directory = args.output
    use_best_path = args.best_path
    combine = args.combine
    use_cpu = args.cpu
    scoring = args.scoring
    verbose = args.verbose
    hw_model = args.model
    no_output = args.no_output
    mode = 'hw_vn'

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    image_paths: List[str] = []
    for root, folder, files in os.walk(input_directory):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, f))

    os.makedirs(output_directory, exist_ok=True)

    if not use_best_path:
        from hw_vn.beam_search_with_lm import beam_search_with_lm

    char_set_path = config['network']['hw']['char_set_path']
    with open(char_set_path, encoding='utf8') as f:
        char_set = json.load(f)
    idx_to_char = {int(k): v for k, v in char_set['idx_to_char'].items()}
    char_to_idx = char_set['char_to_idx']

    sol, lf, hw = init_model(config, use_cpu=use_cpu, hw_model=hw_model)
    hw_german = cnn_lstm.create_model(config['network']['hw_german'])
    hw_german_state = safe_load.torch_state(os.path.join('model', 'best_overall', 'hw_german.pt'))
    hw_german.load_state_dict(hw_german_state)
    e2e = E2EModel(sol, lf, hw_german, hw)
    if use_cpu:
        e2e.to_cpu()
    e2e.eval()

    cer = []
    wer = []

    progress_bar = ProgressBarPrinter(len(image_paths))
    progress_bar.start()
    start_time = time.time()
    for image_path in sorted(image_paths):
        org_img = cv2.imread(image_path)
        txt_path = image_path.split('.')[0] + '.txt'
        label = open(txt_path, encoding='utf8').read()
        len_label = torch.tensor([len(label)])
        h, w = org_img.shape[:2]
        processed = org_img

        scale = 512 / float(w + 0)
        pad_top = pad_bottom = 128 + 0
        pad_left = pad_right = 128 + 0
        pad_values = [np.argmax(cv2.calcHist([processed], channels=[x], mask=None, histSize=[256], ranges=[0, 256]))
                      for x in range(3)]  # pad the mode value for each color channel
        padded_img = np.dstack([np.pad(processed[:, :, x], ((pad_top, pad_bottom), (pad_left, pad_right)),
                                       constant_values=pad_values[x]) for x in range(3)])
        target_width = int(padded_img.shape[0] * scale)
        target_height = int(padded_img.shape[1] * scale)

        full_img = padded_img.astype(np.float32)
        full_img = full_img.transpose([2, 1, 0])[None, ...]  # (H, W, C) -> (1, C, W, H)
        full_img = torch.from_numpy(full_img)
        full_img = full_img / 128 - 1

        resized_img = cv2.resize(padded_img, (target_height, target_width), interpolation=cv2.INTER_CUBIC)
        resized_img = resized_img.astype(np.float32)
        resized_img = resized_img.transpose([2, 1, 0])[None, ...]
        resized_img = torch.from_numpy(resized_img)
        resized_img = resized_img / 128 - 1

        e2e_input = {
            'full_img': full_img,
            'resized_img': resized_img,
            'resize_scale': 1.0 / scale,
            'len_label': len_label,
        }
        try:
            with torch.no_grad():
                out = e2e.forward(e2e_input, mode=mode)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                e2e.to_cpu()
                with torch.no_grad():
                    out = e2e.forward(e2e_input, lf_batch_size=100, mode=mode)
                e2e.to_cuda()
            else:
                raise e
        out = e2e_postprocessing.results_to_numpy(out)

        if out is None:
            print(f'No results for image "{image_path}"')
            continue

        if not verbose:
            # take into account the padding
            out['sol'][:, 0] -= pad_left
            out['sol'][:, 1] -= pad_top
            for line in out['lf']:
                line[:, 0, :2] -= pad_left
                line[:, 1, :2] -= pad_top

        name_file = os.path.basename(os.path.normpath(image_path)).split('.')[0]
        name_txt = name_file + '.txt'
        with open(os.path.join(os.path.dirname(os.path.normpath(image_path)), name_txt), encoding='utf8') as f:
            ground_truth = u' '.join(f.readlines())

        # Postprocessing Steps
        out['idx'] = np.arange(out['sol'].shape[0])
        out = e2e_postprocessing.trim_ends(out)
        e2e_postprocessing.filter_on_pick(out, e2e_postprocessing.select_non_empty_string(out))
        out = e2e_postprocessing.postprocess(out, sol_threshold=config['post_processing']['sol_threshold'],
                                             lf_nms_params={
                                                 'overlap_range': config['post_processing']['lf_nms_range'],
                                                 'overlap_threshold': config['post_processing']['lf_nms_threshold']
                                             })
        order = e2e_postprocessing.read_order(out)
        e2e_postprocessing.filter_on_pick(out, order)
        paragraph = out['hw_vn']

        if not use_best_path:
            for v in range(len(paragraph)):
                line = paragraph[v]
                temp = np.copy(line[:, 0])
                for i in range(0, len(line[0]) - 1):
                    line[:, i] = line[:, i + 1]
                line[:, len(line[0]) - 1] = temp
                softmax_line = softmax(line)
                paragraph[v] = softmax_line

        if combine:
            space_idx = char_to_idx[' '] - 1
            len_decoder = len(char_to_idx) + 1
            paragraph = combine_lines_into_paragraph(paragraph, space_idx, len_decoder)

        output_strings = []
        if use_best_path:
            if combine:
                pred, raw_pred = string_utils.naive_decode(paragraph)
                output_strings.append(string_utils.label2str_single(pred, idx_to_char, False))
            else:
                param = {'hw': np.array(paragraph)}
                output_strings, _ = e2e_postprocessing.decode_handwriting(param, idx_to_char)
        else:
            if combine:
                output_strings.append(beam_search_with_lm(paragraph))
            else:
                for line in paragraph:
                    res = beam_search_with_lm(line)
                    output_strings.append(res)

        if not no_output:
            draw_img = visualization.draw_output(out, padded_img if verbose else org_img)
            out_basename = os.path.basename(os.path.normpath(image_path))
            out_fullname = os.path.join(output_directory, out_basename)
            cv2.imwrite(out_fullname, draw_img)
            if verbose:
                out_dir_name = out_fullname.rsplit('.')[0]
                os.makedirs(out_dir_name, exist_ok=True)
                for img_index in out['idx']:
                    line_name = os.path.join(out_dir_name, f'{img_index}.png')
                    cv2.imwrite(line_name, out['line_imgs'][img_index])

        # Save results
        label_string = '_'
        if use_best_path:
            label_string += 'best_path_'
        else:
            label_string += 'beam_search_lm_'
        file_path = os.path.join(output_directory, out_basename.split('.')[0]) + label_string + '.txt'

        with open(file_path, 'w', encoding='utf8') as f:
            f.write(u'\n'.join(output_strings))

        if scoring:
            script = u' '.join(output_strings)
            cer.append(error_rates.cer(ground_truth, script))
            wer.append(error_rates.wer(ground_truth, script))

        del resized_img, full_img, paragraph, out
        torch.cuda.empty_cache()
        progress_bar.step()

    print(f'Total time for running e2e: {time.time() - start_time} second(s)')
    if scoring:
        cer = sum(cer) / len(cer)
        wer = sum(wer) / len(wer)
        print(f'CER: {cer}, WER: {wer}')
