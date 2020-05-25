import argparse
import codecs
import json
import os

import cv2
import numpy as np
import torch
import yaml

from e2e import e2e_postprocessing, visualization
from e2e.e2e_model import E2EModel
from utils import continuous_state


def adaptive_binary_gaussian(img, constant=10):
    block_size = img.shape[1] // 100  # this has been chosen based on the text line width
    if block_size % 2 == 0:
        block_size += 3
    result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                   blockSize=block_size, C=constant)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def white_balance(img):
    # https://stackoverflow.com/a/46391574
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='vn_config.yaml', type=str, help='The YAML configuration file.')
    parser.add_argument('--input', default='data/input', type=str, help='Path to the input directory.')
    parser.add_argument('--output', default='data/output', type=str, help='Path to the output directory.')
    parser.add_argument('--verbose', action='store_true', help='Export line images and show padded image.')
    args = parser.parse_args()
    input_path = args.input
    out_path = args.output
    verbose = args.verbose
    os.makedirs(out_path, exist_ok=True)
    with open(args.config) as f:
        config = yaml.load(f)

    image_paths = []
    for root, folder, files in os.walk(input_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, f))

    char_set_path = config['network']['hw']['char_set_path']
    with open(char_set_path, encoding='utf-8') as f:
        char_set = json.load(f)
    idx_to_char = {int(k): v for k, v in char_set['idx_to_char'].items()}
    char_to_idx = char_set['char_to_idx']

    sol, lf, hw = continuous_state.init_model(config)
    e2e = E2EModel(sol, lf, hw)
    e2e.eval()

    for image_path in sorted(image_paths):
        org_img = cv2.imread(image_path)
        if isinstance(org_img, np.ndarray):
            print(f'Processing {image_path} ({org_img.shape}) ... ', end='', flush=True)
        else:
            continue
        h, w = org_img.shape[:2]
        processed = org_img
        processed = white_balance(processed)
        processed = adaptive_binary_gaussian(processed)

        target_height = 512
        scale = target_height / float(w + w / 3)
        pad_top = pad_bottom = 128 + w // 6
        pad_left = pad_right = 128 + w // 6
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
            'resize_scale': 1.0 / scale
        }
        try:
            with torch.no_grad():
                out = e2e.forward(e2e_input)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                e2e.to_cpu()
                with torch.no_grad():
                    out = e2e.forward(e2e_input, lf_batch_size=100)
                e2e.to_cuda()
            else:
                raise e
        out = e2e_postprocessing.results_to_numpy(out)

        if out is None:
            print('No Results')
            continue

        if not verbose:
            # take into account the padding
            out['sol'][:, 0] -= pad_left
            out['sol'][:, 1] -= pad_top
            for line in out['lf']:
                line[:, 0, :2] -= pad_left
                line[:, 1, :2] -= pad_top

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

        # Decoding network output
        output_strings, decoded_raw_hw = e2e_postprocessing.decode_handwriting(out, idx_to_char)

        # Export
        draw_img = visualization.draw_output(out, padded_img if verbose else org_img)
        out_basename = os.path.basename(image_path)
        img_out_name = os.path.join(out_path, out_basename)
        cv2.imwrite(f'{img_out_name}.png', draw_img)
        with codecs.open(f'{img_out_name}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_strings))
        if verbose:
            out_dir_name = os.path.join(out_path, out_basename)
            os.makedirs(out_dir_name, exist_ok=True)
            for img_index in out['idx']:
                line_name = os.path.join(out_dir_name, f'{img_index}.png')
                cv2.imwrite(line_name, out['line_imgs'][img_index])
        print('OK')
