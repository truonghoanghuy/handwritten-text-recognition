import yaml
import json
import numpy as np
import torch
import cv2
import os

from e2e import e2e_postprocessing
from e2e.e2e_model import E2EModel
from utils.paragraph_processing import softmax
from utils import safe_load
from hw_vn.continuous_state import init_model
from hw_vn.beam_search_with_lm import beam_search_with_lm
from hw import cnn_lstm

config_path = 'e2e_config.yaml'
with open(config_path) as f:
    config = yaml.load(f)

char_set_path = 'data/char_set_vn.json'
with open(char_set_path, encoding='utf8') as f:
    char_set = json.load(f)
idx_to_char = {int(k): v for k, v in char_set['idx_to_char'].items()}
char_to_idx = char_set['char_to_idx']

sol, lf, hw = init_model(config, use_cpu=False, hw_model='cnn_lstm')
hw_german = cnn_lstm.create_model(config['network']['hw_german'])
hw_german_state = safe_load.torch_state(os.path.join('model', 'best_overall', 'hw_german.pt'))
hw_german.load_state_dict(hw_german_state)
e2e = E2EModel(sol, lf, hw_german, hw)
e2e.eval()
mode = 'hw_vn'


def get_transcript(org_img):
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
        'len_label': None,
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
        return None

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
    paragraph = out['hw_vn']

    for v in range(len(paragraph)):
        line = paragraph[v]
        temp = np.copy(line[:, 0])
        for i in range(0, len(line[0]) - 1):
            line[:, i] = line[:, i + 1]
        line[:, len(line[0]) - 1] = temp
        softmax_line = softmax(line)
        paragraph[v] = softmax_line

    output_strings = []
    for line in paragraph:
        res = beam_search_with_lm(line)
        output_strings.append(res)

    del resized_img, full_img, paragraph, out
    torch.cuda.empty_cache()

    return output_strings
