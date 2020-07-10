import yaml
import json
import numpy as np
import torch
import cv2

from e2e import e2e_postprocessing
from e2e.e2e_model import E2EModel
from utils.paragraph_processing import softmax
from hw_vn.continuous_state import init_model
from hw_vn.beam_search_with_lm import beam_search_with_lm

config_path = 'e2e_config.yaml'
with open(config_path) as f:
    config = yaml.load(f)

char_set_path = 'data/char_set_vn.json'
with open(char_set_path, encoding='utf8') as f:
    char_set = json.load(f)
idx_to_char = {int(k): v for k, v in char_set['idx_to_char'].items()}
char_to_idx = char_set['char_to_idx']

model_mode = 'best_overall'
hw_model = 'cnn_lstm'
mode = 'hw_vn'
sol, lf, hw = init_model(config, sol_dir=model_mode, lf_dir=model_mode, hw_dir=model_mode,
                         use_cpu=False, hw_model=hw_model)

e2e = E2EModel(sol, lf, hw, use_cpu=False)
e2e.eval()


def get_transcript(org_img):
    target_dim1 = 512
    s = target_dim1 / float(org_img.shape[1])

    pad_amount = 128
    org_img = np.pad(org_img, ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), 'constant',
                     constant_values=255)

    target_dim0 = int(org_img.shape[0] * s)
    target_dim1 = int(org_img.shape[1] * s)

    full_img = org_img.astype(np.float32)
    full_img = full_img.transpose([2, 1, 0])[None, ...]
    full_img = torch.from_numpy(full_img)
    full_img = full_img / 128 - 1

    img = cv2.resize(org_img, (target_dim1, target_dim0), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = img.transpose([2, 1, 0])[None, ...]
    img = torch.from_numpy(img)
    img = img / 128 - 1

    e2e_input = {
        'resized_img': img,
        'full_img': full_img,
        'resize_scale': 1.0 / s,
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
    paragraph = out['hw']

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

    # param = {'hw': np.array(paragraph)}
    # output_strings, _ = e2e_postprocessing.decode_handwriting(param, idx_to_char)
    return output_strings
