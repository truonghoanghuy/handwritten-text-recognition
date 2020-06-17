import torch
import cv2
import numpy as np
import yaml
import sys
import json
from os.path import join

from e2e import e2e_postprocessing
from e2e.e2e_model import E2EModel
from utils import continuous_state


def forward(x, e2e_model, use_full_img=True, sol_threshold=0.1, lf_batch_size=10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    resized_img: torch.Tensor = x['resized_img']
    resized_img = resized_img.to(device, dtype)
    if use_full_img:
        full_img: torch.Tensor = x['full_img']
        full_img = full_img.to(device, dtype)
        scale = x['resize_scale']
        results_scale = 1.0
    else:
        full_img = resized_img
        scale = 1.0
        results_scale = x['resize_scale']

    # start of line finder
    original_starts: torch.Tensor = e2e_model.sol(resized_img)  # size: (N, num_sol, 5)
    start = original_starts
    # softer threshold to ensure at least one point will be taken
    sorted_start, _sorted_indices = torch.sort(start[..., 0:1], dim=1, descending=True)
    soft_sol_threshold = sorted_start[0, 1, 0].data.cpu().item()
    accept_threshold = min(sol_threshold, soft_sol_threshold)
    select = start[..., 0:1] >= accept_threshold
    select_idx = np.where(select.data.cpu().numpy())[1]
    start = start[:, select_idx, :]

    start = start.transpose(0, 1)
    positions = torch.cat([
        start[..., 1:3] * scale,
        start[..., 3:4],
        start[..., 4:5] * scale,
        start[..., 0:1]
    ], 2)

    # line follower
    all_xy_positions = []  # type: List[List[torch.Tensor]]  # element size = (num_sol, 3, 2) * num_lf_detected
    line_batches = []  # type: List[torch.Tensor]  # correspond to multiple SOL. element size is (N, C, H, W)
    line_images = []  # type: List[np.ndarray]
    for s in range(0, positions.size(0), lf_batch_size):
        torch.cuda.empty_cache()
        sol_batch = positions[s:s + lf_batch_size, 0, :]
        _, C, H, W = full_img.size()
        batch_image = full_img.expand(sol_batch.size(0), C, H, W)

        steps = 5
        extra_backward = 1
        forward_steps = 40
        grid_lines, _, next_windows, batch_xy_positions = e2e_model.lf(batch_image, sol_batch,
                                                                  steps=steps)
        grid_lines, _, next_windows, batch_xy_positions = e2e_model.lf(batch_image, next_windows[steps],
                                                                  steps=steps + extra_backward, negate_lw=True)
        grid_lines, _, next_windows, batch_xy_positions = e2e_model.lf(batch_image, next_windows[steps + extra_backward],
                                                                  steps=forward_steps, allow_end_early=True)
        all_xy_positions.append(batch_xy_positions)

        batch_image = batch_image.transpose(2, 3).detach()
        # noinspection PyArgumentList
        line_batch = torch.nn.functional.grid_sample(batch_image, grid_lines, align_corners=True)
        line_batch = line_batch.transpose(2, 3)
        line_batches.append(line_batch)

    del batch_image

    # repeat the last element to have the length = num_lf_detected on all element
    lf_xy_positions = []
    max_len = max([len(batch_xy_positions) for batch_xy_positions in all_xy_positions])
    for i, batch_xy_positions in enumerate(all_xy_positions):
        padded = batch_xy_positions + [batch_xy_positions[-1]] * (max_len - len(batch_xy_positions))
        if i == 0:
            lf_xy_positions = padded
        else:
            for j in range(len(lf_xy_positions)):
                lf_xy_positions[j] = torch.cat((lf_xy_positions[j], padded[j]), dim=0)

    # pad constant to have the same width on every line images
    hw_out = []
    maxW = max([line_batch.size(3) for line_batch in line_batches])
    line_images = []
    for line_batch in line_batches:
        N, C, H, W = line_batch.size()
        padded = torch.zeros(N, C, H, maxW - W, dtype=dtype, device=device)
        line_batch = torch.cat([line_batch, padded], dim=3)
        for line in line_batch:
            line = line.transpose(0, 1).transpose(1, 2)  # (C, H, W) -> (H, W, C)
            line = (line + 1) * 128  # [-1, 1) -> [0, 256)
            line_np = line.data.cpu().numpy()
            line_images.append(line_np)
        hw_pred = e2e_model.hw(line_batch)
        hw_out.append(hw_pred)
    hw_out = torch.cat(hw_out, dim=1)
    hw_out = hw_out.transpose(0, 1)

    # import cv2
    # for i, image in enumerate(line_images):
    #     cv2.imwrite(f'debug/line_image_{i}.png', image)
    return {
        'original_sol': original_starts,
        'sol': positions,
        'lf': lf_xy_positions,
        'hw': hw_out,
        'results_scale': results_scale,
        'line_imgs': line_images
    }


def process_image(img_path, config_path, resize_width=512):
    org_img = cv2.imread(img_path)
    with open(config_path) as f:
        config = yaml.load(f)
    with open(config['network']['hw']['char_set_path'], encoding='utf8') as f:
        char_set = json.load(f)
    idx_to_char = char_set['idx_to_char']
    idx_to_char = {int(k): v for k, v in char_set['idx_to_char'].items()}

    full_img = org_img.astype(np.float32)
    full_img = full_img.transpose([2, 1, 0])[None, ...]
    full_img = torch.from_numpy(full_img)
    full_img = full_img / 128 - 1

    target_dim1 = resize_width
    s = target_dim1 / float(org_img.shape[1])
    target_dim0 = int(org_img.shape[0] / float(org_img.shape[1]) * target_dim1)

    img = cv2.resize(org_img, (target_dim1, target_dim0), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = img.transpose([2, 1, 0])[None, ...]
    img = torch.from_numpy(img)
    img = img / 128 - 1

    input = {'resized_img': img, 'full_img': full_img, "resize_scale": 1.0 / s}

    sol, lf, hw = continuous_state.init_model(config)
    e2e = E2EModel(sol, lf, hw)

    try:
        with torch.no_grad():
            out_original = forward(input, e2e_model=e2e, lf_batch_size=config['network']['lf']['batch_size'])
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            e2e.to_cpu()
            with torch.no_grad():
                out_original = forward(input, e2e_model=e2e, lf_batch_size=100)
            e2e.to_cuda()
        else:
            raise e

    line_images = out_original['line_imgs']
    for idx, line_image in enumerate(line_images):
        cv2.imwrite(f'output/{idx}.jpg', line_image)

    out_original = e2e_postprocessing.results_to_numpy(out_original)
    out_original['idx'] = np.arange(out_original['sol'].shape[0])
    e2e_postprocessing.trim_ends(out_original)
    '''
    decoded_hw, decoded_raw_hw = e2e_postprocessing.decode_handwriting(out_original, idx_to_char)
    output_decoded_hw = '\n'.join(decoded_hw)
    with open('output/hw.txt', 'w', encoding='utf8') as f:
        f.write(output_decoded_hw)
    '''

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Your arguments is incorrect!')
    else:
        process_image(sys.argv[1], sys.argv[2])