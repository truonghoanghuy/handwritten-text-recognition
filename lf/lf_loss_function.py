from typing import List, Optional, Dict

import torch
from torch.nn import Module

from lf import lf_loss
from utils import string_utils, error_rates


def calculate(lf_model: Module, input, dtype: torch.dtype, device: torch.device,
              train=True, hw: Optional[Module] = None, idx_to_char: Optional[Dict] = None):
    input = input[0]  # Only single batch for now
    lf_xyrs = input['lf_xyrs']  # type: List[torch.Tensor]
    lf_xyxy = input['lf_xyxy']  # type: List[torch.Tensor]
    img = input['img']  # type: torch.Tensor

    x_i: torch.Tensor
    positions = [x_i.to(device, dtype).unsqueeze_(0) for x_i in lf_xyrs]
    xy_positions = [x_i.to(device, dtype).unsqueeze_(0) for x_i in lf_xyxy]
    img = img.to(device, dtype).unsqueeze_(0)
    # There might be a way to handle this case later, but for now we will skip it
    if len(xy_positions) <= 1:
        raise UserWarning('Skipped 1 sample')
    if train:
        grid_line, _, _, xy_output = lf_model(img, positions[0], steps=len(positions), skip_grid=True,
                                              all_positions=positions, reset_interval=4, randomize=True)
        loss = lf_loss.point_loss(xy_output, xy_positions)
    else:
        skip_hw = hw is None or idx_to_char is None
        grid_line, _, _, xy_output = lf_model(img, positions[0], steps=len(positions), skip_grid=skip_hw)
        if skip_hw:
            loss = lf_loss.point_loss(xy_output, xy_positions)
        else:
            # noinspection PyArgumentList
            line = torch.nn.functional.grid_sample(img.transpose(2, 3), grid_line, align_corners=True)
            line = line.transpose(2, 3)
            predictions = hw(line)

            out = predictions.permute(1, 0, 2).data.cpu().numpy()
            gt_line = input['gt']
            pred, raw_pred = string_utils.naive_decode(out[0])
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            loss = error_rates.cer(gt_line, pred_str)
    return loss
