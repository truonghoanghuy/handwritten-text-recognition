from typing import Dict

import torch
from torch.nn import functional, Module, CTCLoss

from utils import string_utils, error_rates


def calculate_hw_loss(hw_model: Module, input, dtype: torch.dtype, device: torch.device,
                      criterion: CTCLoss, idx_to_char: Dict, train=True):
    line_imgs = input['line_imgs']  # type: torch.Tensor
    labels = input['labels']  # type: torch.Tensor
    label_lengths = input['label_lengths']  # type: torch.Tensor
    line_imgs = line_imgs.to(device, dtype)
    predicts: torch.Tensor = hw_model(line_imgs).cpu()  # predicts size: (input_length, batch_size, num_classes)
    if train:
        inputs = functional.log_softmax(predicts, dim=2)
        input_length, batch_size, _ = predicts.size()
        input_lengths = torch.tensor([input_length] * batch_size)
        loss = criterion(inputs, labels, input_lengths, label_lengths)
        return loss
    else:
        outputs = predicts.permute(1, 0, 2).data.numpy()
        cer = 0.0
        steps = 0
        for i, gt_line in enumerate(input['gt']):
            logits = outputs[i, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer += error_rates.cer(gt_line, pred_str)
            steps += 1
        return cer / steps
