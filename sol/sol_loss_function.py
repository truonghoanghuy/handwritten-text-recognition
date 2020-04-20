import torch
from torch.nn import Module

from sol.alignment_loss import alignment_loss
from utils import transformation_utils


def calculate(sol_model: Module, input, dtype: torch.dtype, device: torch.device, alpha_alignment, alpha_backprop):
    img: torch.Tensor = input['img']
    img = img.to(device=device, dtype=dtype)
    sol_gt = input['sol_gt']
    if isinstance(sol_gt, torch.Tensor):
        # This is needed because if sol_gt is None it means that there
        # no GT positions in the image. The alignment loss will handle,
        # it correctly as None
        sol_gt = sol_gt.to(device=device, dtype=dtype)
    predictions = sol_model(img)
    predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
    loss = alignment_loss(predictions, sol_gt, input['label_sizes'], alpha_alignment, alpha_backprop)
    return loss
