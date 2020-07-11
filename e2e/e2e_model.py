from typing import List

import numpy as np
import torch
from torch import nn


class E2EModel(nn.Module):
    def __init__(self, sol, lf, hw, hw_vn=None, dtype=torch.float32, use_cpu=False):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.sol = sol
        self.lf = lf
        self.hw = hw
        self.hw_vn = hw_vn if hw_vn else hw
        if not use_cpu:
            self.to_cuda()
        else:
            self.to_cpu()

    def to_cuda(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sol = self.sol.to(self.device)
        self.lf = self.lf.to(self.device)
        self.hw = self.hw.to(self.device)
        self.hw_vn = self.hw_vn.to(self.device)

    def to_cpu(self):
        self.device = torch.device('cpu')
        self.sol = self.sol.to(self.device)
        self.lf = self.lf.to(self.device)
        self.hw = self.hw.to(self.device)
        self.hw_vn = self.hw_vn.to(self.device)

    def train(self, mode=True):
        self.sol.train(mode)
        self.lf.train(mode)
        self.hw.train(mode)
        self.hw_vn.train(mode)

    def eval(self):
        self.sol.eval()
        self.lf.eval()
        self.hw.eval()
        self.hw_vn.eval()

    def forward(self, x, use_full_img=True, sol_threshold=0.1, lf_batch_size=10, mode='hw'):
        resized_img: torch.Tensor = x['resized_img']
        resized_img = resized_img.to(self.device, self.dtype)
        if use_full_img:
            full_img: torch.Tensor = x['full_img']
            full_img = full_img.to(self.device, self.dtype)
            scale = x['resize_scale']
            results_scale = 1.0
        else:
            full_img = resized_img
            scale = 1.0
            results_scale = x['resize_scale']

        # start of line finder
        original_starts: torch.Tensor = self.sol(resized_img)  # size: (N, num_sol, 5)
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
            _, c, h, w = full_img.size()
            batch_image = full_img.expand(sol_batch.size(0), c, h, w)

            steps = 5
            extra_backward = 1
            forward_steps = 40
            grid_lines, _, next_windows, batch_xy_positions = self.lf(batch_image, sol_batch,
                                                                      steps=steps)
            grid_lines, _, next_windows, batch_xy_positions = self.lf(batch_image, next_windows[steps],
                                                                      steps=steps + extra_backward, negate_lw=True)
            grid_lines, _, next_windows, batch_xy_positions = self.lf(batch_image, next_windows[steps + extra_backward],
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
        hw_vn_out = []
        max_w = max([line_batch.size(3) for line_batch in line_batches])
        for line_batch in line_batches:
            b, c, h, w = line_batch.size()
            padded = torch.zeros(b, c, h, max_w - w, dtype=self.dtype, device=self.device)
            line_batch = torch.cat([line_batch, padded], dim=3)
            for line in line_batch:
                line = line.transpose(0, 1).transpose(1, 2)  # (C, H, W) -> (H, W, C)
                line = (line + 1) * 128  # [-1, 1) -> [0, 256)
                line_np = line.data.cpu().numpy()
                line_images.append(line_np)
            if mode == 'hw':
                hw_vn_pred = self.hw_vn(line_batch)
            else:
                hw_vn_pred = self.hw_vn(line_batch, x['len_label'])
            hw_pred = self.hw(line_batch)
            hw_out.append(hw_pred)
            hw_vn_out.append(hw_vn_pred)
        hw_out = torch.cat(hw_out, dim=1)
        hw_out = hw_out.transpose(0, 1)
        hw_vn_out = torch.cat(hw_vn_out, dim=1)
        hw_vn_out = hw_vn_out.transpose(0, 1)

        # import cv2
        # for i, image in enumerate(line_images):
        #     cv2.imwrite(f'debug/line_image_{i}.png', image)
        return {
            'original_sol': original_starts,
            'sol': positions,
            'lf': lf_xy_positions,
            'hw': hw_out,
            'hw_vn': hw_vn_out,
            'results_scale': results_scale,
            'line_imgs': line_images
        }
