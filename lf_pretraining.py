import os
import sys
import time
from typing import List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from lf import lf_dataset, lf_loss
from lf.lf_dataset import LfDataset
from lf.line_follower import LineFollower
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper
from utils.dot_progress_printer import DotProgressPrinter

if __name__ == '__main__':
    sys.stdout.flush()
    with open(sys.argv[1]) as f:
        config = yaml.load(f)
    pretrain_config = config['pretraining']
    snapshot_path = pretrain_config['snapshot_path']
    batches_per_epoch = int(pretrain_config['lf']['images_per_epoch'] / pretrain_config['lf']['batch_size'])

    training_set_list = load_file_list(pretrain_config['training_set'])
    train_dataset = LfDataset(training_set_list, augmentation=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                                  collate_fn=lf_dataset.collate)
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)  # TODO: clean this ambiguous epoch count

    eval_set_list = load_file_list(pretrain_config['validation_set'])
    eval_dataset = LfDataset(eval_set_list)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=lf_dataset.collate)

    d_type = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    line_follower = LineFollower().to(device)
    optimizer = torch.optim.Adam(line_follower.parameters(), lr=pretrain_config['lf']['learning_rate'])
    lowest_loss = np.inf
    cnt_since_last_improvement = 0

    start_time = time.time()
    total_time = 0.0
    for epoch in range(1000):
        print(f'Epoch {epoch}')
        for phase in ['train', 'eval']:
            if phase == 'train':
                line_follower.train()
            else:
                line_follower.eval()

            sum_loss = 0.0
            steps = 0
            with torch.set_grad_enabled(phase == 'train'):
                loader = train_dataloader if phase == 'train' else eval_dataloader
                progress_printer = DotProgressPrinter(len(loader), 100)
                for index, batch in enumerate(loader):
                    input = batch[0]  # Only single batch for now
                    lf_xyrs = input['lf_xyrs']  # type: List[torch.Tensor]
                    lf_xyxy = input['lf_xyxy']  # type: List[torch.Tensor]
                    img = input['img']  # type: torch.Tensor

                    x_i: torch.Tensor
                    positions = [x_i.to(device, d_type).unsqueeze_(0) for x_i in lf_xyrs]
                    xy_positions = [x_i.to(device, d_type).unsqueeze_(0) for x_i in lf_xyxy]
                    img = img.to(device, d_type).unsqueeze_(0)

                    # There might be a way to handle this case later, but for now we will skip it
                    if len(xy_positions) <= 1:
                        continue

                    if phase == 'train':
                        grid_line, _, _, xy_output = line_follower(img, positions[:1], steps=len(positions),
                                                                   skip_grid=True, all_positions=positions,
                                                                   reset_interval=4, randomize=True)
                    else:
                        grid_line, _, _, xy_output = line_follower(img, positions[:1], steps=len(positions),
                                                                   skip_grid=True)
                    loss = lf_loss.point_loss(xy_output, xy_positions)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    sum_loss += loss.item()
                    steps += 1
                    progress_printer.step()
                print()
            if steps == 0:
                print('No data was loaded')
                steps = 1
            avg_loss = sum_loss / steps
            if phase == 'train':
                current = time.time()
                total_time = total_time + current - start_time
                print(f'Train loss = {avg_loss}, Real epoch = {train_dataloader.epoch}, '
                      f'Time elapsed: {total_time}, Last epoch: {current - start_time}')
            else:
                print(f'Eval loss = {avg_loss}, Current best loss = {lowest_loss}')
                if lowest_loss > avg_loss:
                    cnt_since_last_improvement = 0
                    lowest_loss = avg_loss
                    print('Better loss. Saving...')
                    if not os.path.exists(snapshot_path):
                        os.makedirs(snapshot_path)
                    torch.save(line_follower.state_dict(), os.path.join(snapshot_path, 'lf.pt'))
                cnt_since_last_improvement += 1
                if cnt_since_last_improvement >= pretrain_config['lf']['stop_after_no_improvement']:
                    exit(0)
