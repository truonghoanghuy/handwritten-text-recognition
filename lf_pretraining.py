import os
import sys
from typing import List

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from lf import lf_dataset, lf_loss
from lf.lf_dataset import LfDataset
from lf.line_follower import LineFollower
from utils import module_trainer
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        config = yaml.load(f)
    pretrain_config = config['pretraining']
    batches_per_epoch = int(pretrain_config['lf']['images_per_epoch'] / pretrain_config['lf']['batch_size'])
    checkpoint_filepath = os.path.join(pretrain_config['snapshot_path'], 'lf_checkpoint.pt')

    training_set_list = load_file_list(pretrain_config['training_set'])
    train_dataloader = DataLoader(LfDataset(training_set_list, augmentation=True),
                                  batch_size=1, shuffle=True, num_workers=0,
                                  collate_fn=lf_dataset.collate)
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)
    eval_set_list = load_file_list(pretrain_config['validation_set'])
    eval_dataloader = DataLoader(LfDataset(eval_set_list),
                                 batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=lf_dataset.collate)

    d_type = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    line_follower = LineFollower().to(device)
    optimizer = torch.optim.Adam(line_follower.parameters(), lr=pretrain_config['lf']['learning_rate'])


    def calculate_lf_loss(lf_model: nn.Module, input, train=True):
        input = input[0]  # Only single batch for now
        lf_xyrs = input['lf_xyrs']  # type: List[torch.Tensor]
        lf_xyxy = input['lf_xyxy']  # type: List[torch.Tensor]
        img = input['img']  # type: torch.Tensor

        x_i: torch.Tensor
        positions = [x_i.to(device, d_type).unsqueeze_(0) for x_i in lf_xyrs]
        xy_positions = [x_i.to(device, d_type).unsqueeze_(0) for x_i in lf_xyxy]
        img = img.to(device, d_type).unsqueeze_(0)
        # There might be a way to handle this case later, but for now we will skip it
        if len(xy_positions) <= 1:
            raise UserWarning('Skipped 1 sample')
        if train:
            grid_line, _, _, xy_output = lf_model(img, positions[:1], steps=len(positions), skip_grid=True,
                                                  all_positions=positions, reset_interval=4, randomize=True)
        else:
            grid_line, _, _, xy_output = lf_model(img, positions[:1], steps=len(positions),
                                                  skip_grid=True)
        loss = lf_loss.point_loss(xy_output, xy_positions)
        return loss


    def calculate_lf_train_loss(lf_model, input):
        return calculate_lf_loss(lf_model, input, True)


    def calculate_lf_evaluate_loss(lf_model, input):
        return calculate_lf_loss(lf_model, input, False)


    module_trainer.train(line_follower, calculate_lf_train_loss, calculate_lf_evaluate_loss,
                         optimizer, train_dataloader, eval_dataloader,
                         checkpoint_filepath, pretrain_config['lf']['stop_after_no_improvement'])
