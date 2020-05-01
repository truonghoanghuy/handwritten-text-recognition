import json
import os
import sys

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from hw import cnn_lstm
from hw import hw_dataset
from hw import hw_loss_function
from hw.hw_dataset import HwDataset
from utils import module_trainer
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper

# noinspection DuplicatedCode
if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        config = yaml.load(f)
    hw_network_config = config['network']['hw']
    pretrain_config = config['pretraining']
    batches_per_epoch = int(pretrain_config['hw']['images_per_epoch'] / pretrain_config['hw']['batch_size'])
    checkpoint_filepath = os.path.join(pretrain_config['snapshot_path'], 'hw_checkpoint.pt')

    char_set_path = hw_network_config['char_set_path']
    with open(char_set_path) as f:
        char_set = json.load(f)
    idx_to_char = {int(k): v for k, v in char_set['idx_to_char'].items()}

    training_set_list = load_file_list(pretrain_config['training_set'])
    train_dataset = HwDataset(training_set_list, char_set['char_to_idx'], augmentation=True,
                              img_height=hw_network_config['input_height'])
    train_dataloader = DataLoader(train_dataset, batch_size=pretrain_config['hw']['batch_size'], shuffle=True,
                                  num_workers=0, drop_last=True, collate_fn=hw_dataset.collate)
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

    eval_set_list = load_file_list(pretrain_config['validation_set'])
    eval_dataset = HwDataset(eval_set_list, char_set['char_to_idx'], img_height=hw_network_config['input_height'])
    eval_dataloader = DataLoader(eval_dataset, batch_size=pretrain_config['hw']['batch_size'], shuffle=False,
                                 num_workers=0, collate_fn=hw_dataset.collate)

    dtype = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hw = cnn_lstm.create_model(hw_network_config).to(device)
    optimizer = torch.optim.Adam(hw.parameters(), lr=pretrain_config['hw']['learning_rate'])
    criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)


    def calculate_hw_train_loss(hw_model, input):
        return hw_loss_function.calculate_hw_loss(hw_model, input, dtype, device, criterion, idx_to_char, train=True)


    def calculate_hw_evaluate_loss(hw_model, input):
        return hw_loss_function.calculate_hw_loss(hw_model, input, dtype, device, criterion, idx_to_char, train=False)


    trainer = module_trainer.ModuleTrainer(hw, optimizer, calculate_hw_train_loss, calculate_hw_evaluate_loss,
                                           train_dataloader, eval_dataloader, checkpoint_filepath, loss_tolerance=0.6)
    trainer.train(stop_after_no_improvement=pretrain_config['sol']['stop_after_no_improvement'])
