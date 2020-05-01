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
from utils.continuous_training_util import ContinuousTrainingUtil
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper

# noinspection DuplicatedCode
if __name__ == '__main__':
    resume = 'resume' in sys.argv
    with open(sys.argv[1]) as f:
        config = yaml.load(f)
    hw_network_config = config['network']['hw']
    char_set_path = hw_network_config['char_set_path']
    with open(char_set_path) as f:
        char_set = json.load(f)
    idx_to_char = {int(k): v for k, v in char_set['idx_to_char'].items()}
    train_config = config['training']
    batches_per_epoch = int(train_config['hw']['images_per_epoch'] / train_config['hw']['batch_size'])
    checkpoint_filepath = os.path.join(train_config['snapshot']['best_validation'], 'hw_checkpoint.pt')
    dtype = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hw = cnn_lstm.create_model(hw_network_config).to(device)
    optimizer = torch.optim.Adam(hw.parameters(), lr=train_config['hw']['learning_rate'])
    criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)
    trainer = module_trainer.ModuleTrainer(hw, optimizer, None, None, None, None, checkpoint_filepath)


    def load_and_train(load_checkpoint):
        train_set_list = load_file_list(train_config['training_set'])  # noqa
        train_dataset = HwDataset(train_set_list, char_set['char_to_idx'], augmentation=True,
                                  img_height=hw_network_config['input_height'])
        train_dataloader = DataLoader(train_dataset, batch_size=train_config['hw']['batch_size'], shuffle=True,
                                      num_workers=0, collate_fn=hw_dataset.collate)
        train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

        eval_set_list = load_file_list(train_config['validation_set'])
        eval_dataset = HwDataset(eval_set_list, char_set['char_to_idx'], img_height=hw_network_config['input_height'],
                                 random_subset_size=train_config['hw']['validation_subset_size'])
        eval_dataloader = DataLoader(eval_dataset, batch_size=train_config['hw']['batch_size'], shuffle=False,
                                     num_workers=0, collate_fn=hw_dataset.collate)

        def calculate_hw_train_loss(hw_model, input):
            return hw_loss_function.calculate_hw_loss(hw_model, input, dtype, device, criterion, idx_to_char,
                                                      train=True)

        def calculate_hw_evaluate_loss(hw_model, input):
            return hw_loss_function.calculate_hw_loss(hw_model, input, dtype, device, criterion, idx_to_char,
                                                      train=False)

        trainer.train_dataloader = train_dataloader
        trainer.eval_dataloader = eval_dataloader
        trainer.train_loss_func = calculate_hw_train_loss
        trainer.evaluate_loss_func = calculate_hw_evaluate_loss
        refresh_interval = train_dataloader.epoch_steps / batches_per_epoch
        refresh_interval = (refresh_interval % 1 != 0) + int(refresh_interval)  # non-negative ceiling
        trainer.train(load_checkpoint=load_checkpoint, resume=resume, continuous_training=True,
                      max_iter=refresh_interval)


    full_epoch = 0
    while ContinuousTrainingUtil.is_running():
        print()
        if full_epoch != 0:
            print('Reloading aligned dataset...')
        print(f'Full epoch {full_epoch}')
        load_and_train(full_epoch == 0)
        full_epoch += 1
