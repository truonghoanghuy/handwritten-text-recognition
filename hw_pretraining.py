import json
import os
import sys

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from hw import cnn_lstm
from hw import hw_dataset
from hw.hw_dataset import HwDataset
from utils import string_utils, error_rates, module_trainer
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper

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
    train_dataset = HwDataset(training_set_list,
                              char_set['char_to_idx'], augmentation=True,
                              img_height=hw_network_config['input_height'])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=pretrain_config['hw']['batch_size'],
                                  shuffle=True, num_workers=0, drop_last=True,
                                  collate_fn=hw_dataset.collate)
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

    eval_set_list = load_file_list(pretrain_config['validation_set'])
    eval_dataset = HwDataset(eval_set_list,
                             char_set['char_to_idx'],
                             img_height=hw_network_config['input_height'])
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=pretrain_config['hw']['batch_size'],
                                 shuffle=False, num_workers=0,
                                 collate_fn=hw_dataset.collate)

    d_type = torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hw = cnn_lstm.create_model(hw_network_config).to(device)
    optimizer = torch.optim.Adam(hw.parameters(), lr=pretrain_config['hw']['learning_rate'])
    criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)


    def calculate_hw_loss(hw_model: nn.Module, input, train=True):
        line_imgs = input['line_imgs']  # type: torch.Tensor
        labels = input['labels']  # type: torch.Tensor
        label_lengths = input['label_lengths']  # type: torch.Tensor

        line_imgs = line_imgs.to(device, d_type)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        predicts = hw_model(line_imgs).cpu()  # type: torch.Tensor
        output_batch = predicts.permute(1, 0, 2)
        out = output_batch.data.numpy()

        if train:
            batch_size = predicts.size(1)
            predicts_size = torch.tensor([predicts.size(0)] * batch_size)
            loss = criterion(predicts, labels, predicts_size, label_lengths)
            return loss
        else:
            cer = 0.0
            for i, gt_line in enumerate(input['gt']):
                logits = out[i, ...]
                pred, raw_pred = string_utils.naive_decode(logits)
                pred_str = string_utils.label2str_single(pred, idx_to_char, False)
                cer += error_rates.cer(gt_line, pred_str)
            return torch.tensor(cer)  # input to the module_trainer.train() method must be a tensor


    def calculate_hw_train_loss(hw_model, input):
        return calculate_hw_loss(hw_model, input, True)


    def calculate_hw_evaluate_loss(hw_model, input):
        return calculate_hw_loss(hw_model, input, False)


    module_trainer.train(hw, calculate_hw_train_loss, calculate_hw_evaluate_loss,
                         optimizer, train_dataloader, eval_dataloader,
                         checkpoint_filepath, pretrain_config['hw']['stop_after_no_improvement'], 0.9)
