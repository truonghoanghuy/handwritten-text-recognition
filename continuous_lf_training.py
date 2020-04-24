import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from hw import cnn_lstm
from lf import lf_dataset, lf_loss_function
from lf.lf_dataset import LfDataset
from lf.line_follower import LineFollower
from utils import module_trainer, safe_load
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        config = yaml.load(f)
    char_set_path = config['network']['hw']['char_set_path']
    with open(char_set_path) as f:
        char_set = json.load(f)
    idx_to_char = {int(k): v for k, v in char_set['idx_to_char'].items()}
    train_config = config['training']
    hw_network_config = config['network']['hw']
    batches_per_epoch = int(train_config['lf']['images_per_epoch'] / train_config['lf']['batch_size'])
    lf_checkpoint_filepath = os.path.join(train_config['snapshot']['best_validation'], 'lf_checkpoint.pt')
    hw_checkpoint_filepath = os.path.join(train_config['snapshot']['best_validation'], 'hw_checkpoint.pt')

    train_set_list = load_file_list(train_config['training_set'])
    train_dataset = LfDataset(train_set_list, augmentation=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                                  collate_fn=lf_dataset.collate)
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)
    eval_set_list = load_file_list(train_config['validation_set'])
    eval_dataset = LfDataset(eval_set_list, random_subset_size=train_config['lf']['validation_subset_size'])
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=lf_dataset.collate)

    dtype = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hw_checkpoint = safe_load.load_checkpoint(hw_checkpoint_filepath)
    hw = cnn_lstm.create_model(hw_network_config)
    hw.load_state_dict(hw_checkpoint['model_state_dict'])
    hw = hw.to(device)
    hw.eval()
    lf = LineFollower().to(device)
    optimizer = torch.optim.Adam(lf.parameters(), lr=train_config['lf']['learning_rate'])


    def calculate_lf_train_loss(lf_model, input):
        return lf_loss_function.calculate(lf_model, input, dtype, device, train=True, hw=hw, idx_to_char=idx_to_char)


    def calculate_lf_evaluate_loss(lf_model, input):
        return lf_loss_function.calculate(lf_model, input, dtype, device, train=False, hw=hw, idx_to_char=idx_to_char)


    trainer = module_trainer.ModuleTrainer(lf, optimizer, calculate_lf_train_loss, calculate_lf_evaluate_loss,
                                           train_dataloader, eval_dataloader, lf_checkpoint_filepath)
    resume = 'resume' in sys.argv
    trainer.train(resume=resume, continuous_training=True)
