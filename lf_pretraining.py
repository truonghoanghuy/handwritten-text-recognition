import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from lf import lf_dataset, lf_loss_function
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
    train_dataset = LfDataset(training_set_list, augmentation=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                                  collate_fn=lf_dataset.collate)
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)
    eval_set_list = load_file_list(pretrain_config['validation_set'])
    eval_dataset = LfDataset(eval_set_list)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=lf_dataset.collate)

    dtype = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lf = LineFollower().to(device)
    optimizer = torch.optim.Adam(lf.parameters(), lr=pretrain_config['lf']['learning_rate'])


    def calculate_lf_train_loss(lf_model, input):
        return lf_loss_function.calculate(lf_model, input, dtype, device, train=True)


    def calculate_lf_evaluate_loss(lf_model, input):
        return lf_loss_function.calculate(lf_model, input, dtype, device, train=False)


    trainer = module_trainer.ModuleTrainer(lf, optimizer, calculate_lf_train_loss, calculate_lf_evaluate_loss,
                                           train_dataloader, eval_dataloader, checkpoint_filepath)
    trainer.train(stop_after_no_improvement=pretrain_config['sol']['stop_after_no_improvement'])
