import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from sol import sol_dataset
from sol import sol_loss_function
from sol.crop_transform import CropTransform
from sol.sol_dataset import SolDataset
from sol.start_of_line_finder import StartOfLineFinder
from utils import module_trainer
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper

# noinspection DuplicatedCode
if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        config = yaml.load(f)
    train_config = config['training']
    base0 = config['network']['sol']['base0']
    base1 = config['network']['sol']['base1']
    alpha_alignment = train_config['sol']['alpha_alignment']
    alpha_backprop = train_config['sol']['alpha_backprop']
    batches_per_epoch = int(train_config['sol']['images_per_epoch'] / train_config['sol']['batch_size'])
    checkpoint_filepath = os.path.join(train_config['snapshot']['best_validation'], 'sol_checkpoint.pt')

    train_set_list = load_file_list(train_config['training_set'])
    train_dataset = SolDataset(train_set_list, rescale_range=train_config['sol']['training_rescale_range'],
                               transform=CropTransform(train_config['sol']['crop_params']))
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['sol']['batch_size'], shuffle=True,
                                  num_workers=0, collate_fn=sol_dataset.collate)
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

    eval_set_list = load_file_list(train_config['validation_set'])
    eval_dataset = SolDataset(eval_set_list, rescale_range=train_config['sol']['validation_rescale_range'],
                              random_subset_size=train_config['sol']['validation_subset_size'], transform=None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=sol_dataset.collate)

    dtype = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sol = StartOfLineFinder(base0, base1).to(device)
    optimizer = torch.optim.Adam(sol.parameters(), lr=train_config['sol']['learning_rate'])


    def sol_loss(model, input):
        return sol_loss_function.calculate(model, input, dtype, device, alpha_alignment, alpha_backprop)


    trainer = module_trainer.ModuleTrainer(sol, optimizer, sol_loss, sol_loss, train_dataloader, eval_dataloader,
                                           checkpoint_filepath)
    resume = 'resume' in sys.argv
    trainer.train(resume=resume, continuous_training=True)
