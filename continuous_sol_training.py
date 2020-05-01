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
from utils.continuous_training_util import ContinuousTrainingUtil
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper

# noinspection DuplicatedCode
if __name__ == '__main__':
    resume = 'resume' in sys.argv
    with open(sys.argv[1]) as f:
        config = yaml.load(f)
    train_config = config['training']
    base0 = config['network']['sol']['base0']
    base1 = config['network']['sol']['base1']
    alpha_alignment = train_config['sol']['alpha_alignment']
    alpha_backprop = train_config['sol']['alpha_backprop']
    batches_per_epoch = int(train_config['sol']['images_per_epoch'] / train_config['sol']['batch_size'])
    checkpoint_filepath = os.path.join(train_config['snapshot']['best_validation'], 'sol_checkpoint.pt')
    dtype = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sol = StartOfLineFinder(base0, base1).to(device)
    optimizer = torch.optim.Adam(sol.parameters(), lr=train_config['sol']['learning_rate'])
    trainer = module_trainer.ModuleTrainer(sol, optimizer, None, None, None, None,
                                           checkpoint_filepath)


    def load_and_train(load_checkpoint):
        train_set_list = load_file_list(train_config['training_set'])  # noqa
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

        def calculate_sol_loss(model, input):
            return sol_loss_function.calculate(model, input, dtype, device, alpha_alignment, alpha_backprop)

        trainer.train_loss_func = calculate_sol_loss
        trainer.evaluate_loss_func = calculate_sol_loss
        trainer.train_dataloader = train_dataloader
        trainer.eval_dataloader = eval_dataloader
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
