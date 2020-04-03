import os
import sys

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from sol import sol_dataset
from sol.alignment_loss import alignment_loss
from sol.crop_transform import CropTransform
from sol.sol_dataset import SolDataset
from sol.start_of_line_finder import StartOfLineFinder
from utils import transformation_utils, module_trainer
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper

if __name__ == '__main__':
    sys.stdout.flush()
    with open(sys.argv[1]) as f:
        config = yaml.load(f)
    pretrain_config = config['pretraining']
    base0 = config['network']['sol']['base0']
    base1 = config['network']['sol']['base1']
    alpha_alignment = pretrain_config['sol']['alpha_alignment']
    alpha_backprop = pretrain_config['sol']['alpha_backprop']
    batches_per_epoch = int(pretrain_config['sol']['images_per_epoch'] / pretrain_config['sol']['batch_size'])
    checkpoint_filepath = os.path.join(pretrain_config['snapshot_path'], 'sol_checkpoint.pt')

    training_set_list = load_file_list(pretrain_config['training_set'])
    train_dataset = SolDataset(training_set_list,
                               rescale_range=pretrain_config['sol']['training_rescale_range'],
                               transform=CropTransform(pretrain_config['sol']['crop_params']))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=pretrain_config['sol']['batch_size'], shuffle=True, num_workers=0,
                                  collate_fn=sol_dataset.collate)
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

    test_set_list = load_file_list(pretrain_config['validation_set'])
    test_dataset = SolDataset(test_set_list,
                              rescale_range=pretrain_config['sol']['validation_rescale_range'],
                              transform=None)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=sol_dataset.collate)

    d_type = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sol = StartOfLineFinder(base0, base1).to(device)
    optimizer = torch.optim.Adam(sol.parameters(), lr=pretrain_config['sol']['learning_rate'])


    def calculate_sol_loss(sol_model: nn.Module, input):
        img: torch.Tensor = input['img']
        img = img.to(device=device, dtype=d_type)
        sol_gt = input['sol_gt']
        if isinstance(sol_gt, torch.Tensor):
            # This is needed because if sol_gt is None it means that there
            # no GT positions in the image. The alignment loss will handle,
            # it correctly as None
            sol_gt = sol_gt.to(device=device, dtype=d_type)
        predictions = sol_model(img)
        predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
        loss = alignment_loss(predictions, sol_gt, input['label_sizes'], alpha_alignment, alpha_backprop)
        return loss


    module_trainer.train(sol, calculate_sol_loss, calculate_sol_loss,
                         optimizer, train_dataloader, test_dataloader,
                         checkpoint_filepath, pretrain_config['sol']['stop_after_no_improvement'])
