import os
import sys
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import sol
from sol.alignment_loss import alignment_loss
from sol.crop_transform import CropTransform
from sol.sol_dataset import SolDataset
from start_of_line_finder import StartOfLineFinder
from utils import transformation_utils
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper

sys.stdout.flush()

with open(sys.argv[1]) as f:
    config = yaml.load(f)

sol_network_config = config['network']['sol']
pretrain_config = config['pretraining']

training_set_list = load_file_list(pretrain_config['training_set'])
train_dataset = SolDataset(training_set_list,
                           rescale_range=pretrain_config['sol']['training_rescale_range'],
                           transform=CropTransform(pretrain_config['sol']['crop_params']))

train_dataloader = DataLoader(train_dataset,
                              batch_size=pretrain_config['sol']['batch_size'],
                              shuffle=True, num_workers=0,
                              collate_fn=sol.sol_dataset.collate)

batches_per_epoch = int(pretrain_config['sol']['images_per_epoch'] / pretrain_config['sol']['batch_size'])
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list = load_file_list(pretrain_config['validation_set'])
test_dataset = SolDataset(test_set_list,
                          rescale_range=pretrain_config['sol']['validation_rescale_range'],
                          transform=None)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=sol.sol_dataset.collate)

base0 = sol_network_config['base0']
base1 = sol_network_config['base1']

alpha_alignment = pretrain_config['sol']['alpha_alignment']
alpha_backprop = pretrain_config['sol']['alpha_backprop']

d_type = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sol = StartOfLineFinder(base0, base1).to(device)

optimizer = torch.optim.Adam(sol.parameters(), lr=pretrain_config['sol']['learning_rate'])
lowest_loss = np.inf
cnt_since_last_improvement = 0

for epoch in range(1000):
    print(f"Epoch {epoch}")
    sum_loss = 0.0
    steps = 0.0
    sol.train()

    for x in train_dataloader:
        img: torch.Tensor = x['img']
        img = img.to(device=device, dtype=d_type)

        sol_gt = None  # type: Optional[torch.Tensor]
        if x['sol_gt'] is not None:
            # This is needed because if sol_gt is None it means that there
            # no GT positions in the image. The alignment loss will handle,
            # it correctly as None
            sol_gt = x['sol_gt']
            sol_gt = sol_gt.to(device=device, dtype=d_type)

        predictions = sol(img)
        predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
        loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        steps += 1

    print(f"Train Loss {sum_loss / steps}")
    # print(f"Real Epoch {train_dataloader.epoch}")

    sol.eval()
    sum_loss = 0.0
    steps = 0.0

    with torch.no_grad():
        for step_i, x in enumerate(test_dataloader):
            img = x['img']
            sol_gt = x['sol_gt']
            img = img.to(device=device, dtype=d_type)
            sol_gt = sol_gt.to(device=device, dtype=d_type)

            predictions = sol(img)
            predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
            loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)

            # Write images to file to visualization
            # org_img = img[0].data.cpu().numpy().transpose([2,1,0])
            # org_img = ((org_img + 1)*128).astype(np.uint8)
            # org_img = org_img.copy()
            # org_img = drawing.draw_sol_torch(predictions, org_img)
            # cv2.imwrite("data/sol_val_2/{}.png".format(step_i), org_img)

            sum_loss += loss.item()
            steps += 1

    cnt_since_last_improvement += 1
    if lowest_loss > sum_loss / steps:
        cnt_since_last_improvement = 0
        lowest_loss = sum_loss / steps
        print("Saving Best")

        if not os.path.exists(pretrain_config['snapshot_path']):
            os.makedirs(pretrain_config['snapshot_path'])

        torch.save(sol.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'sol.pt'))

    print(f"Test Loss {sum_loss / steps} {lowest_loss} \n")

    if cnt_since_last_improvement >= pretrain_config['sol']['stop_after_no_improvement']:
        break
