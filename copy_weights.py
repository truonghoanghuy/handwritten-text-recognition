import os
import shutil
import sys

import torch
import yaml

from hw import cnn_lstm
from lf.line_follower import LineFollower
from sol.start_of_line_finder import StartOfLineFinder
from utils import safe_load

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(sys.argv[1]) as f:
        config = yaml.load(f)
    best_overall_dir = config['training']['snapshot']['best_overall']
    best_validation_dir = config['training']['snapshot']['best_validation']
    if not os.path.exists(best_overall_dir):
        os.makedirs(best_overall_dir)
    if not os.path.exists(best_validation_dir):
        os.makedirs(best_validation_dir)

    sol_checkpoint_filepath = os.path.join(config['pretraining']['snapshot_path'], 'sol_checkpoint.pt')
    sol_checkpoint = safe_load.load_checkpoint(sol_checkpoint_filepath)
    sol = StartOfLineFinder(config['network']['sol']['base0'], config['network']['sol']['base1']).to(device)
    sol.load_state_dict(sol_checkpoint['model_state_dict'])
    target_filepath = os.path.join(best_overall_dir, 'sol.pt')
    torch.save(sol.state_dict(), target_filepath)
    target_filepath = os.path.join(best_validation_dir, 'sol_checkpoint.pt')
    shutil.copy(sol_checkpoint_filepath, target_filepath)

    lf_checkpoint_filepath = os.path.join(config['pretraining']['snapshot_path'], 'lf_checkpoint.pt')
    lf_checkpoint = safe_load.load_checkpoint(lf_checkpoint_filepath)
    lf = LineFollower().to(device)
    lf.load_state_dict(lf_checkpoint['model_state_dict'])
    target_filepath = os.path.join(best_overall_dir, 'lf.pt')
    torch.save(lf.state_dict(), target_filepath)
    target_filepath = os.path.join(best_validation_dir, 'lf_checkpoint.pt')
    shutil.copy(lf_checkpoint_filepath, target_filepath)

    hw_checkpoint_filepath = os.path.join(config['pretraining']['snapshot_path'], 'hw_checkpoint.pt')
    hw_checkpoint = safe_load.load_checkpoint(hw_checkpoint_filepath)
    hw = cnn_lstm.create_model(config['network']['hw']).to(device)
    hw.load_state_dict(hw_checkpoint['model_state_dict'])
    hw_filepath = os.path.join(best_overall_dir, 'hw.pt')
    torch.save(hw.state_dict(), hw_filepath)
    target_filepath = os.path.join(best_validation_dir, 'hw_checkpoint.pt')
    shutil.copy(hw_checkpoint_filepath, target_filepath)
