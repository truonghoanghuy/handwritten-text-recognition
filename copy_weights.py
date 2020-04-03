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
    base0 = config['network']['sol']['base0']
    base1 = config['network']['sol']['base1']
    sol_checkpoint_filepath = os.path.join(config['pretraining']['snapshot_path'], 'sol_checkpoint.pt')
    lf_checkpoint_filepath = os.path.join(config['pretraining']['snapshot_path'], 'lf_checkpoint.pt')
    hw_checkpoint_filepath = os.path.join(config['pretraining']['snapshot_path'], 'hw_checkpoint.pt')

    sol_filepath = os.path.join(config['pretraining']['snapshot_path'], 'sol.pt')
    lf_filepath = os.path.join(config['pretraining']['snapshot_path'], 'lf.pt')
    hw_filepath = os.path.join(config['pretraining']['snapshot_path'], 'hw.pt')

    sol = StartOfLineFinder(base0, base1).to(device)
    lf = LineFollower().to(device)
    hw = cnn_lstm.create_model(config['network']['hw']).to(device)

    sol_checkpoint = safe_load.load_checkpoint(sol_checkpoint_filepath)
    lf_checkpoint = safe_load.load_checkpoint(lf_checkpoint_filepath)
    hw_checkpoint = safe_load.load_checkpoint(hw_checkpoint_filepath)

    sol.load_state_dict(sol_checkpoint['model_state_dict'])
    lf.load_state_dict(lf_checkpoint['model_state_dict'])
    hw.load_state_dict(hw_checkpoint['model_state_dict'])

    torch.save(sol.state_dict(), sol_filepath)
    torch.save(lf.state_dict(), lf_filepath)
    torch.save(hw.state_dict(), hw_filepath)

    for target in ('best_overall', 'best_validation', 'current'):
        dirname = config['training']['snapshot'][target]
        target_filepath = os.path.join(dirname, 'sol.pt')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copy2(sol_filepath, target_filepath)

    for target in ('best_overall', 'best_validation', 'current'):
        dirname = config['training']['snapshot'][target]
        target_filepath = os.path.join(dirname, 'lf.pt')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copy2(lf_filepath, target_filepath)

    for target in ('best_overall', 'best_validation', 'current'):
        dirname = config['training']['snapshot'][target]
        target_filepath = os.path.join(dirname, 'hw.pt')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copy2(hw_filepath, target_filepath)
