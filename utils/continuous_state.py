import os

import torch

from hw import cnn_lstm
from lf.line_follower import LineFollower
from sol.start_of_line_finder import StartOfLineFinder
from utils import safe_load


def init_model(config, sol_dir='best_validation', lf_dir='best_validation', hw_dir='best_validation', only_load=None):
    base_0 = config['network']['sol']['base0']
    base_1 = config['network']['sol']['base1']

    sol = None
    lf = None
    hw = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if only_load is None or only_load == 'sol' or 'sol' in only_load:
        sol = StartOfLineFinder(base_0, base_1)
        sol_state = safe_load.torch_state(os.path.join(config['training']['snapshot'][sol_dir], "sol.pt"))
        sol.load_state_dict(sol_state)
        sol = sol.to(device)

    if only_load is None or only_load == 'lf' or 'lf' in only_load:
        lf = LineFollower(config['network']['hw']['input_height'])
        lf_state = safe_load.torch_state(os.path.join(config['training']['snapshot'][lf_dir], "lf.pt"))

        # special case for backward support of
        # previous way to save the LF weights
        if 'cnn' in lf_state:
            new_state = {}
            for k, v in iter(lf_state.items()):
                if k == 'cnn':
                    for k2, v2 in iter(v.items()):
                        new_state[k + "." + k2] = v2
                if k == 'position_linear':
                    for k2, v2 in iter(v.state_dict().items()):
                        new_state[k + "." + k2] = v2
                # if k == 'learned_window':
                #     new_state[k]=nn.Parameter(v.data)
            lf_state = new_state

        lf.load_state_dict(lf_state)
        lf = lf.to(device)

    if only_load is None or only_load == 'hw' or 'hw' in only_load:
        hw = cnn_lstm.create_model(config['network']['hw'])
        hw_state = safe_load.torch_state(os.path.join(config['training']['snapshot'][hw_dir], "hw.pt"))
        hw.load_state_dict(hw_state)
        hw = hw.to(device)

    return sol, lf, hw
