import json
import os
import time

import torch


def torch_state(path):
    for i in range(10):
        try:
            state = torch.load(path, map_location=lambda storage, loc: storage)
            return state
        except Exception as e:
            print("Failed to load", i, path)
            print(e)
            time.sleep(i)
    print("Failed to load state")
    return


def json_state(path):
    for i in range(10):
        try:
            with open(path) as f:
                state = json.load(f)
            return state
        except Exception as e:
            print("Failed to load", i, path)
            print(e)
            time.sleep(i)
    print("Failed to load state")
    return None


def load_checkpoint(path):
    checkpoint = None
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu' if not torch.cuda.is_available() else None)
    else:
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            print(f'The path {dirname} to checkpoint is not exists. Creating one...')
            os.makedirs(dirname)
    return checkpoint
