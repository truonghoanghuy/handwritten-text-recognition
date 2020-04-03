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
    snapshot = None
    if os.path.exists(path):
        try:
            snapshot = torch.load(path)
        except Exception as e:
            print(f'{path} exists but failed to load due to\n\n{e}\n\nThe training process will overwrite it.')
    elif not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return snapshot
