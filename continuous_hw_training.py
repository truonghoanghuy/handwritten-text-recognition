import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

from hw import cnn_lstm
from hw import hw_dataset
from hw.hw_dataset import HwDataset
from utils import safe_load
from utils import string_utils, error_rates
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper


def training_step(config):
    hw_network_config = config['network']['hw']
    train_config = config['training']

    allowed_training_time = train_config['hw']['reset_interval']
    init_training_time = time.time()

    char_set_path = hw_network_config['char_set_path']

    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k, v in iter(char_set['idx_to_char'].items()):
        idx_to_char[int(k)] = v

    training_set_list = load_file_list(train_config['training_set'])
    train_dataset = HwDataset(training_set_list,
                              char_set['char_to_idx'], augmentation=True,
                              img_height=hw_network_config['input_height'])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config['hw']['batch_size'],
                                  shuffle=False, num_workers=0,
                                  collate_fn=hw_dataset.collate)

    batches_per_epoch = int(train_config['hw']['images_per_epoch'] / train_config['hw']['batch_size'])
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

    test_set_list = load_file_list(train_config['validation_set'])
    test_dataset = HwDataset(test_set_list,
                             char_set['char_to_idx'],
                             img_height=hw_network_config['input_height'],
                             random_subset_size=train_config['hw']['validation_subset_size'])

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=train_config['hw']['batch_size'],
                                 shuffle=False, num_workers=0,
                                 collate_fn=hw_dataset.collate)

    d_type = torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hw = cnn_lstm.create_model(hw_network_config)
    hw_path = os.path.join(train_config['snapshot']['best_validation'], "hw.pt")
    hw_state = safe_load.torch_state(hw_path)
    hw.load_state_dict(hw_state)
    hw = hw.to(device)
    optimizer = torch.optim.Adam(hw.parameters(), lr=train_config['hw']['learning_rate'])
    lowest_loss = np.inf
    criterion = CTCLoss().to(device)
    for epoch in range(10000000000):
        sum_loss = 0.0
        steps = 0.0
        hw.eval()
        with torch.no_grad():
            for x in test_dataloader:
                sys.stdout.flush()
                line_imgs: torch.Tensor = x['line_imgs']
                line_imgs = line_imgs.to(device, d_type)

                preds = hw(line_imgs).cpu()

                output_batch = preds.permute(1, 0, 2)
                out = output_batch.data.cpu().numpy()

                for i, gt_line in enumerate(x['gt']):
                    logits = out[i, ...]
                    pred, raw_pred = string_utils.naive_decode(logits)
                    pred_str = string_utils.label2str_single(pred, idx_to_char, False)
                    cer = error_rates.cer(gt_line, pred_str)
                    sum_loss += cer
                    steps += 1

        if epoch == 0:
            print("First Validation Step Complete")
            print("Benchmark Validation CER:", sum_loss / steps)
            lowest_loss = sum_loss / steps

            hw = cnn_lstm.create_model(hw_network_config)
            hw_path = os.path.join(train_config['snapshot']['current'], "hw.pt")
            hw_state = safe_load.torch_state(hw_path)
            hw.load_state_dict(hw_state)
            hw = hw.to(device)

            optim_path = os.path.join(train_config['snapshot']['current'], "hw_optim.pt")
            if os.path.exists(optim_path):
                print("Loading Optim Settings")
                optimizer.load_state_dict(safe_load.torch_state(optim_path))
            else:
                print("Failed to load Optim Settings")

        if lowest_loss > sum_loss / steps:
            lowest_loss = sum_loss / steps
            print("Saving Best")

            dirname = train_config['snapshot']['best_validation']
            if not len(dirname) != 0 and os.path.exists(dirname):
                os.makedirs(dirname)

            save_path = os.path.join(dirname, "hw.pt")

            torch.save(hw.state_dict(), save_path)

        print("Test Loss", sum_loss / steps, lowest_loss)
        print("")

        if allowed_training_time < (time.time() - init_training_time):
            print("Out of time: Exiting...")
            break

        print("Epoch", epoch)
        sum_loss = 0.0
        steps = 0.0
        hw.train()
        for x in train_dataloader:

            line_imgs: torch.Tensor = x['line_imgs']
            line_imgs = line_imgs.type(d_type)
            labels: torch.Tensor = x['labels']
            label_lengths: torch.Tensor = x['label_lengths']

            preds = hw(line_imgs).cpu()

            output_batch = preds.permute(1, 0, 2)
            out = output_batch.data.cpu().numpy()

            # if i == 0:
            #     for i in range(out.shape[0]):
            #         pred, pred_raw = string_utils.naive_decode(out[i,...])
            #         pred_str = string_utils.label2str_single(pred_raw, idx_to_char, True)
            #         print pred_str

            for i, gt_line in enumerate(x['gt']):
                logits = out[i, ...]
                pred, raw_pred = string_utils.naive_decode(logits)
                pred_str = string_utils.label2str_single(pred, idx_to_char, False)
                cer = error_rates.cer(gt_line, pred_str)
                sum_loss += cer
                steps += 1

            batch_size = preds.size(1)
            preds_size = torch.tensor([preds.size(0)] * batch_size).type(torch.int32)

            loss = criterion(preds, labels, preds_size, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Train Loss", sum_loss / steps)
        print("Real Epoch", train_dataloader.epoch)

    # Save current snapshots for next iteration
    print("Saving Current")
    dirname = train_config['snapshot']['current']
    if not len(dirname) != 0 and os.path.exists(dirname):
        os.makedirs(dirname)

    save_path = os.path.join(dirname, "hw.pt")
    torch.save(hw.state_dict(), save_path)

    optim_path = os.path.join(dirname, "hw_optim.pt")
    torch.save(optimizer.state_dict(), optim_path)


if __name__ == "__main__":
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = yaml.load(f)

    cnt = 0
    while True:
        print("")
        print("Full Step", cnt)
        print("")
        cnt += 1
        training_step(config)
