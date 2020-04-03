import sys
import time
from typing import Callable, Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import safe_load
from utils.dataset_wrapper import DatasetWrapper
from utils.printer import ProgressBarPrinter


def train(model: nn.Module,
          train_loss: Callable[[nn.Module, Any], torch.Tensor], evaluate_loss: Callable[[nn.Module, Any], torch.Tensor],
          optimizer: optim, train_dataloader: DatasetWrapper, eval_dataloader: DataLoader,
          checkpoint_filepath: str, stop_after_no_improvement: int, loss_patience=None):
    """train_loss, evaluate_loss: (model, input) -> loss"""

    sys.stdout.flush()
    lowest_loss = np.inf
    cnt_since_last_improvement = 0
    total_time = 0.0
    print(f'Training set size: {train_dataloader.epoch_steps} batch(es) x {train_dataloader.batch_size} '
          f'sample(s)/batch')

    # Load checkpoint to continue training if exists
    checkpoint = safe_load.load_checkpoint(checkpoint_filepath)
    if checkpoint is not None:
        train_dataloader.epoch = checkpoint['epoch']
        lowest_loss = checkpoint['loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_time = checkpoint['total_time']

    for sub_epoch in range(1000):
        start_time = time.time()
        print(f'Epoch {train_dataloader.epoch}. View-epoch = {sub_epoch}.')
        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
                print(f'Training on {len(train_dataloader)} batch(es):')
            else:
                model.eval()
                print(f'Evaluating on {len(eval_dataloader)} batch(es):')
            sum_loss = 0.0
            steps = 0
            with torch.set_grad_enabled(phase == 'train'):
                loader = train_dataloader if phase == 'train' else eval_dataloader
                progress_printer = ProgressBarPrinter(len(loader))
                progress_printer.start()
                for index, input in enumerate(loader):
                    try:
                        if phase == 'train':
                            loss = train_loss(model, input)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        else:
                            loss = evaluate_loss(model, input)
                    except Exception as e:
                        if not isinstance(e, UserWarning):
                            print('\n', e, flush=True)
                        progress_printer.step(skip=True)
                        continue
                    sum_loss += loss.item()
                    steps += loader.batch_size
                    progress_printer.step()
            if steps == 0:
                steps = loader.batch_size
                print('Possibly error: No data was loaded')
            avg_loss = sum_loss / steps
            if phase == 'train':
                current = time.time()
                total_time = total_time + current - start_time
                print(f'Train loss = {avg_loss}, Time elapsed: {total_time}, Last epoch time: {current - start_time}')
            else:
                print(f'Eval loss = {avg_loss}, Current best loss = {lowest_loss}')
                if lowest_loss > avg_loss:
                    cnt_since_last_improvement = 0
                    lowest_loss = avg_loss
                    print('Better loss. Saving...')
                    torch.save({
                        'epoch': train_dataloader.epoch,
                        'loss': lowest_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'total_time': total_time
                    }, checkpoint_filepath)
                cnt_since_last_improvement += 1
                if cnt_since_last_improvement >= stop_after_no_improvement:
                    if loss_patience is None or lowest_loss < loss_patience:  # python's lazy evaluation
                        return
