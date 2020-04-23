import sys
import time
from typing import *

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from utils import safe_load
from utils.continuous_training_util import ContinuousTrainingUtil
from utils.dataset_wrapper import DatasetWrapper
from utils.printer import ProgressBarPrinter


class ModuleTrainer:
    """Trainer class for torch.nn.Module instances"""

    def __init__(self, model: Module, optimizer: Optimizer,
                 train_loss_func: Callable[[Module, Any], Tensor],
                 evaluate_loss_func: Callable[[Module, Any], Tensor],
                 train_dataloader: DatasetWrapper, eval_dataloader: DataLoader,
                 checkpoint_filepath: str, loss_patience: float = np.inf):
        """train_loss_func, evaluate_loss_func: (model, input) -> batch loss"""

        self.model = model
        self.optimizer = optimizer
        self.train_loss_func = train_loss_func
        self.evaluate_loss_func = evaluate_loss_func
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.checkpoint_filepath = checkpoint_filepath
        self.loss_patience = loss_patience

    def train(self, resume=False, continuous_training=False,
              max_iter=1000, stop_after_no_improvement=10):
        """
        Start the train process

        :param resume: reuse the loss, epoch count, train time read from checkpoint if exists
        :param continuous_training: true if the train process is continuous
        :param max_iter: maximum number of the (train, evaluate) steps
        :param stop_after_no_improvement: early stopping. only if `continuous_flag_filepath` is None
        :return: None
        """

        sys.stdout.flush()
        lowest_loss = np.inf
        total_time = 0.0
        no_improvement_count = 0
        print(f'Training set size: {self.train_dataloader.epoch_steps} batch(es) x {self.train_dataloader.batch_size} '
              f'sample(s)/batch')

        # Load checkpoint to continue training if exists
        checkpoint = safe_load.load_checkpoint(self.checkpoint_filepath)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if resume:  # reuse loss
                self.train_dataloader.epoch = checkpoint['epoch']
                lowest_loss = checkpoint['loss']
                total_time = checkpoint['total_time']
            print('Continue training')

        for view_epoch in range(max_iter):
            start_time = time.time()
            print(f'\nView-epoch = {view_epoch}. Epoch = {self.train_dataloader.epoch}.')
            for phase in ['train', 'eval']:
                if phase == 'train':
                    self.model.train()
                    print(f'Training on {len(self.train_dataloader)} batch(es):')
                else:
                    self.model.eval()
                    print(f'Evaluating on {len(self.eval_dataloader)} batch(es):')
                sum_loss = 0.0
                steps = 0
                with torch.set_grad_enabled(phase == 'train'):
                    loader = self.train_dataloader if phase == 'train' else self.eval_dataloader
                    progress_printer = ProgressBarPrinter(len(loader))
                    progress_printer.start()
                    for input in loader:
                        try:
                            if phase == 'train':
                                loss = self.train_loss_func(self.model, input)
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()
                            else:
                                loss = self.evaluate_loss_func(self.model, input)
                        except Exception as e:
                            if not isinstance(e, UserWarning):
                                print(f'\n{repr(e)}')
                            progress_printer.step(skip=True)
                            continue
                        sum_loss += float(loss)
                        steps += 1
                        progress_printer.step()
                avg_loss = sum_loss / steps
                if phase == 'train':
                    print(f'Train loss = {avg_loss}')
                else:
                    phase_time = time.time() - start_time
                    total_time = total_time + phase_time
                    print(f'Eval loss = {avg_loss}, Current best loss = {lowest_loss}')
                    print(f'Time elapsed: {total_time}, Last epoch time: {phase_time}')
                    no_improvement_count += 1
                    if avg_loss < lowest_loss:
                        no_improvement_count = 0
                        lowest_loss = avg_loss
                        print('Better loss. Saving...')
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': self.train_dataloader.epoch,
                            'loss': lowest_loss,
                            'total_time': total_time
                        }, self.checkpoint_filepath)
                    if continuous_training:
                        if not ContinuousTrainingUtil.is_running():
                            return
                    else:
                        if no_improvement_count >= stop_after_no_improvement and lowest_loss < self.loss_patience:
                            return
