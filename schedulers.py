import os
import shutil
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter


class EarlyStopper:
    def __init__(self, logger, patience=1, min_delta=0):
        self.logger = logger
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.logger.info('Early Stopping Training!!!')
                return True
        return False


class LearningRateScheduler(_LRScheduler):
    r"""
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class WarmupReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Warmup learning rate until `warmup_steps` and reduce learning rate on plateau after.

    Args:
        optimizer (Optimizer): wrapped optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
    """
    def __init__(self, optimizer, peak_lr, final_lr, warmup_steps, patience, cooldown, factor):
        warmup_rate = peak_lr / warmup_steps
        super(WarmupReduceLROnPlateauScheduler, self).__init__(optimizer, warmup_rate)
        self.warmup_steps = warmup_steps
        self.update_steps = 0
        self.schedulers = [
            WarmupLRScheduler(optimizer=optimizer, peak_lr=peak_lr, warmup_steps=warmup_steps),
            ReduceLROnPlateauScheduler(optimizer=optimizer, lr=peak_lr, lr_final=final_lr, patience=patience, factor=factor, cooldown=cooldown),
        ]
    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps
        else:
            return 1, None

    def step(self, val_loss=None):
        stage, steps_in_stage = self._decide_stage()
        if stage == 0:
            self.schedulers[0].step()
        elif stage == 1:
            self.schedulers[1].step(val_loss)
        self.update_steps += 1
        return self.get_lr()

    def state_dict(self):
        """
        Returns a list of dicts, for this class
        and each child schedulers
        Contains entry for every variable which
        is no the optimizer or the schedulers
        """
        this_state = {key:value for key, value in self.__dict__.items() if key not in ['optimizer', 'schedulers']}
        warmup_state = {key:value for key, value in self.schedulers[0].__dict__.items() if key!='optimizer'}
        reduceLR_state = {key:value for key, value in self.schedulers[1].__dict__.items() if key!='optimizer'}
        return [this_state, warmup_state, reduceLR_state]

    def load_state_dict(self, states):
        self.__dict__.update(states[0])
        self.schedulers[0].__dict__.update(states[1])
        self.schedulers[1].__dict__.update(states[2])


class ReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by
    a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement is seen
    for a ‘patience’ number of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Optimizer.
        lr (float): Initial learning rate.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
    """
    def __init__(self, optimizer, lr, lr_final, patience=1, factor=0.3, cooldown=0):
        super(ReduceLROnPlateauScheduler, self).__init__(optimizer, lr)
        self.lr = lr
        self.lr_final = lr_final
        self.patience = patience
        self.factor = factor
        self.cooldown = cooldown
        self.cooldown_count = 0
        self.val_loss = 100.0
        self.count = 0

    def step(self, val_loss):
        if self.lr <= self.lr_final:
            return self.lr
        if val_loss is not None:
            if (self.val_loss <= val_loss) and (self.cooldown_count==0):
                self.count += 1
                #self.val_loss = val_loss
                print(f'Loss doesnt improve for {self.count} time/s')
            elif (self.val_loss <= val_loss) and (self.cooldown!=0):
                print('Loss doesnt improve but were in cooldown')
            else:
                self.count = 0
                self.val_loss = val_loss
                print(f'La loss es igual o mejor)')

            self.cooldown_count = self.cooldown_count-1 if self.cooldown_count>0 else 0
            if self.patience == self.count:
                self.count = 0
                self.cooldown_count = self.cooldown
                old_lr = self.lr
                self.lr *= self.factor
                new_lr = self.lr
                self.set_lr(self.optimizer, self.lr)
                print(f"Changing LR from {old_lr} to {new_lr}")
        return self.lr


class WarmupLRScheduler(LearningRateScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): wrapped optimizer.
    """
    def __init__(self, optimizer, peak_lr, warmup_steps):
        warmup_rate = peak_lr / warmup_steps
        super(WarmupLRScheduler, self).__init__(optimizer, warmup_rate)
        self.warmup_rate = warmup_rate
        self.update_steps = 1
        self.warmup_steps = warmup_steps
        self.init_lr = warmup_rate

    def step(self, val_loss=None):
        if self.update_steps < self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
            self.set_lr(self.optimizer, lr)
            self.lr = lr
        self.update_steps += 1
        return self.lr


