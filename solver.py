import logging 
import os
import shutil
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Solver(object):
    def __init__(self, data, model, loss, optimizer, early_stopper, args):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.early_stopper = early_stopper
        self.loss = loss 
        self.clip = args.clip
        self.logger = logging.getLogger('main_logger')
        self.args = args
        
        # save and load
        self.epoch_sts = args.epoch_start_to_save
        self.save_folder = os.path.join(args.save_folder, args.name)
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        
        # logging
        self.log_freq = args.log_freq
        self.tboard = args.tboard
        if args.tboard:
            self.writer = SummaryWriter(comment='_'+str(args.name))
        self._reset()

    def _reset(self):
        if self.continue_from:
            self.logger.info(f'Loading checkpoint model {self.continue_from}')
            cont = torch.load(self.continue_from)
            self.start_epoch = cont['epoch']
            self.model.load_state_dict(cont['model_state_dict'])
            self.optimizer.load_state_dict(cont['optimizer_state'])
            self.scheduler.load_state_dict(cont['scheduler_state'])
            torch.set_rng_state(cont['trandom_state'])
            np.random.set_state(cont['nrandom_state'])
        else:
            self.start_epoch = 0
        if os.path.isdir(self.save_folder):
            self.logger.info(f"Save folder already exist")
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")

    def train(self):
        for epoch in range(self.start_epoch, self.args.max_epochs):
            self.model.train()
            start = time.time()
            train_loss, lr = self._run_one_epoch(epoch)
            val_loss, _continue = self._run_validation(epoch)
            #TODO: #self.scheduler.step(val_loss)
            if self.tboard:
                pass #TODO: complete this
            if not _continue:
                break
            if self.checkpoint and epoch>self.epoch_sts: # Save model each epoch 
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state(),
                    'scheduler_state' : self.scheduler.state_dict()}, file_path)
                self.logger.info(f'Saving checkpoint model to {file_path}')

            # Save the best model
            if val_loss < self.best_val_loss and epoch>self.epoch_sts:
                self.best_val_loss = val_loss
                best_file_path = os.path.join(self.save_folder, 'temp_best.pth.tar')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, best_file_path)
                self.logger.info(f'Find better validated model, saving to {best_file_path}')

    def _run_one_epoch(self, epoch, validation=False):
        start = time.time()
        total_loss = 0
        dataloader = self.data['train_loader'] if not validation else self.data['val_loader']
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            src, trg = data[0].to(self.args.device), data[1].to(self.args.device)
            src_len, trg_len = data[2], data[3]
            self.optimizer.zero_grad()
            output = self.model(src, trg, src_len, teacher=self.args.teacher)
            loss = self.loss(output, trg, ignored_index=0)
            if not validation:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()
            total_loss += loss.item()
        avrg_loss = total_loss/(i+1)
        if validation:
            return avrg_loss
        else:
            return avrg_loss, self.optimizer.param_groups[0]['lr']

    def _run_validation(self, epoch):
        _continue=True
        self.model.eval() # turn off batchnorm & dropout
        with torch.no_grad():
            val_loss = self._run_one_epoch(epoch, validation=True)
        _continue = False if self.early_stopper.early_stop(val_loss) else True
        return val_loss, _continue
    
if __name__ == '__main__':
    print('Testing')
