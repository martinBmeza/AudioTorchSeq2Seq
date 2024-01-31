import os
import torch
import logging
import torch.optim as optim
from criterion import mse_loss
from schedulers import WarmupReduceLROnPlateauScheduler, EarlyStopper
from data import AudioDataLoader, AudioDataset
from utils import create_logger, read_config_file, init_weights, count_parameters
from model import Encoder, Decoder, Seq2Seq
from solver import Solver
from schedulers import EarlyStopper

def main(args):
    logger = logging.getLogger('main_logger')
    logger.debug('loading dataset...')
    train_dataset = AudioDataset(**args.train_dataset)
    val_dataset = AudioDataset(**args.val_dataset)
    train_loader = AudioDataLoader(train_dataset, **args.dataloader)
    val_loader = AudioDataLoader(val_dataset, **args.dataloader)
    data = {'train_loader': train_loader, 'val_loader': val_loader,}

    # model
    model = Seq2Seq(args.encoder_params, args.decoder_params, args.device).to(args.device)
    
    # Weight Initialization
    model.apply(init_weights)
    count_parameters(model)
    early_stopper = EarlyStopper(logger, patience=args.early_stop_patience) if args.early_stop_patience else None
    solver_params =  {'data' : data,
            'model' : model,
            'loss' : mse_loss,
            'optimizer' : optim.Adam(model.parameters()),
            'early_stopper' : early_stopper,
            'args' : args}
    solver = Solver(**solver_params)
    solver.train()

if __name__ == '__main__':
    create_logger()
    args = read_config_file()
    main(args)
