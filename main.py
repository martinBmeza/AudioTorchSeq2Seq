import torch
import torch.optim as optim
from utils import init_weights, count_parameters
from model import Encoder, Decoder, Seq2Seq
from data import AudioDataset, AudioDataLoader
from criterion import mse_loss

# TODO
#   Add init token in bot src and trg
model = Seq2Seq(encoder_params, decoder_params, device).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters())

# Loss
loss = mse_loss

# Data
dataset = AudioDataset(n_batches=4, durations=(3,5), freqs=(3,5), noise=0.0, fs=1)
dataloader = AudioDataLoader(dataset, batch_size=2, num_workers)


model.train()
epoch_loss = 0
clip = 1.0
for i, batch in enumerate(dataloader):
    
