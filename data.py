import os
import torch
import random
import torch.utils.data as data
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class AudioDataset(data.Dataset):
    def __init__(self, n_batches, durations, freqs, noise, fs):
        """
        durations : (min_duration, max_duration)
        noise : Normal(0,1)*noise + signal_norm*(1-noise)
        """
        super(AudioDataset, self).__init__()
        batches = []
        for i in range(n_batches):
            duration = random.uniform(*durations)
            freq = random.uniform(*freqs)
            phi = random.uniform(0, 2*np.pi)
            t = np.linspace(0, duration, int(duration*fs))
            signal_clean = np.sin((2*np.pi*freq*t)+phi).astype(np.float32)
            signal_noise = (np.random.normal(0,1,signal_clean.size).astype(np.float32))*noise
            signal_noise = signal_noise + (signal_clean*(1-noise))
            batches.append((signal_clean, signal_noise))
        self.batches = batches

    def __getitem__(self, index):
        return self.batches[index]

    def __len__(self):
        return len(self.batches)

class AudioDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def _collate_fn(batches):
    xx, yy = zip(*batches)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_tensor = [torch.from_numpy(x).float() for x in xx]
    yy_tensor = [torch.from_numpy(y).float() for y in yy]
    xx_pad = pad_sequence(xx_tensor, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy_tensor, batch_first=True, padding_value=0)
    return xx_pad[:, :, None], yy_pad[:, :, None], x_lens, y_lens #[B, L, C] batch, lens, channels

if __name__ == '__main__':
    dataset = AudioDataset(100, (2,3), (100,500), 0.1)
    dataloader = AudioDataLoader(dataset, batch_size=4)
    for batch in dataloader:
        src, trg, src_len, trg_len = batch
        import pdb; pdb.set_trace()
