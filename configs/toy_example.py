import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Initialization
encoder_params = {
        'input_dim': 1,
        'hidden_dim': 512,
        'n_layers': 2,
        'dropout': 0.2
        }
decoder_params = {
        'input_dim': 1,
        'output_dim' : 1,
        'hidden_dim': 512,
        'n_layers':2,
        'dropout': 0.2
        }

# Data (toy)
train_dataset = {
        'n_batches': 1000,
        'durations': (1,2),
        'freqs': (5, 15), # Hz
        'noise': 0.2,
        'fs': 100
        }
val_dataset = {
        'n_batches': 100,
        'durations': (1,2), 
        'freqs': (5, 15), # Hz
        'noise': 0.2,
        'fs': 100
        }
dataloader = {
        'batch_size': 16,
        'num_workers': 10,
        }

# Training
max_epochs = 10
clip = 1.0
teacher = 0.1
ignored_index = 0

# Optimizer and Schedulers
scheduler = 'nada'
lr_init = 1e-3
lr_final = 1e-8
lr_warmup = 4 #epochs
lr_cooldown = 4 #epochs
lr_factor = 0.5
lr_patience = 6
early_stop_patience = 30

# Load and save
continue_from = None
checkpoint = True
save_folder = 'experiments/'
model_path = 'final.pth.tar'
epoch_start_to_save = 10

# Logging
log_freq = 10
tboard = True
DEV = False
logger_debug = True
