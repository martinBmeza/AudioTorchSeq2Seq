import argparse
import logging
import logging.config
import sys, os
import re
from importlib.machinery import SourceFileLoader
from types import ModuleType
from torch import nn

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_param:,} trainable parameters")

def read_config_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file with the experiment configurations")
    args = parser.parse_args()
    config_name = re.match(r"configs/(augmented/)?([\w\_\-]+).py",args.config).group(2)
    config = import_configs_objs(args.config)
    config.name = config_name
    return config

def import_configs_objs(config_file):
    """Dynamicaly loads the configuration file"""
    if config_file is None:
        raise ValueError("No config path")
    loader = SourceFileLoader('config', config_file)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)
    for var in ["__name__", "__doc__", "__package__", "__loader__", "__spec__", "__builtins__"]:
        delattr(mod, var)
    return mod

def create_logger():
    logging.config.fileConfig('logging.conf', disable_existing_loggers=True)
    logger = logging.getLogger('main_logger')
    return 


