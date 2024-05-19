import pytorch_lightning as pl
import optuna
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data_loading import get_dataset, get_dataloader
from models.cnn_model import ViTModel, ViTModelConfig, TransformerEncoderConfig
from datetime import datetime
import os
import torch
import yaml

# Load the yaml experiment configurations
config_file_path = 'CIFAR100-Multi-Model-Ablation-Analysis/configurations/yaml/experiment_config.yaml'
with open(config_file_path, 'r') as file:
  config = yaml.safe_load(file)

# Set up general configurations
device = torch.device(config['general']['device'])
seed = config['general']['seed']
num_workers = config['general']['num_workers']

# Create transform
transform = create_transform(transform_type = 
