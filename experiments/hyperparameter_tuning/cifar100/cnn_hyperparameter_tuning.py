import os
import sys
import random
import logging

import yaml
import torch
import torchvision
import numpy as np
import optuna
from optuna.exceptions import TrialPruned
from optuna.samplers import TPESampler
import pytorch_lightning as pl
from torch.utils.data import Subset
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler
from torchvision import transforms
from torchvision.transforms import v2

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(project_root)

# Load the experiment configuration
config_file_path = os.path.join(project_root, 'configurations', 'yaml', 'hyperparameter_tuning', 'cifar100', 'cnn.yaml')
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Convert necessary relative paths to absolute paths
experiment_type = 'hyperparameter_tuning'
dataset = 'cifar100'
model_type = 'cnn'

config['experiment']['log_dir'] = os.path.join(project_root, 'logs', 'experiment_logs', experiment_type, dataset, model_type)
config['experiment']['checkpoints_dir'] = os.path.join(project_root, 'checkpoints', experiment_type, dataset, model_type)
config['experiment']['save_dir'] = os.path.join(project_root, 'results', experiment_type, dataset, model_type)
config['experiment']['tensorboard_log_dir'] = os.path.join(project_root, 'logs', 'tensorboard', experiment_type, dataset, model_type)

from utils.data_loading import get_dataset, get_dataloader
from models.cnn_model import CNNModel, CNNModelConfig, CNNLayerConfig

# Set up general configurations
seed = config['general']['seed']
num_workers = config['general']['num_workers']
deterministic = config['misc']['deterministic']
use_smaller_dataset = config['misc']['use_smaller_dataset']
num_epochs_debug = config['misc']['num_epochs_debug']
profiler_enabled = config['misc']['profiler_enabled']

# Lower precision to enhance performance
torch.set_float32_matmul_precision("high")

# Set random seed for reproducibility
pl.seed_everything(seed)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Manual configuration for deterministic behavior
if deterministic:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set up logging
log_level = getattr(logging, config['logging']['level'].upper(), logging.INFO)
log_format = config['logging']['format']

handlers = []
if config['logging']['log_to_file']:
    log_file = os.path.join(config['experiment']['log_dir'], 'experiment.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file))
if config['logging']['log_to_console']:
    handlers.append(logging.StreamHandler())

logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

logger = logging.getLogger('experiment_logger')
logger.info("Logging configurations have been set.")

# Set debug mode if enabled
if config['misc']['debug']:
    pl.seed_everything(seed, workers=True)
    num_epochs = num_epochs_debug
    profiler = SimpleProfiler() if profiler_enabled else None
else:
    num_epochs = config['hyperparameter_optimization']['n_trials']
    profiler = None

class LitCNNModel(pl.LightningModule):
    def __init__(self, config: CNNModelConfig):
        super().__init__()
        self.model = CNNModel(config)
        self.config = config
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Initialize metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=config.output_shape)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=config.output_shape)
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=config.output_shape)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=config.output_shape)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=config.output_shape)

    def forward(self, x):
        return self.model(x.to(device))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)

        # Log accuracy
        self.train_accuracy(logits, y)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss, prog_bar=True)

        # Logging accuracy, precision, and recall
        self.val_accuracy(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)
        self.val_f1(logits, y)
        self.log('val_acc', self.val_accuracy, prog_bar=True, on_epoch=True)
        self.log('val_precision', self.val_precision, prog_bar=True, on_epoch=True)
        self.log('val_recall', self.val_recall, prog_bar=True, on_epoch=True)
        self.log('val_f1', self.val_f1, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        if self.config.optimizer_class == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.config.optimizer_params)
        elif self.config.optimizer_class == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), **self.config.optimizer_params)
        return optimizer

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size,
                 num_workers,
                 transform_type='standard',
                 size=224,
                 normalize=True,
                 flatten=False,
                 use_smaller_dataset=False):
        super().__init__()
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.transform_type=transform_type
        self.size=size
        self.normalize=normalize
        self.flatten=flatten
        self.use_smaller_dataset=use_smaller_dataset

    def prepare_data(self):
        get_dataset(name='CIFAR100',
                    train=True,
                    transform_type=self.transform_type,
                    size=self.size,
                    normalize=self.normalize,
                    flatten=self.flatten)
        get_dataset(name='CIFAR100',
                    train=False,
                    transform_type=self.transform_type,
                    size=self.size,
                    normalize=self.normalize,
                    flatten=self.flatten)

    def setup(self, stage=None):
        train_dataset = get_dataset(name='CIFAR100',
                                    train=True,
                                    transform_type=self.transform_type,
                                    size=self.size,
                                    normalize=self.normalize,
                                    flatten=self.flatten)
        val_dataset = get_dataset(name='CIFAR100',
                                  train=False,
                                  transform_type=self.transform_type,
                                  size=self.size,
                                  normalize=self.normalize,
                                  flatten=self.flatten)

        if self.use_smaller_dataset:
            train_dataset = Subset(train_dataset, range(len(train_dataset) // 10))
            val_dataset = Subset(val_dataset, range(len(val_dataset) // 10))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return get_dataloader(self.train_dataset,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True)

    def val_dataloader(self):
        return get_dataloader(self.val_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers,
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True)

    def test_dataloader(self):
        return get_dataloader(self.val_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers,
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True)

def initialize_model(model, dummy_input):
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    model.train()

def create_cnn_config(trial):
    try:
        num_layers = trial.suggest_int('num_layers', 6, 12)
        layers = []
        in_channels = 3

        use_strided_conv = num_layers > 8
        default_to_pooling = not use_strided_conv
    
        for i in range(num_layers):
            out_channels = trial.suggest_int(f'out_channels_{i}', 32, 256, step=16)
            
            if i < num_layers // 4:
                kernel_size = 7
            elif i < num_layers // 2:
                kernel_size = 5
            else:
                kernel_size = 3
                
            stride = trial.suggest_int(f'stride_{i}', 1, 2) if use_strided_conv else 1
            padding = (kernel_size - 1) // 2
            use_batch_norm = trial.suggest_categorical(f'use_batch_norm_{i}', [True, False])
            use_dropout = trial.suggest_categorical(f'use_dropout_{i}', [True, False])
            dropout_rate = trial.suggest_float(f'dropout_rate_{i}', 0.1, 0.5) if use_dropout else 0.0
            activation = trial.suggest_categorical(f'activation_{i}', ['ReLU', 'LeakyReLU', 'SiLU'])

            if default_to_pooling and i % 3 == 2:
                use_pool = True
                pool_type = trial.suggest_categorical(f'pool_type_{i}', ['MaxPool2d', 'AvgPool2d'])
                pool_size = trial.suggest_int(f'pool_size_{i}', 2, 3)
                pool_stride = trial.suggest_int(f'pool_stride_{i}', 2, 3)
            else:
                use_pool = False
                pool_type = None
                pool_size = None
                pool_stride = None
    
            layer_config = CNNLayerConfig(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_batch_norm=use_batch_norm,
                use_pool=use_pool,
                pool_size=pool_size,
                pool_stride=pool_stride,
                pool_type=pool_type,
                use_dropout=use_dropout,
                dropout_rate=dropout_rate,
                activation=activation
            )
            layers.append(layer_config)
            in_channels = out_channels
    
        optimizer_class = trial.suggest_categorical('optimizer_class', ['Adam', 'SGD'])
        optimizer_params = {'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True)}
        if optimizer_class == 'SGD':
            optimizer_params['momentum'] = trial.suggest_float('momentum', 0.5, 0.9)
            optimizer_params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
    
        cnn_config = CNNModelConfig(
            model_name=config['experiment']['name'],
            layers=layers,
            input_shape=(3, 224, 224),
            output_shape=100,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            batch_size=trial.suggest_int('batch_size', 32, 128, step=16),
            num_epochs=trial.suggest_int('num_epochs', 10, 30),
            label_smoothing=label_smoothing
        )
    
        return cnn_config

    except Exception as e:
        print(f"Pruning trial due to invalid configuration: {e}")
        raise optuna.exceptions.TrialPruned()

def objective(trial):
    # Mitigation for out of memory errors
    torch.cuda.empty_cache()
    
    cnn_config = create_cnn_config(trial)

    # Suggest image transform
    transform_type = trial.suggest_categorical('transform_type', ['standard', 'augmented'])
    
    data_module = CIFAR100DataModule(
        batch_size=cnn_config.batch_size,
        num_workers=num_workers,
        transform_type=transform_type,
        size=224,
        normalize=True,
        flatten=False,
        use_smaller_dataset=use_smaller_dataset
    )
    model = LitCNNModel(config=cnn_config).to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    initialize_model(model, dummy_input)

    # Set up logging
    loggers = []
    if config['monitoring']['tensorboard']:
        tensorboard_logger = TensorBoardLogger(config['experiment']['tensorboard_log_dir'], name="cnn_model_hpo", version=f"trial_{trial.number}")
        loggers.append(tensorboard_logger)

    early_stopping = EarlyStopping(
        monitor=config['early_stopping']['monitor'],
        patience=config['early_stopping']['patience'],
        min_delta=config['early_stopping']['min_delta']
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['experiment']['checkpoints_dir'],
        monitor=config['checkpointing']['monitor_metric'],
        save_top_k=1 if config['checkpointing']['save_best_only'] else config['checkpointing']['max_checkpoints'],
        mode='min'
    )

    trainer = pl.Trainer(
        logger=loggers,
        max_epochs=num_epochs,
        devices=torch.cuda.device_count(),
        accelerator=device,
        strategy='ddp',
        precision=16 if config['misc']['use_mixed_precision'] else 32,
        deterministic=config['misc']['deterministic'],
        profiler=profiler,
        callbacks=[early_stopping, checkpoint_callback]
    )

    ckpt_path = config['experiment'].get('resume_checkpoint', None)
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    val_result = trainer.validate(model, datamodule=data_module)
    val_loss = val_result[0]['val_loss']

    if tensorboard_logger:
        tensorboard_logger.log_hyperparams(trial.params, {'val_loss': val_loss})

    results_file = os.path.join(config['experiment']['save_dir'], f'results_trial_{trial.number}.yaml')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        yaml.dump({'trial': trial.number, 'params': trial.params, 'val_loss': val_loss}, f)
    
    return val_loss

if __name__ == "__main__":
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction=config['hyperparameter_optimization']['direction'], sampler=sampler)
    study.optimize(objective, n_trials=config['hyperparameter_optimization']['n_trials'])

    logger.info(f'Best trial: {study.best_trial.value}')
    logger.info(f'Best hyperparameters: {study.best_trial.params}')
