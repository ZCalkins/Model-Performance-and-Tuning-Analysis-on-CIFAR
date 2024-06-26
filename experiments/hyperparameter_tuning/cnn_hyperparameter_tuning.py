import os
import sys
import random
import logging

import yaml
import torch
import numpy as np
import optuna
import pytorch_lightning as pl
from torch.utils.data import Subset
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

# Load the experiment configuration
config_file_path = os.path.join(project_root, 'configurations', 'yaml', 'hyperparameter_tuning', 'cnn_hyperparameter_tuning.yaml')
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Convert necessary paths to absolute paths
config['experiment']['log_dir'] = os.path.join(project_root, config['experiment']['log_dir'])
config['experiment']['checkpoints_dir'] = os.path.join(project_root, config['experiment']['checkpoints_dir'])
config['experiment']['save_dir'] = os.path.join(project_root, config['experiment']['save_dir'])
config['experiment']['tensorboard_log_dir'] = os.path.join(project_root, config['experiment']['tensorboard_log_dir'])

from utils.data_loading import get_dataset, get_dataloader, create_transform
from models.cnn_model import CNNModel, CNNModelConfig, CNNLayerConfig

# Set up general configurations
seed = config['general']['seed']
num_workers = config['general']['num_workers']
deterministic = config['misc']['deterministic']
use_smaller_dataset = config['misc']['smaller_dataset']
num_epochs_debug = config['misc']['num_epochs_debug']
profiler_enabled = config['misc']['profiler_enabled']

# Set random seed for reproducibility
pl.seed_everything(seed)

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
logger.info("Logging configuration set up.")

# Set debug mode if enabled
if config['misc']['debug']:
    pl.seed_everything(seed, workers=True)
    num_epochs = num_epochs_debug
    profiler = SimpleProfiler() if profiler_enabled else None
else:
    num_epochs = config['hyperparameter_optimization']['n_trials']
    profiler = None

# Create data transform
transform = create_transform(transform_type='standard', size=224, normalize=True, flatten=False)

# Check if CUDA is available and set devices and accelerator accordingly
if torch.cuda.is_available():
    devices = 1
    accelerator = 'cuda'
else:
    devices = 1
    accelerator = 'cpu'

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
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)

        # Log accuracy
        self.train_accuracy(logits, y)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss, prog_bar=True)

        # Logging accuracy, precision, and recall
        self.val_accuracy(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)
        self.f1(logits, y)
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
    def __init__(self, batch_size, num_workers, transform, use_smaller_dataset):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.use_smaller_dataset = use_smaller_dataset

    def prepare_data(self):
        get_dataset(name='CIFAR100', train=True, transform_config=transform_config)
        get_dataset(name='CIFAR100', train=False, transform_config=transform_config)

    def setup(self, stage=None):
        train_dataset = get_dataset(name='CIFAR100', train=True, transform_config=transform_config)
        val_dataset = get_dataset(name='CIFAR100', train=False, transform_config=transform_config)

        if self.use_smaller_dataset:
            train_dataset = Subset(train_dataset, range(len(train_dataset) // 10))
            val_dataset = Subset(val_dataset, range(len(val_dataset) // 10))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return get_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return get_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return get_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def create_cnn_config(trial):
    num_layers = trial.suggest_int('num_layers', 6, 18)
    layers = []
    in_channels = 3

    for i in range(num_layers):
        out_channels = trial.suggest_int(f'out_channels_{i}', 16, 128, step=16)
        kernel_size = trial.suggest_int(f'kernel_size_{i}', 3, 7, step=2)
        stride = trial.suggest_int(f'stride_{i}', 1, 3)
        padding = trial.suggest_int(f'padding_{i}', 0, 3)
        use_batch_norm = trial.suggest_categorical(f'use_batch_norm_{i}', [True, False])
        use_pool = trial.suggest_categorical(f'use_pool_{i}', [True, False])
        pool_type = trial.suggest_categorical(f'pool_type_{i}', ['MaxPool2d', 'AvgPool2d'])
        pool_size = trial.suggest_int(f'pool_size_{i}', 2, 3)
        pool_stride = trial.suggest_int(f'pool_stride_{i}', 2, 3)
        use_dropout = trial.suggest_categorical(f'use_dropout_{i}', [True, False])
        dropout_rate = trial.suggest_float(f'dropout_rate_{i}', 0.1, 0.5)
        activation = trial.suggest_categorical(f'activation_{i}', ['ReLU', 'LeakyReLU', 'SiLU'])

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
        input_shape=(3, 32, 32),
        output_shape=100,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        batch_size=trial.suggest_int('batch_size', 32, 128, step=16),
        num_epochs=trial.suggest_int('num_epochs', 10, 50),
        label_smoothing=label_smoothing
    )

    return cnn_config

def objective(trial):
    cnn_config = create_cnn_config(trial)

    data_module = CIFAR100DataModule(
        batch_size=cnn_config.batch_size,
        num_workers=num_workers,
        transform=transform,
        use_smaller_dataset=use_smaller_dataset
    )
    model = LitCNNModel(config=cnn_config)

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
        devices=devices,
        accelerator=accelerator,
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
    study = optuna.create_study(direction=config['hyperparameter_optimization']['direction'])
    study.optimize(objective, n_trials=config['hyperparameter_optimization']['n_trials'])

    logger.info(f'Best trial: {study.best_trial.value}')
    logger.info(f'Best hyperparameters: {study.best_trial.params}')
