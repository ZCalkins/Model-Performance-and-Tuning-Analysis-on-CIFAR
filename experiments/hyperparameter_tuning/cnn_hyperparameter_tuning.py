import pytorch_lightning as pl
import optuna
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data_loading import get_dataset, get_dataloader
from models.cnn_model import CNNModel, CNNModelConfig
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
transform = create_transform(transform_type='standard', size=224, normalize=True, flatten=False)

# Set random seed for reproducibility
pl.seed_everything(seed)

class LitCNNModel(pl.LightningModule):
    def __init__(self, config: CNNModelConfig):
        super().__init__()
        self.model = CNNModel(config)
        self.config = config
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.config.optimizer_class == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.config.optimizer_params)
        elif self.config.optimizer_class == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), **self.config.optimizer_params)
        return optimizer

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, transform):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def prepare_data(self):
        get_dataset(name='CIFAR100', train=True, transform=self.transform)
        get_dataset(name='CIFAR100', train=False, transform=self.transform)

    def setup(self, stage=None):
        self.train_dataset = get_dataset(name='CIFAR100', train=True, transform=self.transform)
        self.val_dataset = get_dataset(name='CIFAR100', train=False, transform=self.transform)

    def train_dataloader(self):
        return get_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return get_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return get_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def create_cnn_config(trial):
    num_layers = trial.suggest_int('num_layers', 2, 5)
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

    data_module = CIFAR100DataModule(batch_size=cnn_config.batch_size, num_workers=num_workers, transform=transform)
    model = LitCNNModel(config=cnn_config)

    # Use trial.number to differentiate each trial
    logger = TensorBoardLogger("logs/tensorboard", name="cnn_model", version=f"trial_{trial.number}")
    early_stopping = EarlyStopping('val_loss', patience=5)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cnn_config.num_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[early_stopping, checkpoint_callback]
    )

    trainer.fit(model, datamodule=data_module)
    val_result = trainer.validate(model, datamodule=data_module)
    val_loss = val_result[0]['val_loss']

    # Log hyperparameters and their corresponding validation loss
    logger.log_hyperparams(trial.params, {'val_loss': val_loss})

    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print(f'Best trial: {study.best_trial.value}')
    print(f'Best hyperparameters: {study.best_trial.params}')
  train_dataset =  get_dataset(train=True)
