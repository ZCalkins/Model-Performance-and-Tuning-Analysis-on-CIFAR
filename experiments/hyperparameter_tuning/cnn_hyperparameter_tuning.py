import torch
from torch import nn, optim
import pytorch_lightning as pl
import optuna
from models.cnn_model import CNNModel, CNNModelConfig, CNNLayerConfig
from utils.data_loading import get_dataset, get_dataloader

class LitCNNModel(pl.LightningModule):
    def __init__(self, config: CNNModelConfig):
        super().__init__()
        self.model = CNNModel(config)
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()

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

def create_cnn_config(trial):
  num_layers = trial.suggest_int('num_layers', 3, 12) # testing 3 to 12 layers

  layers = []
  in_channels = 3 # starting input channels for CIFAR100

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

    optimizer_class = trial.suggest_categorical('optimizer_class', ['Adam', 'SGD', 'RMSprop'])
    optimizer_params = {'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True)}
    if optimizer_class == 'SGD':
        optimizer_params['momentum'] = trial.suggest_float('momentum', 0.5, 0.9)
        optimizer_params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    cnn_config = CNNModelConfig(
        layers=layers,
        input_shape=(3, 32, 32),
        output_shape=100,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        batch_size=trial.suggest_int('batch_size', 32, 128, step=16),
        num_epochs=trial.suggest_int('num_epochs', 10, 50)
    )

    return cnn_config

def objective(trial):
  cnn_config = create_cnn_config(trial)

  train_dataset =  get_dataset(train=True)
