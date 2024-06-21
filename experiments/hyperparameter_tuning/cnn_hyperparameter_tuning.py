import os
import yaml
import torch
import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils.data_loading import get_dataset, get_dataloader, create_transform
from models.cnn_model import CNNModel, CNNModelConfig
import torchmetrics
from pytorch_lightning.profiler import SimpleProfiler

# Load the experiment configuration
config_file_path = 'configurations/yaml/cnn_hyperparameter_tuning.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Set up general configurations
device = torch.device(config['general']['device'])
seed = config['general']['seed']
num_workers = config['general']['num_workers']
deterministic = config['misc']['deterministic']

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
    handlers.append(logging.FileHandler(log_file))
if config['logging']['log_to_console']:
    handlers.append(logging.StreamHandler())

logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

logger = logging.getLogger('experiment_logger')
logger.info("Logging configuration set up.")

# Set debug mode if enabled
if config['misc']['debug']:
    pl.seed_everything(seed, workers=True)
    num_epochs = config['misc']['num_epochs_debug']
    profiler = SimpleProfiler() if config['misc']['profiler_enabled'] else None
else:
    num_epochs = config['hyperparameter_optimization']['num_epochs']
    profiler = None

# Create data transform
transform = create_transform(transform_type='standard', size=224, normalize=True, flatten=False)

class LitCNNModel(pl.LightningModule):
    def __init__(self, config: CNNModelConfig):
        super().__init__()
        self.model = CNNModel(config)
        self.config = config
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Initialize metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_precision = torchmetrics.Precision(num_classes=config.output_shape)
        self.val_recall = torchmetrics.Recall(num_classes=config.output_shape)
        self.val_f1 = torchmetrics.F1(num_classes=config.output_shape)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)

        # Log accuracy
        self.train_accuracy(logits, y)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True)

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
    def __init__(self, batch_size, num_workers, transform):
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

        if self.use_smaller_subset:
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

    # Set up logging
    loggers = []
    if config['monitoring']['tensorboard']:
        loggers.append(TensorBoardLogger(log_dir, name="cnn_model", version=f"trial_{trial.number}"))

    early_stopping = EarlyStopping(monitor=config['early_stopping']['monitor'], patience=config['early_stopping']['patience'])
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=config['checkpointing']['monitor_metric'],
        save_top_k=1,
        mode='min'
    )

    trainer = pl.Trainer(
        logger=loggers,
        max_epochs=num_epochs,
        gpus=1 if device.type == 'cuda' else 0,
        precision=16 if config['misc']['use_mixed_precision'] else 32,
        deterministic=config['misc']['deterministic'],
        profiler=profiler,
        callbacks=[early_stopping, checkpoint_callback]
    )

    trainer.fit(model, datamodule=data_module)
    val_result = trainer.validate(model, datamodule=data_module)
    val_loss = val_result[0]['val_loss']

    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction=config['hyperparameter_optimization']['direction'])
    study.optimize(objective, n_trials=config['hyperparameter_optimization']['n_trials'])

    print(f'Best trial: {study.best_trial.value}')
    print(f'Best hyperparameters: {study.best_trial.params}')
