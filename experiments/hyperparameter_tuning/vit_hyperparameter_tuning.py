import pytorch_lightning as pl
import optuna
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data_loading import create_transform, get_dataset, get_dataloader
from models.vit_model import ViTModel, ViTModelConfig, TransformerEncoderConfig
from optuna.integration import PyTorchLightningPruningCallback
import torch
import yaml
import torchmetrics

# Load the experiment configuration
config_file_path = 'CIFAR100-Multi-Model-Ablation-Analysis/configurations/yaml/experiment_config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Set up general configurations
device = torch.device(config['general']['device'])
seed = config['general']['seed']
num_workers = config['general']['num_workers']

# Set the random seed for reproducibility
pl.seed_everything(seed)

# Create the transform for non-augmented data
transform = create_transform(transform_type='standard', size=224, normalize=True, flatten=False)

# Creating pytorch lightning module for the model
class LitViTModel(pl.LightningModule):
    def __init__(self, config: ViTModelConfig):
        super().__init__()
        self.model = ViTModel(config)
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.train_accuracy(y_pred, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.val_accuracy(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.config.optimizer_params)
        elif self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), **self.config.optimizer_params)
        return optimizer

# Creating data module using pytorch lightning
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

# Creating configurations for optuna study
def create_vit_config(trial):
    image_size = 224
    patch_size = trial.suggest_categorical('patch_size', [16, 32])
    num_channels = 3
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=64)
    num_layers = trial.suggest_int('num_layers', 6, 12)
    num_heads = trial.suggest_int('num_heads', 4, 8)
    head_dim = hidden_dim // num_heads
    mlp_dim = trial.suggest_int('mlp_dim', 128, 512, step=128)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    attention_dropout_rate = trial.suggest_float('attention_dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    num_epochs = trial.suggest_int('num_epochs', 10, 100)
    batch_size = trial.suggest_int('batch_size', 32, 128, step=16)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
    
    optimizer_class = trial.suggest_categorical('optimizer_class', ['adam', 'sgd'])
    optimizer_params = {'lr': learning_rate}
    if optimizer_class == 'sgd':
        optimizer_params['momentum'] = trial.suggest_float('momentum', 0.5, 0.9)
        optimizer_params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    encoder_configs = [
        TransformerEncoderConfig(
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate
        ) for _ in range(num_layers)
    ]

    vit_config = ViTModelConfig(
        model_name='vit',
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        encoder_configs=encoder_configs,
        num_classes=100,
        optimizer=optimizer_class,
        optimizer_params=optimizer_params,
        num_epochs=num_epochs,
        batch_size=batch_size,
        label_smoothing=label_smoothing
    )

    return vit_config

# Creating objective function for optuna study
def objective(trial):
    vit_config = create_vit_config(trial)

    data_module = CIFAR100DataModule(
        batch_size=vit_config.batch_size, 
        num_workers=num_workers, 
        transform=transform
    )
    model = LitViTModel(config=vit_config)

    # Using trial.number to differentiate each trial from one another
    logger = TensorBoardLogger("logs/tensorboard", name="vit_model", version=f"trial_{trial.number}")
    early_stopping = EarlyStopping('val_loss', patience=5)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    # Create trainer variable for pytorch lightning
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=vit_config.num_epochs,  # Using the number of epochs from config
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[early_stopping, checkpoint_callback, pruning_callback]
    )

    # Fit the model for each trial
    trainer.fit(model, datamodule=data_module)
    val_result = trainer.validate(model, datamodule=data_module)
    val_loss = val_result[0]['val_loss']

    # Log hyperparameters and their corresponding validation loss
    logger.log_hyperparams(trial.params, {'val_loss': val_loss})

    return val_loss

# for running script/cli
if __name__ == "__main__":
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=100)

    print(f'Best trial: {study.best_trial.value}')
    print(f'Best hyperparameters: {study.best_trial.params}')
