import gin
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.init as init
import importlib

@dataclass
class CNNLayerConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: Optional[int] = 0
    use_batch_norm: bool = False
    use_pool: bool = False
    pool_size: Optional[int] = 2
    pool_stride: Optional[int] = 2
    pool_type: Optional[str] = 'MaxPool2d'
    use_dropout: bool = False
    dropout_rate: Optional[float] = 0.0
    activation: str = 'ReLU'
    norm_layer: Optional[str] = None
    norm_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def pool_params(self):
        return {
            'kernel_size': self.pool_size,
            'stride': self.pool_stride,
        }

@dataclass
class CNNModelConfig:
    model_name: str
    layers: List[CNNLayerConfig] = field(default_factory=list)
    input_shape: Tuple[int, int, int] = (3, 32, 32)
    output_shape: int = 100
    optimizer_class: str = 'adam'
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {'lr': 0.001})
    scheduler_class: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 32
    num_epochs: int = 10
    label_smoothing: float = 0.0

def initialize_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        if hasattr(layer, 'weight') and layer.weight is not None:
            if isinstance(layer, nn.Conv2d):
                if any(isinstance(activation, nn.ReLU) for activation in layer.children()):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif any(isinstance(activation, nn.LeakyReLU) for activation in layer.children()):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
                else:
                    nn.init.xavier_normal_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.BatchNorm2d):
        if hasattr(layer, 'weight') and layer.weight is not None:
            nn.init.constant_(layer.weight, 1)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

class CNNModel(nn.Module):
    def __init__(self, config: CNNModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential()
        
        for idx, layer_config in enumerate(config.layers):
            modules = []
            
            # Convolutional layers
            conv_layer = nn.Conv2d(
                in_channels=layer_config.in_channels,
                out_channels=layer_config.out_channels,
                kernel_size=layer_config.kernel_size,
                stride=layer_config.stride,
                padding=layer_config.padding
            )
            modules.append(conv_layer)
            
            # Optional Batch Normalization
            if layer_config.use_batch_norm:
                batch_norm = nn.BatchNorm2d(layer_config.out_channels)
                modules.append(batch_norm)

            # Optional Activation Function with Error Checking
            if layer_config.activation:
                try:
                    activation_module = importlib.import_module('torch.nn')
                    ActivationFunction = getattr(activation_module, layer_config.activation)
                    activation = ActivationFunction()
                    modules.append(activation)
                except AttributeError:
                    raise ValueError(f"Activation function '{layer_config.activation}' is not a valid function in torch.nn.")
                except Exception as e:
                    raise ValueError(f"An error occurred while setting up the activation function '{layer_config.activation}': {e}")
            
            # Optional Pooling
            if layer_config.use_pool and layer_config.pool_type:
                pool_module = importlib.import_module('torch.nn')
                PoolClass = getattr(pool_module, layer_config.pool_type)
                pool = PoolClass(**layer_config.pool_params)
                modules.append(pool)
            
            # Optional Dropout
            if layer_config.use_dropout:
                dropout = nn.Dropout(layer_config.dropout_rate)
                modules.append(dropout)
            
            # Optional Normalization Layer
            if layer_config.norm_layer:
                norm_module = importlib.import_module('torch.nn')
                NormLayer = getattr(norm_module, layer_config.norm_layer)
                norm_layer = NormLayer(**layer_config.norm_params)
                modules.append(norm_layer)

            # Move the configured sub-layer (block of layers) to the main layer stack
            self.layers.add_module(f"block_{idx}", nn.Sequential(*modules))

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
            nn.Flatten(),
            nn.LazyLinear(config.output_shape)
        )

        self.apply(initialize_weights)

        dummy_input = torch.zeros(1, *config.input_shape)
        self(dummy_input)

    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x
