from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch
from torch import nn

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
    pool_type: Optional[str] = 'max'
    use_dropout: bool = False
    dropout_rate: Optional[float] = 0.0

@dataclass
class CNNModelConfig:
    model_name: str
    layers: List[CNNLayerConfig] = field(default_factory=list)
    input_shape: Tuple[int, int, int] = (3, 32, 32)
    output_shape: int = 100
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10

class CNNModel(nn.Module):
    def __init__(self, config: CNNModelConfig):
        super().__init__()
        self.layers = nn.Sequential()
        for idx, layer_config in enumerate(config.layers):
            self.layers.add_module(f"conv{idx}", nn.Conv2d(
                in_channels=layer_config.in_channels,
                out_channels=layer_config.out_channels,
                kernel_size=layer_config.kernel_size,
                stride=layer_config.stride,
                padding=layer_config.padding
            ))
            if layer_config.use_batch_norm:
                self.layers.add_module(f"batch_norm{idx}", nn.BatchNorm2d(layer_config.out_channels))
            self.layers.add_module(f"activation{idx}", nn.SiLU())
            if layer_config.use_pool:
                pool = nn.MaxPool2d if layer_config.pool_type == 'max' else nn.AvgPool2d
                self.layers.add_module(f"pool{idx}", pool(kernel_size=layer_config.pool_size, stride=layer_config.pool_stride))
            if layer_config.use_dropout:
                self.layers.add_module(f"dropout{idx}", nn.Dropout(layer_config.dropout_rate))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(config.output_shape)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x
