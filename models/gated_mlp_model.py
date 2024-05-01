import gin
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn

@gin.configurable
@dataclass
class GatedMLPLayerConfig:
    input_dim: int
    output_dim: int
    use_gate: bool
    dropout_rate: float

class GatedMLPLayer(nn.Module):
    def __init__(self, config: GatedMLPLayerConfig):
        super(GatedMLPLayer, self).__init__()
        self.linear = nn.Linear(config.input_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gate = nn.Linear(config.output_dim, config.output_dim) if config.use_gate else None

    def forward(self, x):
        x = self.linear(x)
        if self.gate:
            gate = torch.sigmoid(self.gate(x))
            x = x * gate
        x = self.dropout(x)
        return x

@gin.configurable
@dataclass
class GatedMLPModelConfig:
    model_name: str
    input_dim: int
    output_dim: int
    layers: List[GatedMLPLayerConfig] = field(default_factory=list)
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 10
    dropout_rate: float = 0.1

@gin.configurable
class GatedMLP(nn.Module):
    def __init__(self, config: GatedMLPModelConfig):
        super(GatedMLP, self).__init__()
        self.layers = nn.ModuleList()
        
        current_dim = config.input_dim
        for layer_config in config.layers:
            layer = GatedMLPLayer(GatedMLPLayerConfig(
                input_dim=current_dim,
                output_dim=layer_config.output_dim,
                use_gate=layer_config.use_gate,
                dropout_rate=layer_config.dropout_rate
            ))
            self.layers.append(layer)
            current_dim = layer_config.output_dim

        self.classifier = nn.Linear(current_dim, config.output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.classifier(x)
        return x
