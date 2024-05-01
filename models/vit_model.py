import gin
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

@gin.configurable
@dataclass
class ViTLayerConfig:
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float
    attention_dropout_rate: float
    use_layer_norm: bool = True
    layer_norm_eps: Optional[float] = 1e-6

@gin.configurable
@dataclass
class ViTModelConfig:
    model_name: str
    image_size: int
    patch_size: int
    num_channels: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    num_classes: int
    dropout_rate: float
    attention_dropout_rate: float
    optimizer: str = 'adam'
    learning_rate: float = 0.001

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, hidden_dim: int, img_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class ViTModel(nn.Module):
    def __init__(self, config: ViTModelConfig):
        super().__init__()
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.patch_embed = PatchEmbedding(
            in_channels=config.num_channels,
            patch_size=config.patch_size,
            hidden_dim=config.hidden_dim,
            img_size=config.image_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout_rate,
            activation='gelu'
        )
        self.encoder = TransformerEncoder(encoder_layer, config.num_layers)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='linear')
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.encoder(x)
        x = x[:, 0]
        x = self.head(x)
        return x
