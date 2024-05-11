import torch
import torch.nn as nn
import gin
from dataclasses import dataclass, field
from typing import Optional, List

@gin.configurable
@dataclass
class TransformerEncoderConfig:
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float
    attention_dropout_rate: float
    use_dropout: bool = True
    use_attention_dropout: bool = True
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
    encoder_configs: List[TransformerEncoderConfig]
    num_classes: int
    optimizer: str = 'adam'
    learning_rate: float = 0.001

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # project and create patches
        x = x.flatten(2)  # flatten patches
        x = x.transpose(1, 2)  # change to B, N, C where N is the number of patches
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.head_dim, config.num_heads, dropout=config.attention_dropout_rate if config.use_attention_dropout else 0)
        self.norm1 = nn.LayerNorm(config.head_dim)
        self.norm2 = nn.LayerNorm(config.head_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.head_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.mlp_dim, config.head_dim),
            nn.Dropout(config.dropout_rate)
        )

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.self_attn(x1, x1, x1)[0]
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class ViTModel(nn.Module):
    def __init__(self, config: ViTModelConfig):
        super().__init__()
        self.patch_embed = PatchEmbedding(config.num_channels, config.patch_size, config.hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, (config.image_size // config.patch_size) ** 2 + 1, config.hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.layers = nn.ModuleList([TransformerEncoderLayer(cfg) for cfg in config.encoder_configs])
        self.head = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.head(x[:, 0])
        return x
