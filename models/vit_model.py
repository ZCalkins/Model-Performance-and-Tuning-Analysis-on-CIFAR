import gin
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder

@gin.configurable
@dataclass
class ViTLayerConfig:
    num_heads: int
    head_dim: int
    mlp_dim: int
    use_dropout: bool = False
    dropout_rate: Optional[float] = 0.0
    use_attention_dropout: bool = False
    attention_dropout_rate: Optional[float] = 0.0
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
    optimizer: str = 'adam'
    learning_rate: float = 0.001

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout_rate, attention_dropout_rate, use_dropout=True, use_attention_dropout=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout_rate if use_attention_dropout else 0.0)
        self.linear1 = nn.Linear(hidden_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout_rate if use_dropout else 0.0)
        self.linear2 = nn.Linear(mlp_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate if use_dropout else 0.0)
        self.dropout2 = nn.Dropout(dropout_rate if use_dropout else 0.0)
        
        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(attn_output)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

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
        self.encoder = nn.ModuleList([
            CustomTransformerEncoderLayer(
                hidden_dim=config.hidden_dim, 
                num_heads=config.num_heads, 
                mlp_dim=config.mlp_dim, 
                dropout_rate=config.dropout_rate, 
                attention_dropout_rate=config.attention_dropout_rate, 
                use_dropout=config.use_dropout, 
                use_attention_dropout=config.use_attention_dropout
            ) for _ in range(config.num_layers)
        ])
        self.head = nn.Linear(config.hidden_dim, config.num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
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
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        for layer in self.encoder:
            x = layer(x)
        x = x[:, 0]
        x = self.head(x)
        return x
