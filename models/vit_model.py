@dataclass
class ViTLayerConfig:
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float
    attention_dropout_rate: float
    use_layer_norm: bool = True
    layer_norm_eps: Optional[float] = 1e-6

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
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, hidden_dim, H/P, W/P)
        x = x.flatten(2)  # (B, hidden_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, hidden_dim)
        return x

class ViTModel(nn.Module):
    def __init__(self, config: ViTModelConfig):
        super().__init__()
        # Calculate the number of patches
        self.num_patches = (config.image_size // config.patch_size) ** 2

        self.patch_embed = PatchEmbedding(
            in_channels=config.num_channels,
            patch_size=config.patch_size,
            hidden_dim=config.hidden_dim,
            img_size=config.image_size
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))

        encoder_layers = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout_rate,
            activation='gelu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.num_layers)

        self.head = nn.Linear(config.hidden_dim, config.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.transformer_encoder(x)

        x = self.head(x[:, 0])
        return x
