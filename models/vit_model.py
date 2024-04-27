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
    batch_size: int = 64
    num_epochs: int = 10

