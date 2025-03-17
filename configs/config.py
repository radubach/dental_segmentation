from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
from omegaconf import MISSING

@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    # Paths
    image_dir: str = MISSING
    coco_json: str = MISSING
    mask_dir: Optional[str] = None
    
    # Image processing
    input_size: Tuple[int, int] = (256, 256)
    normalize_mean: Tuple[float, ...] = (0.5,)
    normalize_std: Tuple[float, ...] = (0.5,)
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation
    augment: bool = False
    augment_prob: float = 0.5
    rotation_limit: int = 10
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    noise_limit: Tuple[float, float] = (10.0, 50.0)

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    type: str = MISSING  # "unet" or "maskrcnn"
    
    # Common parameters
    in_channels: int = 1
    out_channels: int = 33  # 32 teeth + background
    pretrained: bool = False
    
    # UNet specific
    features_start: int = 64  # Number of features in first layer
    num_layers: int = 4      # Number of down/up-sampling layers
    
    # Mask R-CNN specific
    backbone: str = "resnet50"
    trainable_backbone_layers: int = 5
    rpn_score_thresh: float = 0.05
    box_score_thresh: float = 0.05

@dataclass
class OptimizerConfig:
    """Configuration for optimizer and learning rate."""
    type: str = "adam"  # "adam", "sgd", "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9    # For Adam
    beta2: float = 0.999  # For Adam

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # Basic training parameters
    epochs: int = 100
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_frequency: int = 1
    keep_last_k: int = 5
    
    # Validation
    val_frequency: int = 1
    early_stopping_patience: int = 10
    
    # Logging
    log_dir: str = "logs"
    experiment_name: str = "dental_segmentation"
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    
    # Loss weights (for multi-task learning)
    loss_weights: Optional[dict] = None

@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Global settings
    seed: int = 42
    debug: bool = False
    verbose: bool = True

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from yaml file or return default."""
    from omegaconf import OmegaConf
    
    # Create default config
    default_config = OmegaConf.structured(Config)
    
    if config_path is not None:
        # Load and merge with user config
        user_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(default_config, user_config)
    else:
        config = default_config
    
    return config

def save_config(config: Config, save_path: str) -> None:
    """Save configuration to yaml file."""
    from omegaconf import OmegaConf
    
    with open(save_path, 'w') as f:
        OmegaConf.save(config=config, f=f) 