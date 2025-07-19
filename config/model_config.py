"""
Model configuration for Q-XAI framework.
Contains all hyperparameters and settings for the Complex-Valued Transformer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math


@dataclass
class ComplexTransformerConfig:
    """Configuration for the Complex-Valued Transformer architecture."""
    
    # Model architecture
    input_dim: int = 64  # Mel-spectrogram frequency bins
    embed_dim: int = 256  # Hidden dimension (d in paper)
    num_heads: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of transformer layers (L in paper)
    ff_dim: int = 1024  # Feed-forward dimension (4 * embed_dim)
    num_classes: int = 10  # Number of output classes
    max_seq_length: int = 1000  # Maximum sequence length
    
    # Dropout and regularization
    dropout: float = 0.1  # Dropout rate (p in paper)
    attention_dropout: float = 0.1  # Attention dropout
    path_dropout: float = 0.0  # Stochastic depth
    
    # Complex-specific parameters
    complex_init_std: float = 0.02  # Standard deviation for complex initialization
    wirtinger_alpha: float = 0.5  # Weighting for Wirtinger derivatives in QISA
    
    # Activation functions
    activation: str = "complex_gelu"  # complex_relu, complex_gelu
    
    # Layer normalization
    layer_norm_eps: float = 1e-5
    
    # Positional encoding
    pos_encoding_type: str = "complex_sinusoidal"  # complex_sinusoidal, learnable
    pos_encoding_dropout: float = 0.1


@dataclass
class QISAConfig:
    """Configuration for Quantum-Inspired State Attribution (QISA)."""
    
    # Wirtinger calculus parameters
    wirtinger_alpha: float = 0.5  # Equal weighting for holomorphic/anti-holomorphic derivatives
    create_graph: bool = False  # Whether to create computation graph for higher-order derivatives
    
    # Attribution computation
    attribution_method: str = "squared_magnitude"  # squared_magnitude, magnitude, real_part
    aggregate_heads: bool = True  # Whether to aggregate attention heads
    aggregate_layers: bool = True  # Whether to aggregate across layers
    
    # Smoothing and post-processing
    smooth_attributions: bool = True
    smooth_kernel_size: int = 3
    normalize_attributions: bool = True


@dataclass
class AUQConfig:
    """Configuration for Amplitude-based Uncertainty Quantification (AUQ)."""
    
    # Monte Carlo sampling
    num_mc_samples: int = 50  # M in paper
    mc_dropout_rate: float = 0.1  # Dropout rate for MC sampling
    
    # Uncertainty decomposition
    compute_epistemic: bool = True  # Model uncertainty
    compute_aleatoric: bool = True  # Data uncertainty
    compute_covariance: bool = True  # Novel covariance uncertainty
    
    # Convergence criteria
    convergence_threshold: float = 0.01  # MSE threshold for convergence
    min_samples: int = 30  # Minimum number of samples before checking convergence
    
    # Uncertainty aggregation
    uncertainty_aggregation: str = "mean"  # mean, max, weighted


@dataclass
class QICPConfig:
    """Configuration for Quantum-Inspired Conformal Prediction (QICP)."""
    
    # Conformal prediction parameters
    confidence_level: float = 0.9  # 1 - α (target coverage)
    calibration_split: float = 0.2  # Fraction of training data for calibration
    
    # Nonconformity score
    score_function: str = "born_rule"  # born_rule (|f(x)_y|²), softmax, entropy
    
    # Prediction set construction
    adaptive_threshold: bool = True  # Whether to use adaptive thresholds
    min_set_size: int = 1  # Minimum prediction set size
    max_set_size: Optional[int] = None  # Maximum prediction set size
    
    # Validation
    validate_coverage: bool = True  # Whether to validate coverage on test set


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    optimizer: str = "adamw"  # adam, adamw, sgd
    learning_rate: float = 1e-4  # Learning rate
    weight_decay: float = 1e-2  # Weight decay
    beta1: float = 0.9  # Adam beta1
    beta2: float = 0.999  # Adam beta2
    eps: float = 1e-8  # Adam epsilon
    
    # Learning rate scheduling
    scheduler: str = "cosine"  # cosine, step, exponential, none
    warmup_epochs: int = 10  # Warmup epochs
    warmup_start_lr: float = 1e-6  # Starting LR for warmup
    cosine_eta_min: float = 0.0  # Minimum LR for cosine annealing
    
    # Training parameters
    epochs: int = 100  # Total training epochs
    batch_size: int = 32  # Batch size
    accumulate_grad_batches: int = 1  # Gradient accumulation
    clip_grad_norm: Optional[float] = 1.0  # Gradient clipping
    
    # Validation and checkpointing
    val_check_interval: float = 1.0  # Validation frequency (fraction of epoch)
    save_top_k: int = 3  # Number of best checkpoints to save
    monitor_metric: str = "val_accuracy"  # Metric to monitor for checkpointing
    monitor_mode: str = "max"  # max or min
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15  # Early stopping patience
    min_delta: float = 0.001  # Minimum change for improvement
    
    # Loss function
    loss_function: str = "cross_entropy"  # cross_entropy, focal_loss
    label_smoothing: float = 0.0  # Label smoothing parameter
    
    # Mixed precision training
    use_amp: bool = False  # Automatic mixed precision (complex numbers may not support)
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class DataConfig:
    """Data configuration."""
    
    # Audio preprocessing
    sample_rate: int = 16000  # Audio sample rate
    n_mels: int = 64  # Number of mel-frequency bins
    n_fft: int = 2048  # FFT window size
    hop_length: int = 160  # Hop length (10ms at 16kHz)
    win_length: int = 400  # Window length (25ms at 16kHz)
    f_min: float = 50.0  # Minimum frequency
    f_max: Optional[float] = None  # Maximum frequency (None = sr/2)
    
    # Spectrogram parameters
    power: float = 2.0  # Power for mel spectrogram
    norm: Optional[str] = None  # Normalization for mel basis
    mel_scale: str = "htk"  # Mel scale (htk, slaney)
    
    # Data augmentation
    use_specaugment: bool = True  # SpecAugment during training
    freq_mask_param: int = 8  # Frequency masking parameter
    time_mask_param: int = 25  # Time masking parameter
    num_masks: int = 2  # Number of masks to apply
    
    # Normalization
    normalize_audio: bool = True  # Normalize audio amplitude
    normalize_spectrogram: bool = True  # Normalize spectrogram
    norm_type: str = "instance"  # instance, batch, layer
    
    # Sequence length
    max_audio_length: float = 10.0  # Maximum audio length in seconds
    pad_mode: str = "constant"  # Padding mode for shorter sequences
    
    # Dataset splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Data loading
    num_workers: int = 4  # Number of data loading workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    persistent_workers: bool = True  # Keep workers alive between epochs


@dataclass
class ExperimentConfig:
    """Experiment and evaluation configuration."""
    
    # Experiment tracking
    experiment_name: str = "q_xai_asc"
    project_name: str = "interpretable_asc"
    tags: List[str] = field(default_factory=lambda: ["complex_transformer", "interpretability"])
    
    # Logging
    log_every_n_steps: int = 10
    log_gradients: bool = False
    log_model_architecture: bool = True
    
    # Evaluation
    evaluate_on_datasets: List[str] = field(default_factory=lambda: [
        "tut2016", "dcase2019", "esc50", "cochlscene", "dcase2025"
    ])
    
    # Robustness testing
    test_robustness: bool = True
    noise_levels_snr: List[float] = field(default_factory=lambda: [20, 15, 10, 5, 0, -5])
    reverberation_rt60: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.5, 0.8])
    
    # Statistical testing
    num_runs: int = 5  # Number of runs for statistical significance
    significance_level: float = 0.05  # p-value threshold
    
    # Computational efficiency
    measure_efficiency: bool = True
    profile_memory: bool = True
    benchmark_inference: bool = True


@dataclass
class QXAIConfig:
    """Complete Q-XAI framework configuration."""
    
    model: ComplexTransformerConfig = field(default_factory=ComplexTransformerConfig)
    qisa: QISAConfig = field(default_factory=QISAConfig)
    auq: AUQConfig = field(default_factory=AUQConfig)
    qicp: QICPConfig = field(default_factory=QICPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Model validation
        assert self.model.embed_dim % self.model.num_heads == 0, \
            "embed_dim must be divisible by num_heads"
        assert self.model.ff_dim > 0, "ff_dim must be positive"
        assert self.model.num_layers > 0, "num_layers must be positive"
        
        # Training validation
        assert 0 < self.training.learning_rate < 1, "learning_rate must be in (0, 1)"
        assert self.training.batch_size > 0, "batch_size must be positive"
        assert self.training.epochs > 0, "epochs must be positive"
        
        # Data validation
        assert self.data.sample_rate > 0, "sample_rate must be positive"
        assert self.data.n_mels > 0, "n_mels must be positive"
        assert abs(self.data.train_split + self.data.val_split + self.data.test_split - 1.0) < 1e-6, \
            "Data splits must sum to 1.0"
        
        # Q-XAI validation
        assert 0 <= self.qisa.wirtinger_alpha <= 1, "wirtinger_alpha must be in [0, 1]"
        assert self.auq.num_mc_samples > 0, "num_mc_samples must be positive"
        assert 0 < self.qicp.confidence_level < 1, "confidence_level must be in (0, 1)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QXAIConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Predefined configurations for different datasets and experiments

def get_tut2016_config() -> QXAIConfig:
    """Configuration optimized for TUT 2016 dataset."""
    config = QXAIConfig()
    config.model.num_classes = 15  # 15 urban scenes
    config.training.epochs = 100
    config.training.batch_size = 32
    return config


def get_dcase2019_config() -> QXAIConfig:
    """Configuration optimized for DCASE 2019 Task 1A."""
    config = QXAIConfig()
    config.model.num_classes = 10  # 10 scenes
    config.training.epochs = 120
    config.training.batch_size = 16  # Smaller batch for device mismatch
    config.training.learning_rate = 5e-5  # Lower LR for challenging conditions
    return config


def get_esc50_config() -> QXAIConfig:
    """Configuration optimized for ESC-50 dataset."""
    config = QXAIConfig()
    config.model.num_classes = 50  # 50 environmental sound classes
    config.training.epochs = 150
    config.training.batch_size = 64  # Larger batch for diverse classes
    config.data.max_audio_length = 5.0  # ESC-50 clips are 5 seconds
    return config


def get_ablation_config() -> QXAIConfig:
    """Configuration for ablation studies."""
    config = QXAIConfig()
    config.experiment.num_runs = 5  # Multiple runs for statistical significance
    config.training.epochs = 80  # Shorter training for faster ablation
    return config


def get_efficiency_config() -> QXAIConfig:
    """Configuration for computational efficiency experiments."""
    config = QXAIConfig()
    config.model.embed_dim = 128  # Smaller model for efficiency testing
    config.model.num_heads = 4
    config.model.num_layers = 4
    config.experiment.measure_efficiency = True
    config.experiment.profile_memory = True
    config.experiment.benchmark_inference = True
    return config