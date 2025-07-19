"""
Data configuration for Q-XAI framework.
Contains dataset-specific configurations and preprocessing parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os


@dataclass
class AudioConfig:
    """Audio preprocessing configuration."""
    
    # Basic audio parameters
    sample_rate: int = 16000
    mono: bool = True
    normalize: bool = True
    
    # Audio loading
    offset: float = 0.0  # Start offset in seconds
    duration: Optional[float] = None  # Duration to load (None = full audio)
    
    # Resampling
    resample_method: str = "kaiser_best"  # kaiser_best, kaiser_fast, scipy
    
    # Audio normalization
    norm_type: str = "peak"  # peak, rms, lufs
    target_level: float = -20.0  # Target level in dB for normalization


@dataclass
class SpectrogramConfig:
    """Spectrogram computation configuration."""
    
    # STFT parameters
    n_fft: int = 2048
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    window: str = "hann"
    center: bool = True
    pad_mode: str = "constant"
    
    # Mel-spectrogram parameters
    n_mels: int = 64
    f_min: float = 50.0
    f_max: Optional[float] = None  # None = sr/2
    power: float = 2.0
    norm: Optional[str] = None
    mel_scale: str = "htk"
    
    # Post-processing
    to_db: bool = True
    ref: float = 1.0
    amin: float = 1e-10
    top_db: Optional[float] = 80.0
    
    # Spectrogram normalization
    normalize_spec: bool = True
    spec_norm_type: str = "instance"  # instance, batch, global


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    
    # SpecAugment parameters
    use_specaugment: bool = True
    freq_mask_param: int = 8
    time_mask_param: int = 25
    num_freq_masks: int = 1
    num_time_masks: int = 1
    mask_value: float = 0.0
    
    # Audio augmentations
    use_audio_augment: bool = True
    
    # Time stretching
    time_stretch_prob: float = 0.3
    time_stretch_rate_range: Tuple[float, float] = (0.8, 1.2)
    
    # Pitch shifting
    pitch_shift_prob: float = 0.3
    pitch_shift_range: Tuple[float, float] = (-2.0, 2.0)  # In semitones
    
    # Noise addition
    noise_prob: float = 0.4
    noise_snr_range: Tuple[float, float] = (10.0, 30.0)  # SNR in dB
    
    # Volume perturbation
    volume_prob: float = 0.5
    volume_range: Tuple[float, float] = (0.7, 1.3)
    
    # Reverberation
    reverb_prob: float = 0.2
    reverb_rt60_range: Tuple[float, float] = (0.1, 0.8)
    
    # Mixup augmentation
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5


@dataclass
class DatasetConfig:
    """Base dataset configuration."""
    
    name: str
    root_path: str
    num_classes: int
    class_names: List[str]
    
    # Dataset splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # File format and structure
    audio_format: str = "wav"
    label_format: str = "csv"  # csv, json, folder_structure
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Caching
    cache_preprocessed: bool = False
    cache_dir: Optional[str] = None


@dataclass
class TUT2016Config(DatasetConfig):
    """TUT Acoustic Scenes 2016 dataset configuration."""
    
    name: str = "tut2016"
    root_path: str = "data/TUT-acoustic-scenes-2016"
    num_classes: int = 15
    class_names: List[str] = field(default_factory=lambda: [
        'beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
        'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
        'office', 'park', 'residential_area', 'train', 'tram'
    ])
    
    # TUT 2016 specific parameters
    audio_duration: float = 30.0  # 30-second clips
    fold_based_split: bool = True  # Use official folds
    evaluation_setup: str = "fold1_evaluate.txt"


@dataclass
class DCASE2019Config(DatasetConfig):
    """DCASE 2019 Task 1A dataset configuration."""
    
    name: str = "dcase2019"
    root_path: str = "data/DCASE2019-task1a"
    num_classes: int = 10
    class_names: List[str] = field(default_factory=lambda: [
        'airport', 'bus', 'metro', 'metro_station', 'park',
        'public_square', 'shopping_mall', 'street_pedestrian',
        'street_traffic', 'tram'
    ])
    
    # DCASE 2019 specific parameters
    audio_duration: float = 10.0  # 10-second clips
    device_mismatch: bool = True  # Include device mismatch conditions
    devices: List[str] = field(default_factory=lambda: ['a', 'b', 'c'])
    target_device: str = 'a'  # Primary recording device
    
    # Evaluation setup
    use_official_split: bool = True
    evaluation_mode: str = "full"  # full, development


@dataclass
class ESC50Config(DatasetConfig):
    """ESC-50 dataset configuration."""
    
    name: str = "esc50"
    root_path: str = "data/ESC-50"
    num_classes: int = 50
    class_names: List[str] = field(default_factory=lambda: [
        # Animals (0-9)
        'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
        # Natural soundscapes & water sounds (10-19)
        'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops',
        'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
        # Human, non-speech sounds (20-29)
        'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps',
        'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
        # Interior/domestic sounds (30-39)
        'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks',
        'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm',
        'clock_tick', 'glass_breaking',
        # Exterior/urban noises (40-49)
        'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train',
        'church_bells', 'airplane', 'fireworks', 'hand_saw'
    ])
    
    # ESC-50 specific parameters
    audio_duration: float = 5.0  # 5-second clips
    fold_based_eval: bool = True  # 5-fold cross-validation
    num_folds: int = 5


@dataclass
class CochlSceneConfig(DatasetConfig):
    """CochlScene dataset configuration."""
    
    name: str = "cochlscene"
    root_path: str = "data/CochlScene"
    num_classes: int = 13
    class_names: List[str] = field(default_factory=lambda: [
        'airport', 'bus', 'cafe', 'car', 'city_center', 'forest',
        'grocery_store', 'home', 'library', 'metro', 'office',
        'park', 'street'
    ])
    
    # CochlScene specific parameters
    crowdsourced: bool = True
    variable_duration: bool = True
    min_duration: float = 3.0
    max_duration: float = 30.0
    quality_filter: bool = True  # Filter low-quality recordings


@dataclass
class DCASE2025Config(DatasetConfig):
    """DCASE 2025 Task 1 dataset configuration."""
    
    name: str = "dcase2025"
    root_path: str = "data/DCASE2025-task1"
    num_classes: int = 10
    class_names: List[str] = field(default_factory=lambda: [
        'airport', 'bus', 'metro', 'metro_station', 'park',
        'public_square', 'shopping_mall', 'street_pedestrian',
        'street_traffic', 'tram'
    ])
    
    # DCASE 2025 specific parameters
    low_complexity: bool = True  # Low-complexity constraints
    device_information: bool = True  # Include device metadata
    max_parameters: int = 500000  # Parameter constraint for low-complexity
    
    # Development vs evaluation
    development_mode: bool = True
    include_mismatched: bool = True


@dataclass
class RobustnessTestConfig:
    """Configuration for robustness testing."""
    
    # Noise robustness
    test_noise: bool = True
    noise_types: List[str] = field(default_factory=lambda: ['white', 'pink', 'brown'])
    snr_levels: List[float] = field(default_factory=lambda: [20, 15, 10, 5, 0, -5])
    
    # Reverberation robustness
    test_reverberation: bool = True
    rt60_values: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 1.0])
    room_types: List[str] = field(default_factory=lambda: ['small', 'medium', 'large'])
    
    # Compression robustness
    test_compression: bool = True
    compression_formats: List[str] = field(default_factory=lambda: ['mp3', 'aac', 'ogg'])
    bitrates: List[int] = field(default_factory=lambda: [128, 256, 320])
    
    # Speed perturbation
    test_speed: bool = True
    speed_factors: List[float] = field(default_factory=lambda: [0.9, 0.95, 1.05, 1.1])
    
    # Pitch shift
    test_pitch: bool = True
    pitch_shifts: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.5, 1.0])


@dataclass
class DataLoaderConfig:
    """Data loader configuration."""
    
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Collation
    pad_sequences: bool = True
    max_length: Optional[int] = None
    padding_value: float = 0.0
    
    # Batching strategy
    batch_sampler: Optional[str] = None  # None, balanced, length_grouped
    
    # Memory optimization
    multiprocessing_context: Optional[str] = None  # spawn, fork, forkserver


class DatasetRegistry:
    """Registry for dataset configurations."""
    
    _configs = {
        'tut2016': TUT2016Config,
        'dcase2019': DCASE2019Config,
        'esc50': ESC50Config,
        'cochlscene': CochlSceneConfig,
        'dcase2025': DCASE2025Config,
    }
    
    @classmethod
    def get_config(cls, dataset_name: str) -> DatasetConfig:
        """Get configuration for a specific dataset."""
        if dataset_name not in cls._configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available datasets: {list(cls._configs.keys())}")
        
        return cls._configs[dataset_name]()
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all available datasets."""
        return list(cls._configs.keys())
    
    @classmethod
    def register_dataset(cls, name: str, config_class: type):
        """Register a new dataset configuration."""
        cls._configs[name] = config_class


@dataclass
class PreprocessingConfig:
    """Complete preprocessing configuration."""
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # Sequence processing
    max_length: Optional[int] = None  # Maximum sequence length
    pad_mode: str = "constant"  # constant, reflect, replicate
    pad_value: float = 0.0
    
    # Normalization
    global_normalize: bool = False
    normalization_stats_path: Optional[str] = None
    
    # Complex preprocessing
    preserve_phase: bool = True  # Keep phase information for complex processing
    phase_augmentation: bool = False  # Augment phase information
    
    def validate(self) -> None:
        """Validate preprocessing configuration."""
        assert self.audio.sample_rate > 0, "Sample rate must be positive"
        assert self.spectrogram.n_mels > 0, "Number of mel bins must be positive"
        assert self.spectrogram.n_fft > 0, "FFT size must be positive"
        assert self.spectrogram.hop_length > 0, "Hop length must be positive"
        
        # Validate augmentation probabilities
        assert 0 <= self.augmentation.time_stretch_prob <= 1
        assert 0 <= self.augmentation.pitch_shift_prob <= 1
        assert 0 <= self.augmentation.noise_prob <= 1
        assert 0 <= self.augmentation.volume_prob <= 1
        assert 0 <= self.augmentation.reverb_prob <= 1


def get_dataset_paths() -> Dict[str, str]:
    """Get default dataset paths."""
    data_root = os.environ.get('QXAI_DATA_ROOT', 'data')
    
    return {
        'tut2016': os.path.join(data_root, 'TUT-acoustic-scenes-2016'),
        'dcase2019': os.path.join(data_root, 'DCASE2019-task1a'),
        'esc50': os.path.join(data_root, 'ESC-50'),
        'cochlscene': os.path.join(data_root, 'CochlScene'),
        'dcase2025': os.path.join(data_root, 'DCASE2025-task1'),
    }


def create_dataset_config(
    dataset_name: str,
    root_path: Optional[str] = None,
    **kwargs
) -> DatasetConfig:
    """Create a dataset configuration with custom parameters."""
    config = DatasetRegistry.get_config(dataset_name)
    
    if root_path is not None:
        config.root_path = root_path
    
    # Update with custom parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown parameter '{key}' for dataset '{dataset_name}'")
    
    return config


def get_preprocessing_for_dataset(dataset_name: str) -> PreprocessingConfig:
    """Get recommended preprocessing configuration for a specific dataset."""
    base_config = PreprocessingConfig()
    
    if dataset_name == 'esc50':
        # ESC-50 has 5-second clips
        base_config.audio.duration = 5.0
        base_config.spectrogram.n_mels = 64
        
    elif dataset_name in ['dcase2019', 'dcase2025']:
        # DCASE datasets have 10-second clips
        base_config.audio.duration = 10.0
        base_config.spectrogram.n_mels = 64
        # Less aggressive augmentation for device mismatch scenarios
        base_config.augmentation.time_stretch_prob = 0.2
        base_config.augmentation.pitch_shift_prob = 0.2
        
    elif dataset_name == 'tut2016':
        # TUT 2016 has 30-second clips
        base_config.audio.duration = 30.0
        base_config.spectrogram.n_mels = 64
        
    elif dataset_name == 'cochlscene':
        # CochlScene has variable duration
        base_config.audio.duration = None  # Use full audio
        base_config.max_length = 1000  # Limit sequence length
        base_config.spectrogram.n_mels = 64
        
    return base_config