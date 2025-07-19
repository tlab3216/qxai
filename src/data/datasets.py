"""
Dataset implementations for Q-XAI framework.
Supports multiple acoustic scene classification datasets with complex preprocessing.
"""

import os
import json
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

from config.data_config import (
    DatasetConfig, AudioConfig, SpectrogramConfig, AugmentationConfig,
    TUT2016Config, DCASE2019Config, ESC50Config, CochlSceneConfig, DCASE2025Config
)


class BaseASCDataset(Dataset, ABC):
    """Base class for Acoustic Scene Classification datasets."""
    
    def __init__(
        self,
        config: DatasetConfig,
        audio_config: AudioConfig,
        spectrogram_config: SpectrogramConfig,
        augmentation_config: Optional[AugmentationConfig] = None,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        self.config = config
        self.audio_config = audio_config
        self.spectrogram_config = spectrogram_config
        self.augmentation_config = augmentation_config
        self.split = split
        self.transform = transform
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(config.class_names)
        
        # Load dataset metadata
        self.samples = self._load_samples()
        
        # Setup augmentations
        self.use_augmentation = (split == 'train' and 
                               augmentation_config is not None and 
                               augmentation_config.use_specaugment)
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    @abstractmethod
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples and metadata."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load audio
        audio_path = sample['audio_path']
        audio, sr = self._load_audio(audio_path)
        
        # Apply audio augmentations
        if self.use_augmentation and self.augmentation_config.use_audio_augment:
            audio = self._apply_audio_augmentations(audio, sr)
        
        # Convert to spectrogram
        spectrogram = self._audio_to_spectrogram(audio, sr)
        
        # Apply spectrogram augmentations
        if self.use_augmentation:
            spectrogram = self._apply_spec_augmentations(spectrogram)
        
        # Convert to complex tensor for Q-XAI
        complex_spectrogram = self._to_complex_spectrogram(spectrogram)
        
        # Get label
        label = self.label_encoder.transform([sample['label']])[0]
        
        # Create output dictionary
        output = {
            'spectrogram': complex_spectrogram,
            'label': torch.tensor(label, dtype=torch.long),
            'audio_path': audio_path,
            'sample_metadata': sample
        }
        
        # Apply additional transforms
        if self.transform:
            output = self.transform(output)
        
        return output
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        try:
            # Use librosa for consistent loading
            audio, sr = librosa.load(
                audio_path,
                sr=self.audio_config.sample_rate,
                mono=self.audio_config.mono,
                offset=self.audio_config.offset,
                duration=self.audio_config.duration
            )
            
            # Normalize if requested
            if self.audio_config.normalize:
                if self.audio_config.norm_type == 'peak':
                    audio = audio / (np.max(np.abs(audio)) + 1e-8)
                elif self.audio_config.norm_type == 'rms':
                    rms = np.sqrt(np.mean(audio**2))
                    audio = audio / (rms + 1e-8)
            
            return audio, sr
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return silence if loading fails
            duration = self.audio_config.duration or 5.0
            length = int(duration * self.audio_config.sample_rate)
            return np.zeros(length), self.audio_config.sample_rate
    
    def _audio_to_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Convert audio to mel-spectrogram."""
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.spectrogram_config.n_fft,
            hop_length=self.spectrogram_config.hop_length,
            win_length=self.spectrogram_config.win_length,
            window=self.spectrogram_config.window,
            center=self.spectrogram_config.center,
            pad_mode=self.spectrogram_config.pad_mode,
            n_mels=self.spectrogram_config.n_mels,
            fmin=self.spectrogram_config.f_min,
            fmax=self.spectrogram_config.f_max,
            power=self.spectrogram_config.power,
            norm=self.spectrogram_config.norm
        )
        
        # Convert to dB if requested
        if self.spectrogram_config.to_db:
            mel_spec = librosa.power_to_db(
                mel_spec,
                ref=self.spectrogram_config.ref,
                amin=self.spectrogram_config.amin,
                top_db=self.spectrogram_config.top_db
            )
        
        # Normalize spectrogram
        if self.spectrogram_config.normalize_spec:
            if self.spectrogram_config.spec_norm_type == 'instance':
                mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            elif self.spectrogram_config.spec_norm_type == 'global':
                # Global normalization (would need precomputed stats)
                pass
        
        return mel_spec.T  # Shape: (time, freq)
    
    def _to_complex_spectrogram(self, spectrogram: np.ndarray) -> torch.Tensor:
        """Convert real spectrogram to complex tensor for Q-XAI."""
        # For real mel-spectrograms, we need to create a complex representation
        # One approach: use the magnitude as real part, add small random phase
        real_part = torch.tensor(spectrogram, dtype=torch.float32)
        
        # Add small imaginary component for phase diversity
        # This could be learned or based on some audio property
        imag_part = 0.1 * torch.randn_like(real_part)
        
        complex_spec = torch.complex(real_part, imag_part)
        
        return complex_spec
    
    def _apply_audio_augmentations(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply audio-level augmentations."""
        if not self.augmentation_config:
            return audio
        
        # Time stretching
        if (self.augmentation_config.time_stretch_prob > 0 and 
            random.random() < self.augmentation_config.time_stretch_prob):
            rate = random.uniform(*self.augmentation_config.time_stretch_rate_range)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Pitch shifting
        if (self.augmentation_config.pitch_shift_prob > 0 and 
            random.random() < self.augmentation_config.pitch_shift_prob):
            n_steps = random.uniform(*self.augmentation_config.pitch_shift_range)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        
        # Add noise
        if (self.augmentation_config.noise_prob > 0 and 
            random.random() < self.augmentation_config.noise_prob):
            noise_level = random.uniform(*self.augmentation_config.noise_snr_range)
            noise = np.random.normal(0, 1, audio.shape)
            noise_power = np.mean(noise**2)
            signal_power = np.mean(audio**2)
            noise_factor = np.sqrt(signal_power / (noise_power * 10**(noise_level/10)))
            audio = audio + noise_factor * noise
        
        # Volume perturbation
        if (self.augmentation_config.volume_prob > 0 and 
            random.random() < self.augmentation_config.volume_prob):
            volume_factor = random.uniform(*self.augmentation_config.volume_range)
            audio = audio * volume_factor
        
        return audio
    
    def _apply_spec_augmentations(self, spectrogram: np.ndarray) -> np.ndarray:
        """Apply SpecAugment to spectrogram."""
        if not self.augmentation_config or not self.augmentation_config.use_specaugment:
            return spectrogram
        
        spec_tensor = torch.tensor(spectrogram)
        
        # Frequency masking
        for _ in range(self.augmentation_config.num_freq_masks):
            freq_mask_param = self.augmentation_config.freq_mask_param
            if freq_mask_param > 0:
                f = random.randint(0, freq_mask_param)
                f0 = random.randint(0, max(1, spec_tensor.shape[1] - f))
                spec_tensor[:, f0:f0+f] = self.augmentation_config.mask_value
        
        # Time masking
        for _ in range(self.augmentation_config.num_time_masks):
            time_mask_param = self.augmentation_config.time_mask_param
            if time_mask_param > 0:
                t = random.randint(0, time_mask_param)
                t0 = random.randint(0, max(1, spec_tensor.shape[0] - t))
                spec_tensor[t0:t0+t, :] = self.augmentation_config.mask_value
        
        return spec_tensor.numpy()
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training."""
        labels = [sample['label'] for sample in self.samples]
        label_counts = pd.Series(labels).value_counts()
        
        weights = []
        for class_name in self.config.class_names:
            count = label_counts.get(class_name, 1)
            weights.append(1.0 / count)
        
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        return weights


class TUT2016Dataset(BaseASCDataset):
    """TUT Acoustic Scenes 2016 dataset."""
    
    def __init__(self, config: TUT2016Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load TUT 2016 samples."""
        samples = []
        
        # Load metadata file
        if self.config.fold_based_split:
            # Use official evaluation setup
            meta_file = os.path.join(self.config.root_path, 'meta.txt')
            eval_file = os.path.join(self.config.root_path, self.config.evaluation_setup)
            
            # Read all files first
            all_files = {}
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            filename = parts[0]
                            scene_label = parts[1]
                            all_files[filename] = scene_label
            
            # Determine which files to use based on split
            if self.split == 'train':
                # Use files not in evaluation setup
                eval_files = set()
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f:
                        for line in f:
                            filename = line.strip().split('\t')[0]
                            eval_files.add(filename)
                
                for filename, label in all_files.items():
                    if filename not in eval_files:
                        audio_path = os.path.join(self.config.root_path, filename)
                        if os.path.exists(audio_path):
                            samples.append({
                                'audio_path': audio_path,
                                'label': label,
                                'filename': filename
                            })
            
            else:  # test split
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f:
                        for line in f:
                            filename = line.strip().split('\t')[0]
                            if filename in all_files:
                                audio_path = os.path.join(self.config.root_path, filename)
                                if os.path.exists(audio_path):
                                    samples.append({
                                        'audio_path': audio_path,
                                        'label': all_files[filename],
                                        'filename': filename
                                    })
        
        else:
            # Random split
            audio_dir = os.path.join(self.config.root_path, 'audio')
            for audio_file in os.listdir(audio_dir):
                if audio_file.endswith('.wav'):
                    # Extract label from filename (assuming format: scene_label_*.wav)
                    scene_label = audio_file.split('_')[0]
                    if scene_label in self.config.class_names:
                        audio_path = os.path.join(audio_dir, audio_file)
                        samples.append({
                            'audio_path': audio_path,
                            'label': scene_label,
                            'filename': audio_file
                        })
            
            # Split randomly
            random.shuffle(samples)
            n_total = len(samples)
            n_train = int(n_total * self.config.train_split)
            n_val = int(n_total * self.config.val_split)
            
            if self.split == 'train':
                samples = samples[:n_train]
            elif self.split == 'val':
                samples = samples[n_train:n_train+n_val]
            else:  # test
                samples = samples[n_train+n_val:]
        
        return samples


class DCASE2019Dataset(BaseASCDataset):
    """DCASE 2019 Task 1A dataset."""
    
    def __init__(self, config: DCASE2019Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load DCASE 2019 samples."""
        samples = []
        
        if self.config.use_official_split:
            # Use official meta files
            if self.split == 'train':
                meta_file = 'meta_train.csv'
            elif self.split == 'val':
                meta_file = 'meta_dev.csv'
            else:
                meta_file = 'meta_eval.csv'
            
            meta_path = os.path.join(self.config.root_path, meta_file)
            
            if os.path.exists(meta_path):
                df = pd.read_csv(meta_path, sep='\t')
                
                for _, row in df.iterrows():
                    filename = row['filename']
                    scene_label = row['scene_label']
                    
                    # Handle device mismatch
                    if self.config.device_mismatch:
                        device = filename.split('-')[-1].split('.')[0]  # Extract device info
                        if device not in self.config.devices:
                            continue
                    
                    audio_path = os.path.join(self.config.root_path, 'audio', filename)
                    if os.path.exists(audio_path):
                        samples.append({
                            'audio_path': audio_path,
                            'label': scene_label,
                            'filename': filename,
                            'device': device if self.config.device_mismatch else 'unknown'
                        })
        
        return samples


class ESC50Dataset(BaseASCDataset):
    """ESC-50 dataset."""
    
    def __init__(self, config: ESC50Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load ESC-50 samples."""
        samples = []
        
        # Load metadata
        meta_file = os.path.join(self.config.root_path, 'meta', 'esc50.csv')
        
        if os.path.exists(meta_file):
            df = pd.read_csv(meta_file)
            
            if self.config.fold_based_eval:
                # Use 5-fold cross-validation
                # For simplicity, use fold 1 for test, others for train
                if self.split == 'train':
                    df_split = df[df['fold'] != 1]
                    # Further split for validation
                    train_folds = df_split[df_split['fold'] != 2]
                    val_folds = df_split[df_split['fold'] == 2]
                    
                    if self.split == 'train':
                        df_split = train_folds
                    else:  # val
                        df_split = val_folds
                        
                elif self.split == 'test':
                    df_split = df[df['fold'] == 1]
                else:  # val
                    df_split = df[df['fold'] == 2]
            else:
                # Random split
                n_total = len(df)
                n_train = int(n_total * self.config.train_split)
                n_val = int(n_total * self.config.val_split)
                
                if self.split == 'train':
                    df_split = df.iloc[:n_train]
                elif self.split == 'val':
                    df_split = df.iloc[n_train:n_train+n_val]
                else:  # test
                    df_split = df.iloc[n_train+n_val:]
            
            for _, row in df_split.iterrows():
                filename = row['filename']
                category = row['category']
                
                audio_path = os.path.join(self.config.root_path, 'audio', filename)
                if os.path.exists(audio_path):
                    samples.append({
                        'audio_path': audio_path,
                        'label': category,
                        'filename': filename,
                        'fold': row['fold'],
                        'target': row['target'],
                        'category': category
                    })
        
        return samples


class CochlSceneDataset(BaseASCDataset):
    """CochlScene dataset."""
    
    def __init__(self, config: CochlSceneConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load CochlScene samples."""
        samples = []
        
        # Load metadata
        meta_file = os.path.join(self.config.root_path, 'meta.csv')
        
        if os.path.exists(meta_file):
            df = pd.read_csv(meta_file)
            
            # Apply quality filter if requested
            if self.config.quality_filter:
                # Filter based on some quality criteria (duration, SNR, etc.)
                df = df[df['duration'] >= self.config.min_duration]
                df = df[df['duration'] <= self.config.max_duration]
            
            # Random split
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            n_total = len(df)
            n_train = int(n_total * self.config.train_split)
            n_val = int(n_total * self.config.val_split)
            
            if self.split == 'train':
                df_split = df.iloc[:n_train]
            elif self.split == 'val':
                df_split = df.iloc[n_train:n_train+n_val]
            else:  # test
                df_split = df.iloc[n_train+n_val:]
            
            for _, row in df_split.iterrows():
                filename = row['filename']
                scene_label = row['scene_label']
                
                audio_path = os.path.join(self.config.root_path, 'audio', filename)
                if os.path.exists(audio_path):
                    samples.append({
                        'audio_path': audio_path,
                        'label': scene_label,
                        'filename': filename,
                        'duration': row['duration'],
                        'quality_score': row.get('quality_score', 1.0)
                    })
        
        return samples


class DCASE2025Dataset(BaseASCDataset):
    """DCASE 2025 Task 1 dataset."""
    
    def __init__(self, config: DCASE2025Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load DCASE 2025 samples."""
        samples = []
        
        # Similar to DCASE 2019 but with device information
        if self.config.development_mode:
            meta_file = f'meta_{self.split}.csv'
        else:
            meta_file = 'meta_eval.csv'
        
        meta_path = os.path.join(self.config.root_path, meta_file)
        
        if os.path.exists(meta_path):
            df = pd.read_csv(meta_path, sep='\t')
            
            for _, row in df.iterrows():
                filename = row['filename']
                scene_label = row['scene_label']
                device_info = row.get('device', 'unknown')
                
                audio_path = os.path.join(self.config.root_path, 'audio', filename)
                if os.path.exists(audio_path):
                    samples.append({
                        'audio_path': audio_path,
                        'label': scene_label,
                        'filename': filename,
                        'device': device_info,
                        'low_complexity': self.config.low_complexity
                    })
        
        return samples


def create_dataloader(
    dataset: BaseASCDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """Create a DataLoader for ASC dataset."""
    
    def default_collate_fn(batch):
        """Default collation function for complex spectrograms."""
        spectrograms = []
        labels = []
        audio_paths = []
        metadata = []
        
        for item in batch:
            spectrograms.append(item['spectrogram'])
            labels.append(item['label'])
            audio_paths.append(item['audio_path'])
            metadata.append(item['sample_metadata'])
        
        # Pad spectrograms to same length
        max_length = max(spec.shape[0] for spec in spectrograms)
        padded_specs = []
        
        for spec in spectrograms:
            if spec.shape[0] < max_length:
                padding = max_length - spec.shape[0]
                padded_spec = F.pad(spec, (0, 0, 0, padding), value=0.0)
            else:
                padded_spec = spec
            padded_specs.append(padded_spec)
        
        return {
            'spectrogram': torch.stack(padded_specs),
            'label': torch.stack(labels),
            'audio_path': audio_paths,
            'sample_metadata': metadata
        }
    
    if collate_fn is None:
        collate_fn = default_collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        drop_last=shuffle  # Drop last for training to avoid batch size issues
    )