"""
Advanced preprocessing pipeline for Q-XAI framework.
Includes complex-valued preprocessing, robustness testing, and phase-aware transformations.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
import scipy.signal
from scipy.stats import norm
import pyroomacoustics as pra

from config.data_config import AudioConfig, SpectrogramConfig, AugmentationConfig, RobustnessTestConfig


class AudioPreprocessor:
    """Advanced audio preprocessing with complex-valued support."""
    
    def __init__(
        self,
        audio_config: AudioConfig,
        spectrogram_config: SpectrogramConfig,
        augmentation_config: Optional[AugmentationConfig] = None,
        robustness_config: Optional[RobustnessTestConfig] = None
    ):
        self.audio_config = audio_config
        self.spectrogram_config = spectrogram_config
        self.augmentation_config = augmentation_config
        self.robustness_config = robustness_config
        
        # Initialize mel-filterbank for consistent processing
        self.mel_fb = librosa.filters.mel(
            sr=audio_config.sample_rate,
            n_fft=spectrogram_config.n_fft,
            n_mels=spectrogram_config.n_mels,
            fmin=spectrogram_config.f_min,
            fmax=spectrogram_config.f_max or audio_config.sample_rate // 2,
            norm=spectrogram_config.norm
        )
        
        # Precompute window function
        self.window = scipy.signal.get_window(
            spectrogram_config.window,
            spectrogram_config.win_length or spectrogram_config.n_fft
        )
    
    def load_audio(
        self, 
        audio_path: Union[str, Path], 
        apply_robustness: bool = False,
        robustness_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            apply_robustness: Whether to apply robustness testing transformations
            robustness_params: Parameters for robustness testing
            
        Returns:
            Tuple of (audio, sample_rate)
        """
        try:
            # Load audio with librosa for consistency
            audio, sr = librosa.load(
                audio_path,
                sr=self.audio_config.sample_rate,
                mono=self.audio_config.mono,
                offset=self.audio_config.offset,
                duration=self.audio_config.duration
            )
            
            # Apply normalization
            audio = self._normalize_audio(audio)
            
            # Apply robustness transformations if requested
            if apply_robustness and robustness_params:
                audio = self._apply_robustness_transforms(audio, sr, robustness_params)
            
            return audio, sr
            
        except Exception as e:
            warnings.warn(f"Error loading audio {audio_path}: {e}")
            # Return silence as fallback
            duration = self.audio_config.duration or 5.0
            length = int(duration * self.audio_config.sample_rate)
            return np.zeros(length, dtype=np.float32), self.audio_config.sample_rate
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio according to configuration."""
        if not self.audio_config.normalize:
            return audio
        
        if self.audio_config.norm_type == 'peak':
            # Peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
                
        elif self.audio_config.norm_type == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / rms
                
        elif self.audio_config.norm_type == 'lufs':
            # LUFS normalization (simplified)
            # This is a simplified version; proper LUFS requires more complex filtering
            target_lufs = self.audio_config.target_level
            current_lufs = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-8)
            gain_db = target_lufs - current_lufs
            gain_linear = 10**(gain_db / 20)
            audio = audio * gain_linear
        
        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _apply_robustness_transforms(
        self, 
        audio: np.ndarray, 
        sr: int, 
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply robustness testing transformations."""
        
        # Add noise
        if 'noise_snr' in params:
            audio = self._add_noise(audio, params['noise_snr'], params.get('noise_type', 'white'))
        
        # Apply reverberation
        if 'rt60' in params:
            audio = self._add_reverberation(audio, sr, params['rt60'], params.get('room_size', 'medium'))
        
        # Speed perturbation
        if 'speed_factor' in params:
            audio = self._apply_speed_perturbation(audio, params['speed_factor'])
        
        # Pitch shift
        if 'pitch_shift' in params:
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=params['pitch_shift'])
        
        # Compression
        if 'compression' in params:
            audio = self._apply_compression(audio, params['compression'])
        
        return audio
    
    def _add_noise(self, audio: np.ndarray, snr_db: float, noise_type: str = 'white') -> np.ndarray:
        """Add noise to audio signal."""
        # Calculate signal power
        signal_power = np.mean(audio**2)
        
        # Generate noise
        if noise_type == 'white':
            noise = np.random.normal(0, 1, audio.shape)
        elif noise_type == 'pink':
            # Generate pink noise (1/f spectrum)
            noise = self._generate_pink_noise(len(audio))
        elif noise_type == 'brown':
            # Generate brown noise (1/f^2 spectrum)
            noise = self._generate_brown_noise(len(audio))
        else:
            noise = np.random.normal(0, 1, audio.shape)
        
        # Calculate noise power for desired SNR
        noise_power = np.mean(noise**2)
        noise_factor = np.sqrt(signal_power / (noise_power * 10**(snr_db/10)))
        
        # Add noise
        noisy_audio = audio + noise_factor * noise
        
        return noisy_audio
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f spectrum)."""
        # Generate white noise
        white_noise = np.random.normal(0, 1, length)
        
        # Apply 1/f filter in frequency domain
        fft = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(length)
        
        # Create 1/f filter (avoid division by zero)
        filter_response = 1.0 / (np.abs(freqs) + 1e-8)
        filter_response[0] = 1.0  # DC component
        
        # Apply filter
        filtered_fft = fft * filter_response
        pink_noise = np.real(np.fft.ifft(filtered_fft))
        
        # Normalize
        pink_noise = pink_noise / np.std(pink_noise)
        
        return pink_noise
    
    def _generate_brown_noise(self, length: int) -> np.ndarray:
        """Generate brown noise (1/f^2 spectrum)."""
        # Generate white noise
        white_noise = np.random.normal(0, 1, length)
        
        # Apply 1/f^2 filter
        fft = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(length)
        
        # Create 1/f^2 filter
        filter_response = 1.0 / (np.abs(freqs)**2 + 1e-8)
        filter_response[0] = 1.0  # DC component
        
        # Apply filter
        filtered_fft = fft * filter_response
        brown_noise = np.real(np.fft.ifft(filtered_fft))
        
        # Normalize
        brown_noise = brown_noise / np.std(brown_noise)
        
        return brown_noise
    
    def _add_reverberation(
        self, 
        audio: np.ndarray, 
        sr: int, 
        rt60: float, 
        room_size: str = 'medium'
    ) -> np.ndarray:
        """Add reverberation using pyroomacoustics."""
        try:
            # Define room dimensions based on size
            if room_size == 'small':
                room_dim = [4, 6, 3]  # Small room
            elif room_size == 'medium':
                room_dim = [8, 10, 4]  # Medium room
            elif room_size == 'large':
                room_dim = [15, 20, 5]  # Large room
            else:
                room_dim = [8, 10, 4]  # Default to medium
            
            # Create room with specified RT60
            room = pra.ShoeBox(
                room_dim,
                fs=sr,
                materials=pra.Material(energy_absorption="from_rt60", rt60_tgt=rt60),
                ray_tracing=True,
                air_absorption=True
            )
            
            # Add source and microphone
            source_pos = [room_dim[0]/4, room_dim[1]/2, room_dim[2]/2]
            mic_pos = [3*room_dim[0]/4, room_dim[1]/2, room_dim[2]/2]
            
            room.add_source(source_pos, signal=audio)
            room.add_microphone_array([mic_pos])
            
            # Simulate
            room.simulate()
            reverb_audio = room.mic_array.signals[0, :]
            
            # Normalize to prevent clipping
            if np.max(np.abs(reverb_audio)) > 0:
                reverb_audio = reverb_audio / np.max(np.abs(reverb_audio))
            
            return reverb_audio[:len(audio)]  # Ensure same length
            
        except Exception as e:
            warnings.warn(f"Reverberation simulation failed: {e}")
            return audio  # Return original audio if simulation fails
    
    def _apply_speed_perturbation(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """Apply speed perturbation (time stretching without pitch change)."""
        try:
            stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
            
            # Trim or pad to original length
            if len(stretched) > len(audio):
                stretched = stretched[:len(audio)]
            elif len(stretched) < len(audio):
                pad_length = len(audio) - len(stretched)
                stretched = np.pad(stretched, (0, pad_length), mode='constant')
            
            return stretched
            
        except Exception as e:
            warnings.warn(f"Speed perturbation failed: {e}")
            return audio
    
    def _apply_compression(self, audio: np.ndarray, compression_params: Dict[str, Any]) -> np.ndarray:
        """Apply audio compression effects."""
        # Simple dynamic range compression
        threshold = compression_params.get('threshold', -20)  # dB
        ratio = compression_params.get('ratio', 4.0)
        
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-8)
        
        # Apply compression
        compressed_db = np.where(
            audio_db > threshold,
            threshold + (audio_db - threshold) / ratio,
            audio_db
        )
        
        # Convert back to linear
        compressed_audio = np.sign(audio) * 10**(compressed_db / 20)
        
        return compressed_audio


class ComplexSpectrogramProcessor:
    """Processor for creating complex-valued spectrograms with phase information."""
    
    def __init__(self, spectrogram_config: SpectrogramConfig):
        self.config = spectrogram_config
        
        # Initialize STFT parameters
        self.stft_params = {
            'n_fft': spectrogram_config.n_fft,
            'hop_length': spectrogram_config.hop_length,
            'win_length': spectrogram_config.win_length,
            'window': spectrogram_config.window,
            'center': spectrogram_config.center,
            'pad_mode': spectrogram_config.pad_mode
        }
    
    def compute_complex_spectrogram(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Compute complex-valued spectrogram preserving phase information.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Complex spectrogram tensor of shape (time, frequency)
        """
        # Compute STFT to preserve phase
        stft = librosa.stft(audio, **self.stft_params)
        
        # Apply mel-filterbank to magnitude while preserving phase
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply mel-filtering to magnitude
        mel_magnitude = np.dot(
            librosa.filters.mel(
                sr=sr,
                n_fft=self.config.n_fft,
                n_mels=self.config.n_mels,
                fmin=self.config.f_min,
                fmax=self.config.f_max or sr // 2,
                norm=self.config.norm
            ),
            magnitude
        )
        
        # Interpolate phase to mel-frequency bins
        # For simplicity, we'll use linear interpolation
        mel_phase = self._interpolate_phase_to_mel(phase, mel_magnitude.shape[0])
        
        # Apply power scaling
        if self.config.power != 1.0:
            mel_magnitude = mel_magnitude ** (self.config.power / 2.0)
        
        # Convert to dB if requested
        if self.config.to_db:
            mel_magnitude = librosa.power_to_db(
                mel_magnitude,
                ref=self.config.ref,
                amin=self.config.amin,
                top_db=self.config.top_db
            )
        
        # Create complex spectrogram
        complex_spec = mel_magnitude * np.exp(1j * mel_phase)
        
        # Transpose to (time, frequency) and convert to tensor
        complex_spec = torch.tensor(complex_spec.T, dtype=torch.complex64)
        
        # Apply normalization
        if self.config.normalize_spec:
            complex_spec = self._normalize_complex_spectrogram(complex_spec)
        
        return complex_spec
    
    def _interpolate_phase_to_mel(self, phase: np.ndarray, n_mels: int) -> np.ndarray:
        """Interpolate phase information to mel-frequency bins."""
        # Simple linear interpolation of phase
        # In practice, more sophisticated methods could be used
        
        n_freqs, n_frames = phase.shape
        mel_phase = np.zeros((n_mels, n_frames))
        
        # Create interpolation indices
        mel_indices = np.linspace(0, n_freqs - 1, n_mels)
        
        for t in range(n_frames):
            # Interpolate phase for each time frame
            mel_phase[:, t] = np.interp(mel_indices, np.arange(n_freqs), phase[:, t])
        
        return mel_phase
    
    def _normalize_complex_spectrogram(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """Normalize complex spectrogram."""
        if self.config.spec_norm_type == 'instance':
            # Instance normalization on magnitude
            magnitude = torch.abs(complex_spec)
            phase = torch.angle(complex_spec)
            
            # Normalize magnitude
            mean_mag = torch.mean(magnitude)
            std_mag = torch.std(magnitude)
            normalized_magnitude = (magnitude - mean_mag) / (std_mag + 1e-8)
            
            # Reconstruct complex spectrogram
            complex_spec = normalized_magnitude * torch.exp(1j * phase)
            
        elif self.config.spec_norm_type == 'batch':
            # Batch normalization would be applied during training
            pass
            
        return complex_spec


class PhaseAugmentation:
    """Phase-aware augmentation techniques for complex spectrograms."""
    
    def __init__(self, augmentation_config: AugmentationConfig):
        self.config = augmentation_config
    
    def apply_phase_augmentation(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """Apply phase-aware augmentations."""
        if not self.config.phase_augmentation:
            return complex_spec
        
        # Phase noise addition
        complex_spec = self._add_phase_noise(complex_spec)
        
        # Phase shift
        complex_spec = self._apply_phase_shift(complex_spec)
        
        # Magnitude-phase decoupling
        complex_spec = self._magnitude_phase_shuffle(complex_spec)
        
        return complex_spec
    
    def _add_phase_noise(self, complex_spec: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
        """Add noise to phase while preserving magnitude."""
        magnitude = torch.abs(complex_spec)
        phase = torch.angle(complex_spec)
        
        # Add Gaussian noise to phase
        phase_noise = torch.normal(0, noise_std, phase.shape)
        noisy_phase = phase + phase_noise
        
        # Reconstruct complex spectrogram
        return magnitude * torch.exp(1j * noisy_phase)
    
    def _apply_phase_shift(self, complex_spec: torch.Tensor, max_shift: float = np.pi) -> torch.Tensor:
        """Apply random phase shift."""
        # Random phase shift
        shift = torch.uniform(-max_shift, max_shift, (1,))
        phase_shift = torch.exp(1j * shift)
        
        return complex_spec * phase_shift
    
    def _magnitude_phase_shuffle(self, complex_spec: torch.Tensor, prob: float = 0.1) -> torch.Tensor:
        """Randomly shuffle magnitude and phase relationships."""
        if torch.rand(1) < prob:
            magnitude = torch.abs(complex_spec)
            phase = torch.angle(complex_spec)
            
            # Shuffle phase along frequency dimension
            freq_dim = complex_spec.shape[1]
            shuffle_indices = torch.randperm(freq_dim)
            shuffled_phase = phase[:, shuffle_indices]
            
            return magnitude * torch.exp(1j * shuffled_phase)
        
        return complex_spec


class RobustnessTester:
    """Test model robustness under various acoustic conditions."""
    
    def __init__(self, robustness_config: RobustnessTestConfig):
        self.config = robustness_config
        self.preprocessor = None  # Will be set externally
    
    def generate_robustness_variants(
        self, 
        audio: np.ndarray, 
        sr: int
    ) -> Dict[str, np.ndarray]:
        """Generate multiple robustness test variants of the input audio."""
        variants = {'original': audio}
        
        # Noise variants
        if self.config.test_noise:
            for noise_type in self.config.noise_types:
                for snr in self.config.snr_levels:
                    key = f"noise_{noise_type}_snr{snr}"
                    variants[key] = self.preprocessor._add_noise(audio, snr, noise_type)
        
        # Reverberation variants
        if self.config.test_reverberation:
            for rt60 in self.config.rt60_values:
                for room_type in self.config.room_types:
                    if rt60 > 0:  # Skip RT60=0 (no reverb)
                        key = f"reverb_{room_type}_rt60_{rt60}"
                        variants[key] = self.preprocessor._add_reverberation(
                            audio, sr, rt60, room_type
                        )
        
        # Speed variants
        if self.config.test_speed:
            for speed_factor in self.config.speed_factors:
                key = f"speed_{speed_factor}"
                variants[key] = self.preprocessor._apply_speed_perturbation(audio, speed_factor)
        
        # Pitch variants
        if self.config.test_pitch:
            for pitch_shift in self.config.pitch_shifts:
                key = f"pitch_{pitch_shift}"
                variants[key] = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
        
        return variants


def create_preprocessing_pipeline(
    audio_config: AudioConfig,
    spectrogram_config: SpectrogramConfig,
    augmentation_config: Optional[AugmentationConfig] = None,
    robustness_config: Optional[RobustnessTestConfig] = None
) -> Tuple[AudioPreprocessor, ComplexSpectrogramProcessor, Optional[PhaseAugmentation]]:
    """Create complete preprocessing pipeline."""
    
    audio_processor = AudioPreprocessor(
        audio_config, spectrogram_config, augmentation_config, robustness_config
    )
    
    spectrogram_processor = ComplexSpectrogramProcessor(spectrogram_config)
    
    phase_augmenter = None
    if augmentation_config and augmentation_config.phase_augmentation:
        phase_augmenter = PhaseAugmentation(augmentation_config)
    
    return audio_processor, spectrogram_processor, phase_augmenter


def preprocess_audio_file(
    audio_path: Union[str, Path],
    audio_config: AudioConfig,
    spectrogram_config: SpectrogramConfig,
    apply_augmentation: bool = False,
    augmentation_config: Optional[AugmentationConfig] = None,
    robustness_params: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Complete preprocessing pipeline for a single audio file.
    
    Returns:
        Complex spectrogram tensor ready for Q-XAI model
    """
    # Create processors
    audio_processor, spec_processor, phase_augmenter = create_preprocessing_pipeline(
        audio_config, spectrogram_config, augmentation_config
    )
    
    # Load and preprocess audio
    audio, sr = audio_processor.load_audio(
        audio_path, 
        apply_robustness=robustness_params is not None,
        robustness_params=robustness_params
    )
    
    # Convert to complex spectrogram
    complex_spec = spec_processor.compute_complex_spectrogram(audio, sr)
    
    # Apply phase augmentation if requested
    if apply_augmentation and phase_augmenter:
        complex_spec = phase_augmenter.apply_phase_augmentation(complex_spec)
    
    return complex_spec