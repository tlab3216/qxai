"""
Visualization utilities for Q-XAI framework.
Provides comprehensive visualization for complex spectrograms, attention maps, uncertainty, and conformal prediction results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path
import librosa.display


class ComplexSpectrogramVisualizer:
    """Visualizer for complex-valued spectrograms and their components."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8'):
        self.figsize = figsize
        plt.style.use(style)
        self.cmap_magnitude = 'viridis'
        self.cmap_phase = 'twilight'
        
    def plot_complex_spectrogram(
        self,
        complex_spec: torch.Tensor,
        sr: int = 16000,
        hop_length: int = 160,
        title: str = "Complex Spectrogram",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot complex spectrogram showing magnitude and phase.
        
        Args:
            complex_spec: Complex spectrogram tensor (time, freq)
            sr: Sample rate
            hop_length: Hop length for time axis
            title: Plot title
            save_path: Optional path to save figure
        """
        # Convert to numpy and transpose for librosa display format
        if torch.is_complex(complex_spec):
            magnitude = torch.abs(complex_spec).cpu().numpy().T
            phase = torch.angle(complex_spec).cpu().numpy().T
        else:
            magnitude = complex_spec.cpu().numpy().T
            phase = np.zeros_like(magnitude)
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Time and frequency axes
        times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length)
        freqs = np.arange(magnitude.shape[0])
        
        # Magnitude spectrogram
        im1 = axes[0, 0].imshow(
            magnitude, aspect='auto', origin='lower', 
            cmap=self.cmap_magnitude, extent=[times[0], times[-1], freqs[0], freqs[-1]]
        )
        axes[0, 0].set_title('Magnitude')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Mel Bin')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Phase spectrogram
        im2 = axes[0, 1].imshow(
            phase, aspect='auto', origin='lower',
            cmap=self.cmap_phase, extent=[times[0], times[-1], freqs[0], freqs[-1]],
            vmin=-np.pi, vmax=np.pi
        )
        axes[0, 1].set_title('Phase')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Mel Bin')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Real part
        real_part = magnitude * np.cos(phase)
        im3 = axes[1, 0].imshow(
            real_part, aspect='auto', origin='lower',
            cmap='RdBu_r', extent=[times[0], times[-1], freqs[0], freqs[-1]]
        )
        axes[1, 0].set_title('Real Part')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Mel Bin')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Imaginary part
        imag_part = magnitude * np.sin(phase)
        im4 = axes[1, 1].imshow(
            imag_part, aspect='auto', origin='lower',
            cmap='RdBu_r', extent=[times[0], times[-1], freqs[0], freqs[-1]]
        )
        axes[1, 1].set_title('Imaginary Part')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Mel Bin')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_phase_evolution(
        self,
        complex_spec: torch.Tensor,
        freq_bins: List[int],
        sr: int = 16000,
        hop_length: int = 160,
        title: str = "Phase Evolution"
    ) -> plt.Figure:
        """Plot phase evolution over time for specific frequency bins."""
        phase = torch.angle(complex_spec).cpu().numpy()
        times = librosa.frames_to_time(np.arange(phase.shape[0]), sr=sr, hop_length=hop_length)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, freq_bin in enumerate(freq_bins):
            if freq_bin < phase.shape[1]:
                ax.plot(times, phase[:, freq_bin], 
                       label=f'Mel bin {freq_bin}', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase (radians)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-np.pi, np.pi])
        
        return fig
    
    def plot_interference_pattern(
        self,
        complex_spec1: torch.Tensor,
        complex_spec2: torch.Tensor,
        title: str = "Interference Pattern"
    ) -> plt.Figure:
        """Visualize interference between two complex spectrograms."""
        # Compute interference
        interference = complex_spec1 + complex_spec2
        
        magnitude1 = torch.abs(complex_spec1).cpu().numpy().T
        magnitude2 = torch.abs(complex_spec2).cpu().numpy().T
        interference_mag = torch.abs(interference).cpu().numpy().T
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)
        
        # Individual magnitudes and interference
        im1 = axes[0].imshow(magnitude1, aspect='auto', origin='lower', cmap=self.cmap_magnitude)
        axes[0].set_title('Signal 1 Magnitude')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(magnitude2, aspect='auto', origin='lower', cmap=self.cmap_magnitude)
        axes[1].set_title('Signal 2 Magnitude')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(interference_mag, aspect='auto', origin='lower', cmap=self.cmap_magnitude)
        axes[2].set_title('Interference Magnitude')
        plt.colorbar(im3, ax=axes[2])
        
        for ax in axes:
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Mel Bin')
        
        plt.tight_layout()
        return fig


class QISAVisualizer:
    """Visualizer for Quantum-Inspired State Attribution (QISA) results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_attribution_map(
        self,
        spectrogram: torch.Tensor,
        attribution: torch.Tensor,
        sr: int = 16000,
        hop_length: int = 160,
        title: str = "QISA Attribution Map",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot QISA attribution map overlaid on spectrogram.
        
        Args:
            spectrogram: Input spectrogram (time, freq)
            attribution: Attribution map (time, freq)
            sr: Sample rate
            hop_length: Hop length
            title: Plot title
            save_path: Optional save path
        """
        # Convert to numpy and transpose for display
        if torch.is_complex(spectrogram):
            spec_display = torch.abs(spectrogram).cpu().numpy().T
        else:
            spec_display = spectrogram.cpu().numpy().T
            
        attr_display = attribution.cpu().numpy().T
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=16)
        
        # Time axis
        times = librosa.frames_to_time(np.arange(spec_display.shape[1]), sr=sr, hop_length=hop_length)
        freqs = np.arange(spec_display.shape[0])
        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        
        # Original spectrogram
        im1 = axes[0].imshow(spec_display, aspect='auto', origin='lower', 
                            cmap='viridis', extent=extent)
        axes[0].set_title('Input Spectrogram')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Mel Bin')
        plt.colorbar(im1, ax=axes[0])
        
        # Attribution map
        im2 = axes[1].imshow(attr_display, aspect='auto', origin='lower', 
                            cmap='hot', extent=extent)
        axes[1].set_title('QISA Attribution')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Mel Bin')
        plt.colorbar(im2, ax=axes[1])
        
        # Overlay
        # Normalize attribution for overlay
        attr_norm = (attr_display - attr_display.min()) / (attr_display.max() - attr_display.min() + 1e-8)
        
        axes[2].imshow(spec_display, aspect='auto', origin='lower', 
                      cmap='viridis', extent=extent, alpha=0.7)
        im3 = axes[2].imshow(attr_norm, aspect='auto', origin='lower', 
                            cmap='Reds', extent=extent, alpha=0.6)
        axes[2].set_title('Attribution Overlay')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Mel Bin')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_wirtinger_components(
        self,
        grad_z: torch.Tensor,
        grad_z_conj: torch.Tensor,
        title: str = "Wirtinger Gradient Components"
    ) -> plt.Figure:
        """Plot holomorphic and anti-holomorphic gradient components."""
        grad_z_mag = torch.abs(grad_z).cpu().numpy().T
        grad_z_conj_mag = torch.abs(grad_z_conj).cpu().numpy().T
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)
        
        # Holomorphic component
        im1 = axes[0].imshow(grad_z_mag, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('|∂L/∂z| (Holomorphic)')
        axes[0].set_xlabel('Time Frames')
        axes[0].set_ylabel('Mel Bin')
        plt.colorbar(im1, ax=axes[0])
        
        # Anti-holomorphic component
        im2 = axes[1].imshow(grad_z_conj_mag, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('|∂L/∂z*| (Anti-holomorphic)')
        axes[1].set_xlabel('Time Frames')
        axes[1].set_ylabel('Mel Bin')
        plt.colorbar(im2, ax=axes[1])
        
        # Combined attribution
        combined = grad_z_mag**2 + grad_z_conj_mag**2
        im3 = axes[2].imshow(combined, aspect='auto', origin='lower', cmap='hot')
        axes[2].set_title('Combined Attribution')
        axes[2].set_xlabel('Time Frames')
        axes[2].set_ylabel('Mel Bin')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        return fig
    
    def plot_attribution_comparison(
        self,
        spectrogram: torch.Tensor,
        qisa_attr: torch.Tensor,
        baseline_attr: torch.Tensor,
        baseline_name: str = "Grad-CAM",
        title: str = "Attribution Comparison"
    ) -> plt.Figure:
        """Compare QISA attribution with baseline method."""
        spec_display = torch.abs(spectrogram).cpu().numpy().T if torch.is_complex(spectrogram) else spectrogram.cpu().numpy().T
        qisa_display = qisa_attr.cpu().numpy().T
        baseline_display = baseline_attr.cpu().numpy().T
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Input spectrogram
        im1 = axes[0, 0].imshow(spec_display, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Input Spectrogram')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # QISA attribution
        im2 = axes[0, 1].imshow(qisa_display, aspect='auto', origin='lower', cmap='hot')
        axes[0, 1].set_title('QISA Attribution')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Baseline attribution
        im3 = axes[1, 0].imshow(baseline_display, aspect='auto', origin='lower', cmap='hot')
        axes[1, 0].set_title(f'{baseline_name} Attribution')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Difference map
        diff = qisa_display - baseline_display
        im4 = axes[1, 1].imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[1, 1].set_title('Difference (QISA - Baseline)')
        plt.colorbar(im4, ax=axes[1, 1])
        
        for ax in axes.flat:
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Mel Bin')
        
        plt.tight_layout()
        return fig


class AUQVisualizer:
    """Visualizer for Amplitude-based Uncertainty Quantification (AUQ) results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_uncertainty_decomposition(
        self,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
        covariance: torch.Tensor,
        class_names: Optional[List[str]] = None,
        title: str = "AUQ Uncertainty Decomposition"
    ) -> plt.Figure:
        """
        Plot the three-component uncertainty decomposition.
        
        Args:
            epistemic: Epistemic uncertainty per class
            aleatoric: Aleatoric uncertainty per class
            covariance: Covariance uncertainty per class (novel)
            class_names: Optional class names
            title: Plot title
        """
        # Convert to numpy
        epistemic_np = epistemic.cpu().numpy()
        aleatoric_np = aleatoric.cpu().numpy()
        covariance_np = covariance.cpu().numpy()
        
        n_classes = len(epistemic_np)
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        x = np.arange(n_classes)
        width = 0.25
        
        # Individual uncertainty components
        axes[0, 0].bar(x, epistemic_np, width, label='Epistemic', alpha=0.8, color='blue')
        axes[0, 0].set_title('Epistemic Uncertainty')
        axes[0, 0].set_ylabel('Uncertainty Value')
        if n_classes <= 10:
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(class_names, rotation=45)
        
        axes[0, 1].bar(x, aleatoric_np, width, label='Aleatoric', alpha=0.8, color='orange')
        axes[0, 1].set_title('Aleatoric Uncertainty')
        axes[0, 1].set_ylabel('Uncertainty Value')
        if n_classes <= 10:
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(class_names, rotation=45)
        
        axes[1, 0].bar(x, covariance_np, width, label='Covariance', alpha=0.8, color='green')
        axes[1, 0].set_title('Covariance Uncertainty (Novel)')
        axes[1, 0].set_ylabel('Uncertainty Value')
        if n_classes <= 10:
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(class_names, rotation=45)
        
        # Combined stacked bar plot
        axes[1, 1].bar(x - width, epistemic_np, width, label='Epistemic', alpha=0.8, color='blue')
        axes[1, 1].bar(x, aleatoric_np, width, label='Aleatoric', alpha=0.8, color='orange')
        axes[1, 1].bar(x + width, covariance_np, width, label='Covariance', alpha=0.8, color='green')
        axes[1, 1].set_title('Combined Uncertainty Decomposition')
        axes[1, 1].set_ylabel('Uncertainty Value')
        axes[1, 1].legend()
        if n_classes <= 10:
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(class_names, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_reliability_diagram(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        bin_centers: np.ndarray,
        bin_accuracies: np.ndarray,
        bin_counts: np.ndarray,
        ece: float,
        title: str = "Reliability Diagram"
    ) -> plt.Figure:
        """Plot reliability diagram for calibration assessment."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle(f'{title} (ECE: {ece:.4f})', fontsize=16)
        
        # Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
        ax1.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, 
               edgecolor='black', label='Actual')
        ax1.plot(bin_centers, bin_centers, 'ro-', label='Expected', markersize=6)
        
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Sample distribution
        ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, color='gray')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_uncertainty_vs_accuracy(
        self,
        total_uncertainty: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10,
        title: str = "Uncertainty vs Accuracy"
    ) -> plt.Figure:
        """Plot relationship between uncertainty and accuracy."""
        uncertainty_np = total_uncertainty.cpu().numpy()
        correct = (predictions == targets).cpu().numpy().astype(float)
        
        # Create uncertainty bins
        sorted_indices = np.argsort(uncertainty_np)
        bin_size = len(uncertainty_np) // n_bins
        
        bin_uncertainties = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(uncertainty_np)
            
            bin_indices = sorted_indices[start_idx:end_idx]
            bin_uncertainty = np.mean(uncertainty_np[bin_indices])
            bin_accuracy = np.mean(correct[bin_indices])
            bin_count = len(bin_indices)
            
            bin_uncertainties.append(bin_uncertainty)
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(bin_count)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Uncertainty vs Accuracy
        ax1.plot(bin_uncertainties, bin_accuracies, 'bo-', markersize=8, linewidth=2)
        ax1.set_xlabel('Average Uncertainty')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Uncertainty vs Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_coef = np.corrcoef(bin_uncertainties, bin_accuracies)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # Sample distribution across uncertainty bins
        ax2.bar(range(n_bins), bin_counts, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Uncertainty Bin')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Distribution Across Uncertainty Bins')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class QICPVisualizer:
    """Visualizer for Quantum-Inspired Conformal Prediction (QICP) results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_coverage_vs_set_size(
        self,
        confidence_levels: List[float],
        coverages: List[float],
        set_sizes: List[float],
        target_coverage: float = 0.9,
        title: str = "QICP Coverage vs Set Size"
    ) -> plt.Figure:
        """Plot coverage and set size relationship."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Coverage plot
        ax1.plot(confidence_levels, coverages, 'bo-', linewidth=2, markersize=6, label='Actual Coverage')
        ax1.axhline(y=target_coverage, color='r', linestyle='--', linewidth=2, label=f'Target ({target_coverage})')
        ax1.set_xlabel('Confidence Level')
        ax1.set_ylabel('Empirical Coverage')
        ax1.set_title('Coverage vs Confidence Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Set size plot
        ax2.plot(confidence_levels, set_sizes, 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel('Confidence Level')
        ax2.set_ylabel('Average Set Size')
        ax2.set_title('Set Size vs Confidence Level')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_sets_distribution(
        self,
        set_sizes: List[int],
        class_names: Optional[List[str]] = None,
        title: str = "Prediction Set Size Distribution"
    ) -> plt.Figure:
        """Plot distribution of prediction set sizes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Histogram of set sizes
        ax1.hist(set_sizes, bins=range(1, max(set_sizes) + 2), alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Prediction Set Size')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Set Sizes')
        ax1.grid(True, alpha=0.3)
        
        # Statistics
        mean_size = np.mean(set_sizes)
        std_size = np.std(set_sizes)
        median_size = np.median(set_sizes)
        
        stats_text = f'Mean: {mean_size:.2f}\nStd: {std_size:.2f}\nMedian: {median_size:.1f}'
        ax1.text(0.7, 0.8, stats_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # Cumulative distribution
        sorted_sizes = np.sort(set_sizes)
        y_values = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
        ax2.plot(sorted_sizes, y_values, 'g-', linewidth=2)
        ax2.set_xlabel('Prediction Set Size')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_nonconformity_scores(
        self,
        scores: np.ndarray,
        threshold: float,
        accuracies_dict: Dict[str, np.ndarray],
        perturbation_levels: np.ndarray,
        title: str = "Nonconformity Scores Distribution"
    ) -> plt.Figure:
        """Plot distribution of nonconformity scores."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram of scores
        ax.hist(scores, bins=50, alpha=0.7, color='lightblue', edgecolor='black', density=True)
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'Threshold: {threshold:.3f}')
        
        ax.set_xlabel('Nonconformity Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight best performing method
        if len(accuracies_dict) > 1:
            # Find method with highest area under curve
            best_method = max(accuracies_dict.keys(), 
                            key=lambda x: np.trapz(accuracies_dict[x], perturbation_levels))
            ax.text(0.02, 0.98, f'Best: {best_method}', transform=ax.transAxes,
                   bbox=dict(boxstyle="round", facecolor='lightgreen'), 
                   verticalalignment='top')
        
        return fig
    
    def plot_noise_analysis(
        self,
        clean_spectrogram: torch.Tensor,
        noisy_spectrogram: torch.Tensor,
        qisa_clean: torch.Tensor,
        qisa_noisy: torch.Tensor,
        snr_db: float,
        title: str = "Noise Robustness Analysis"
    ) -> plt.Figure:
        """Analyze model behavior under noise using QISA."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{title} (SNR: {snr_db} dB)', fontsize=16)
        
        # Convert to display format
        clean_spec = torch.abs(clean_spectrogram).cpu().numpy().T if torch.is_complex(clean_spectrogram) else clean_spectrogram.cpu().numpy().T
        noisy_spec = torch.abs(noisy_spectrogram).cpu().numpy().T if torch.is_complex(noisy_spectrogram) else noisy_spectrogram.cpu().numpy().T
        qisa_clean_disp = qisa_clean.cpu().numpy().T
        qisa_noisy_disp = qisa_noisy.cpu().numpy().T
        
        # Top row: Spectrograms
        im1 = axes[0, 0].imshow(clean_spec, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Clean Signal')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(noisy_spec, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title(f'Noisy Signal (SNR: {snr_db} dB)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Noise visualization
        noise = noisy_spec - clean_spec
        im3 = axes[0, 2].imshow(noise, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[0, 2].set_title('Noise Component')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Bottom row: QISA attributions
        im4 = axes[1, 0].imshow(qisa_clean_disp, aspect='auto', origin='lower', cmap='hot')
        axes[1, 0].set_title('QISA (Clean)')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(qisa_noisy_disp, aspect='auto', origin='lower', cmap='hot')
        axes[1, 1].set_title('QISA (Noisy)')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Attribution difference
        attr_diff = qisa_noisy_disp - qisa_clean_disp
        im6 = axes[1, 2].imshow(attr_diff, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[1, 2].set_title('Attribution Shift')
        plt.colorbar(im6, ax=axes[1, 2])
        
        for ax in axes.flat:
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Mel Bin')
        
        plt.tight_layout()
        return fig


class InteractiveVisualizer:
    """Interactive visualizations using Plotly."""
    
    def __init__(self):
        pass
    
    def create_interactive_spectrogram(
        self,
        complex_spec: torch.Tensor,
        sr: int = 16000,
        hop_length: int = 160,
        title: str = "Interactive Complex Spectrogram"
    ) -> go.Figure:
        """Create interactive complex spectrogram with magnitude and phase."""
        # Convert to numpy
        magnitude = torch.abs(complex_spec).cpu().numpy()
        phase = torch.angle(complex_spec).cpu().numpy()
        
        # Create time and frequency axes
        times = librosa.frames_to_time(np.arange(magnitude.shape[0]), sr=sr, hop_length=hop_length)
        freqs = np.arange(magnitude.shape[1])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Magnitude', 'Phase', 'Real Part', 'Imaginary Part'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Magnitude
        fig.add_trace(
            go.Heatmap(z=magnitude.T, x=times, y=freqs, colorscale='Viridis', name='Magnitude'),
            row=1, col=1
        )
        
        # Phase
        fig.add_trace(
            go.Heatmap(z=phase.T, x=times, y=freqs, colorscale='Twilight', 
                      zmin=-np.pi, zmax=np.pi, name='Phase'),
            row=1, col=2
        )
        
        # Real part
        real_part = magnitude * np.cos(phase)
        fig.add_trace(
            go.Heatmap(z=real_part.T, x=times, y=freqs, colorscale='RdBu', name='Real'),
            row=2, col=1
        )
        
        # Imaginary part
        imag_part = magnitude * np.sin(phase)
        fig.add_trace(
            go.Heatmap(z=imag_part.T, x=times, y=freqs, colorscale='RdBu', name='Imaginary'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        # Update axis labels
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Time (s)", row=i, col=j)
                fig.update_yaxes(title_text="Mel Bin", row=i, col=j)
        
        return fig
    
    def create_interactive_attribution(
        self,
        spectrogram: torch.Tensor,
        attribution: torch.Tensor,
        class_names: Optional[List[str]] = None,
        predicted_class: Optional[int] = None,
        title: str = "Interactive QISA Attribution"
    ) -> go.Figure:
        """Create interactive attribution visualization."""
        # Convert to numpy
        spec_display = torch.abs(spectrogram).cpu().numpy() if torch.is_complex(spectrogram) else spectrogram.cpu().numpy()
        attr_display = attribution.cpu().numpy()
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Input Spectrogram', 'QISA Attribution', 'Attribution Overlay'),
            horizontal_spacing=0.1
        )
        
        times = np.arange(spec_display.shape[0])
        freqs = np.arange(spec_display.shape[1])
        
        # Input spectrogram
        fig.add_trace(
            go.Heatmap(z=spec_display.T, x=times, y=freqs, colorscale='Viridis', 
                      name='Spectrogram', showscale=True),
            row=1, col=1
        )
        
        # Attribution
        fig.add_trace(
            go.Heatmap(z=attr_display.T, x=times, y=freqs, colorscale='Hot', 
                      name='Attribution', showscale=True),
            row=1, col=2
        )
        
        # Overlay (normalize attribution for better visualization)
        attr_norm = (attr_display - attr_display.min()) / (attr_display.max() - attr_display.min() + 1e-8)
        overlay = spec_display * 0.7 + attr_norm * 0.3
        
        fig.add_trace(
            go.Heatmap(z=overlay.T, x=times, y=freqs, colorscale='Viridis', 
                      name='Overlay', showscale=True),
            row=1, col=3
        )
        
        # Update layout
        title_text = title
        if predicted_class is not None and class_names is not None:
            title_text += f" (Predicted: {class_names[predicted_class]})"
        
        fig.update_layout(
            title=title_text,
            height=500,
            showlegend=False
        )
        
        # Update axis labels
        for i in range(1, 4):
            fig.update_xaxes(title_text="Time Frames", row=1, col=i)
            fig.update_yaxes(title_text="Mel Bin", row=1, col=i)
        
        return fig
    
    def create_uncertainty_dashboard(
        self,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
        covariance: torch.Tensor,
        class_names: Optional[List[str]] = None,
        title: str = "AUQ Uncertainty Dashboard"
    ) -> go.Figure:
        """Create interactive uncertainty dashboard."""
        # Convert to numpy
        epistemic_np = epistemic.cpu().numpy()
        aleatoric_np = aleatoric.cpu().numpy()
        covariance_np = covariance.cpu().numpy()
        
        n_classes = len(epistemic_np)
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Epistemic Uncertainty', 'Aleatoric Uncertainty', 
                          'Covariance Uncertainty', 'Combined View'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Individual uncertainty components
        fig.add_trace(
            go.Bar(x=class_names, y=epistemic_np, name='Epistemic', marker_color='blue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=class_names, y=aleatoric_np, name='Aleatoric', marker_color='orange'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=class_names, y=covariance_np, name='Covariance', marker_color='green'),
            row=2, col=1
        )
        
        # Combined view (grouped bars)
        fig.add_trace(
            go.Bar(x=class_names, y=epistemic_np, name='Epistemic', marker_color='blue',
                  offsetgroup=1),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(x=class_names, y=aleatoric_np, name='Aleatoric', marker_color='orange',
                  offsetgroup=2),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(x=class_names, y=covariance_np, name='Covariance', marker_color='green',
                  offsetgroup=3),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig


class ComprehensiveReportGenerator:
    """Generate comprehensive visual reports for Q-XAI evaluation."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize individual visualizers
        self.spec_viz = ComplexSpectrogramVisualizer()
        self.qisa_viz = QISAVisualizer()
        self.auq_viz = AUQVisualizer()
        self.qicp_viz = QICPVisualizer()
        self.robust_viz = RobustnessVisualizer()
        self.interactive_viz = InteractiveVisualizer()
    
    def generate_model_performance_report(
        self,
        results: Dict[str, Any],
        dataset_name: str,
        save_plots: bool = True
    ) -> List[plt.Figure]:
        """Generate comprehensive model performance report."""
        figures = []
        
        # 1. Classification results
        if 'confusion_matrix' in results:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                       cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {dataset_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            figures.append(fig)
            
            if save_plots:
                fig.savefig(self.output_dir / f'{dataset_name}_confusion_matrix.png', 
                           dpi=300, bbox_inches='tight')
        
        # 2. Performance metrics
        if 'classification_report' in results:
            report = results['classification_report']
            
            # Extract metrics for visualization
            classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            precisions = [report[c]['precision'] for c in classes]
            recalls = [report[c]['recall'] for c in classes]
            f1s = [report[c]['f1-score'] for c in classes]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(classes))
            width = 0.25
            
            ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
            ax.bar(x, recalls, width, label='Recall', alpha=0.8)
            ax.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('Score')
            ax.set_title(f'Per-Class Performance - {dataset_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            figures.append(fig)
            
            if save_plots:
                fig.savefig(self.output_dir / f'{dataset_name}_per_class_metrics.png', 
                           dpi=300, bbox_inches='tight')
        
        return figures
    
    def generate_interpretability_report(
        self,
        sample_data: Dict[str, torch.Tensor],
        attribution_results: Dict[str, torch.Tensor],
        faithfulness_metrics: Dict[str, float],
        save_plots: bool = True
    ) -> List[plt.Figure]:
        """Generate interpretability analysis report."""
        figures = []
        
        # 1. Sample attribution visualization
        if 'spectrogram' in sample_data and 'qisa_attribution' in attribution_results:
            fig = self.qisa_viz.plot_attribution_map(
                sample_data['spectrogram'],
                attribution_results['qisa_attribution'],
                title="Sample QISA Attribution Analysis"
            )
            figures.append(fig)
            
            if save_plots:
                fig.savefig(self.output_dir / 'sample_qisa_attribution.png', 
                           dpi=300, bbox_inches='tight')
        
        # 2. Wirtinger components visualization
        if 'grad_z' in attribution_results and 'grad_z_conj' in attribution_results:
            fig = self.qisa_viz.plot_wirtinger_components(
                attribution_results['grad_z'],
                attribution_results['grad_z_conj'],
                title="Wirtinger Gradient Components"
            )
            figures.append(fig)
            
            if save_plots:
                fig.savefig(self.output_dir / 'wirtinger_components.png', 
                           dpi=300, bbox_inches='tight')
        
        # 3. Faithfulness metrics visualization
        if faithfulness_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics_names = list(faithfulness_metrics.keys())
            metrics_values = list(faithfulness_metrics.values())
            
            bars = ax.bar(metrics_names, metrics_values, alpha=0.7, color='skyblue')
            ax.set_ylabel('Metric Value')
            ax.set_title('Interpretability Faithfulness Metrics')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
            
            figures.append(fig)
            
            if save_plots:
                fig.savefig(self.output_dir / 'faithfulness_metrics.png', 
                           dpi=300, bbox_inches='tight')
        
        return figures
    
    def generate_uncertainty_report(
        self,
        uncertainty_data: Dict[str, torch.Tensor],
        calibration_metrics: Dict[str, float],
        class_names: Optional[List[str]] = None,
        save_plots: bool = True
    ) -> List[plt.Figure]:
        """Generate uncertainty quantification report."""
        figures = []
        
        # 1. Uncertainty decomposition
        if all(key in uncertainty_data for key in ['epistemic', 'aleatoric', 'covariance']):
            fig = self.auq_viz.plot_uncertainty_decomposition(
                uncertainty_data['epistemic'],
                uncertainty_data['aleatoric'],
                uncertainty_data['covariance'],
                class_names=class_names,
                title="AUQ Uncertainty Decomposition Analysis"
            )
            figures.append(fig)
            
            if save_plots:
                fig.savefig(self.output_dir / 'uncertainty_decomposition.png', 
                           dpi=300, bbox_inches='tight')
        
        # 2. Calibration analysis
        if 'reliability_data' in uncertainty_data:
            rel_data = uncertainty_data['reliability_data']
            fig = self.auq_viz.plot_reliability_diagram(
                rel_data['confidences'],
                rel_data['accuracies'],
                rel_data['bin_centers'],
                rel_data['bin_accuracies'],
                rel_data['bin_counts'],
                calibration_metrics.get('ece', 0.0),
                title="Model Calibration Analysis"
            )
            figures.append(fig)
            
            if save_plots:
                fig.savefig(self.output_dir / 'calibration_analysis.png', 
                           dpi=300, bbox_inches='tight')
        
        return figures
    
    def generate_robustness_report(
        self,
        robustness_results: Dict[str, Dict[str, float]],
        save_plots: bool = True
    ) -> List[plt.Figure]:
        """Generate robustness analysis report."""
        figures = []
        
        # Generate robustness curves for different perturbation types
        for perturbation_type, results in robustness_results.items():
            if isinstance(results, dict) and len(results) > 1:
                # Extract perturbation levels and accuracies
                levels = np.array(list(results.keys()))
                accuracies = {'Q-XAI': np.array(list(results.values()))}
                
                fig = self.robust_viz.plot_robustness_curves(
                    levels, accuracies, 
                    perturbation_type=perturbation_type,
                    title=f"Robustness to {perturbation_type}"
                )
                figures.append(fig)
                
                if save_plots:
                    safe_name = perturbation_type.replace(' ', '_').replace('(', '').replace(')', '')
                    fig.savefig(self.output_dir / f'robustness_{safe_name}.png', 
                               dpi=300, bbox_inches='tight')
        
        return figures
    
    def generate_complete_report(
        self,
        all_results: Dict[str, Any],
        dataset_name: str = "evaluation",
        save_html: bool = True
    ) -> str:
        """Generate complete HTML report with all visualizations."""
        # Generate all figure types
        performance_figs = []
        interpretability_figs = []
        uncertainty_figs = []
        robustness_figs = []
        
        if 'performance' in all_results:
            performance_figs = self.generate_model_performance_report(
                all_results['performance'], dataset_name
            )
        
        if 'interpretability' in all_results:
            interpretability_figs = self.generate_interpretability_report(
                all_results['interpretability'].get('sample_data', {}),
                all_results['interpretability'].get('attribution_results', {}),
                all_results['interpretability'].get('faithfulness_metrics', {})
            )
        
        if 'uncertainty' in all_results:
            uncertainty_figs = self.generate_uncertainty_report(
                all_results['uncertainty'].get('uncertainty_data', {}),
                all_results['uncertainty'].get('calibration_metrics', {})
            )
        
        if 'robustness' in all_results:
            robustness_figs = self.generate_robustness_report(
                all_results['robustness']
            )
        
        # Close all figures to save memory
        for fig_list in [performance_figs, interpretability_figs, uncertainty_figs, robustness_figs]:
            for fig in fig_list:
                plt.close(fig)
        
        # Generate HTML report
        if save_html:
            html_content = self._generate_html_report(all_results, dataset_name)
            html_path = self.output_dir / f'{dataset_name}_complete_report.html'
            with open(html_path, 'w') as f:
                f.write(html_content)
            return str(html_path)
        
        return ""
    
    def _generate_html_report(self, results: Dict[str, Any], dataset_name: str) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Q-XAI Evaluation Report - {dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .figure {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Q-XAI Evaluation Report</h1>
                <h2>Dataset: {dataset_name}</h2>
                <p>Generated using Q-XAI framework for interpretable complex-valued transformers</p>
            </div>
        """
        
        # Add performance metrics
        if 'performance' in results:
            html += """
            <div class="section">
                <h2>Performance Metrics</h2>
            """
            
            perf = results['performance']
            if 'accuracy' in perf:
                html += f'<div class="metric"><strong>Accuracy:</strong> {perf["accuracy"]:.4f}</div>'
            if 'macro_f1' in perf:
                html += f'<div class="metric"><strong>Macro F1:</strong> {perf["macro_f1"]:.4f}</div>'
            if 'ece' in perf:
                html += f'<div class="metric"><strong>ECE:</strong> {perf["ece"]:.4f}</div>'
            
            html += "</div>"
        
        # Add interpretability section
        if 'interpretability' in results:
            html += """
            <div class="section">
                <h2>Interpretability Analysis (QISA)</h2>
                <p>Quantum-Inspired State Attribution using Wirtinger calculus for faithful explanations of complex-valued models.</p>
            </div>
            """
        
        # Add uncertainty section
        if 'uncertainty' in results:
            html += """
            <div class="section">
                <h2>Uncertainty Quantification (AUQ)</h2>
                <p>Amplitude-based uncertainty decomposition into epistemic, aleatoric, and novel covariance components.</p>
            </div>
            """
        
        # Add robustness section
        if 'robustness' in results:
            html += """
            <div class="section">
                <h2>Robustness Analysis</h2>
                <p>Model performance under various acoustic perturbations including noise, reverberation, and compression.</p>
            </div>
            """
        
        html += """
            <div class="section">
                <h2>Generated Figures</h2>
                <p>All visualization figures have been saved to the reports directory.</p>
            </div>
        </body>
        </html>
        """
        
        return html


# Utility functions for visualization

def save_attention_visualization(
    attention_weights: torch.Tensor,
    input_sequence: torch.Tensor,
    output_path: str,
    head_idx: int = 0,
    layer_idx: int = -1
) -> None:
    """Save attention weight visualization."""
    # Extract specific head and layer
    if attention_weights.dim() == 4:  # (batch, heads, seq_len, seq_len)
        attn = attention_weights[0, head_idx].cpu().numpy()
    else:
        attn = attention_weights.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap='Blues', aspect='auto')
    ax.set_title(f'Attention Weights - Head {head_idx}, Layer {layer_idx}')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_comparison_plot(
    results_dict: Dict[str, Dict[str, float]],
    metric_name: str,
    title: str,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Create comparison plot for multiple methods."""
    methods = list(results_dict.keys())
    values = [results_dict[method][metric_name] for method in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, values, alpha=0.7, color='skyblue')
    
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot training and validation learning curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curves - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Learning Curves - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig 


class RobustnessVisualizer:
    """Visualizer for robustness evaluation results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_robustness_curves(
        self,
        perturbation_levels: np.ndarray,
        accuracies_dict: Dict[str, np.ndarray],
        perturbation_type: str = "SNR (dB)",
        title: str = "Robustness Analysis"
    ) -> plt.Figure:
        """
        Plot robustness curves for different methods.
        
        Args:
            perturbation_levels: Array of perturbation levels (e.g., SNR levels)
            accuracies_dict: Dictionary mapping method names to accuracy arrays
            perturbation_type: Type of perturbation (for x-axis label)
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(accuracies_dict)))
        
        for (method_name, accuracies), color in zip(accuracies_dict.items(), colors):
            ax.plot(perturbation_levels, accuracies, 'o-', 
                   label=method_name, linewidth=2, markersize=6, color=color)
        
        ax.set_xlabel(perturbation_type)
        ax.set_ylabel('Classification Accuracy (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        for method_name, accuracies in accuracies_dict.items():
            mean_score = np.mean(accuracies)
            std_score = np.std(accuracies)
            stats_text = f'Mean: {mean_score:.3f}\nStd: {std_score:.3f}'
            ax.text(0.7, 0.8, stats_text, transform=ax.transAxes,
                   bbox=dict(boxstyle="round", facecolor='wheat'))

        return fig