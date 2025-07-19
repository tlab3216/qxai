"""
Amplitude-based Uncertainty Quantification (AUQ) for Q-XAI framework.
Implements three-component uncertainty decomposition for complex-valued models.
Novel covariance uncertainty captures phase relationship stability.
"""

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
import warnings
from dataclasses import dataclass
import time

from utils.complex_math import compute_covariance_uncertainty, complex_dropout
from config.model_config import AUQConfig
from models.complex_transformer import ComplexTransformerClassifier
from utils.metrics import UncertaintyMetrics


@dataclass
class UncertaintyComponents:
    """Structure for holding the three uncertainty components."""
    epistemic: torch.Tensor  # Model uncertainty (Equation 9)
    aleatoric: torch.Tensor  # Data uncertainty (Equation 10)
    covariance: torch.Tensor  # Novel covariance uncertainty (Equation 11)
    total: torch.Tensor  # Combined uncertainty
    predictions: torch.Tensor  # MC predictions used for computation


class AUQEstimator:
    """
    Main AUQ class implementing three-component uncertainty decomposition.
    
    Implements Equations 9-11 from the paper:
    - σ²_epistemic = Var_m[E[f(x)^(m)]]  (Equation 9)
    - σ²_aleatoric = E_m[Var[f(x)^(m)]]  (Equation 10) 
    - σ²_cov = E_m[|Cov[Re(f(x)^(m)), Im(f(x)^(m))]|]  (Equation 11)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: AUQConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or next(model.parameters()).device
        
        # Ensure model supports MC dropout
        self._setup_mc_dropout()
        
        # Cache for storing MC predictions
        self._mc_cache = {}
        
    def _setup_mc_dropout(self):
        """Setup model for Monte Carlo dropout."""
        # Enable dropout for all dropout layers
        def enable_dropout(module):
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
            # Handle complex dropout from our implementation
            elif hasattr(module, 'dropout') and isinstance(module.dropout, float):
                module.train()
        
        self.model.apply(enable_dropout)
    
    def estimate_uncertainty(
        self,
        inputs: torch.Tensor,
        num_samples: Optional[int] = None,
        return_predictions: bool = False,
        use_cache: bool = False
    ) -> Union[UncertaintyComponents, Tuple[UncertaintyComponents, torch.Tensor]]:
        """
        Estimate uncertainty using Monte Carlo sampling.
        
        Args:
            inputs: Input tensor (batch_size, seq_len, features)
            num_samples: Number of MC samples (uses config default if None)
            return_predictions: Whether to return MC predictions
            use_cache: Whether to use cached predictions
            
        Returns:
            UncertaintyComponents with epistemic, aleatoric, and covariance uncertainties
        """
        if num_samples is None:
            num_samples = self.config.num_mc_samples
        
        inputs = inputs.to(self.device)
        batch_size = inputs.shape[0]
        
        # Check cache
        cache_key = (inputs.shape, num_samples) if use_cache else None
        if use_cache and cache_key in self._mc_cache:
            mc_predictions = self._mc_cache[cache_key]
        else:
            # Perform Monte Carlo sampling
            mc_predictions = self._monte_carlo_sampling(inputs, num_samples)
            
            if use_cache:
                self._mc_cache[cache_key] = mc_predictions
        
        # Compute uncertainty components
        uncertainty_components = self._compute_uncertainty_components(mc_predictions)
        
        if return_predictions:
            return uncertainty_components, mc_predictions
        else:
            return uncertainty_components
    
    def _monte_carlo_sampling(
        self,
        inputs: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """
        Perform Monte Carlo sampling with dropout.
        
        Args:
            inputs: Input tensor
            num_samples: Number of MC samples
            
        Returns:
            MC predictions tensor (num_samples, batch_size, num_classes)
        """
        predictions = []
        
        # Set model to training mode for dropout
        original_training = self.model.training
        self.model.train()
        
        with torch.no_grad():
            for i in range(num_samples):
                # Forward pass with dropout
                outputs = self.model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    # Model returns (logits, complex_amplitudes, intermediate)
                    logits, complex_amplitudes = outputs[0], outputs[1]
                elif hasattr(outputs, 'logits'):
                    # Model returns TransformerOutput
                    logits = outputs.logits
                    complex_amplitudes = outputs.complex_amplitudes
                else:
                    # Model returns raw logits
                    logits = outputs
                    complex_amplitudes = None
                
                # Use complex amplitudes if available (better for uncertainty)
                if complex_amplitudes is not None and torch.is_complex(complex_amplitudes):
                    predictions.append(complex_amplitudes)
                else:
                    # Convert logits to complex for consistency
                    predictions.append(torch.complex(logits, torch.zeros_like(logits)))
        
        # Restore original training mode
        self.model.train(original_training)
        
        # Stack predictions: (num_samples, batch_size, num_classes)
        mc_predictions = torch.stack(predictions, dim=0)
        
        return mc_predictions
    
    def _compute_uncertainty_components(
        self,
        mc_predictions: torch.Tensor
    ) -> UncertaintyComponents:
        """
        Compute the three uncertainty components from MC predictions.
        Implements Equations 9-11 from the paper.
        
        Args:
            mc_predictions: MC predictions (num_samples, batch_size, num_classes)
            
        Returns:
            UncertaintyComponents with all three uncertainty types
        """
        num_samples, batch_size, num_classes = mc_predictions.shape
        
        # Compute epistemic uncertainty (Equation 9)
        # σ²_epistemic = Var_m[E[f(x)^(m)]]
        # Mean prediction across classes for each sample
        mean_predictions = torch.mean(mc_predictions, dim=2)  # (num_samples, batch_size)
        epistemic_uncertainty = torch.var(mean_predictions, dim=0)  # (batch_size,)
        
        # Expand to per-class uncertainty
        epistemic_per_class = epistemic_uncertainty.unsqueeze(1).expand(batch_size, num_classes)
        
        # Compute aleatoric uncertainty (Equation 10)
        # σ²_aleatoric = E_m[Var[f(x)^(m)]]
        # Variance across classes for each sample, then mean across samples
        prediction_magnitudes = torch.abs(mc_predictions)  # Use magnitude for variance
        class_variances = torch.var(prediction_magnitudes, dim=2)  # (num_samples, batch_size)
        aleatoric_uncertainty = torch.mean(class_variances, dim=0)  # (batch_size,)
        
        # Expand to per-class uncertainty
        aleatoric_per_class = aleatoric_uncertainty.unsqueeze(1).expand(batch_size, num_classes)
        
        # Compute novel covariance uncertainty (Equation 11)
        # σ²_cov = E_m[|Cov[Re(f(x)^(m)), Im(f(x)^(m))]|]
        covariance_uncertainty = compute_covariance_uncertainty(mc_predictions)
        
        # Compute total uncertainty
        total_uncertainty = epistemic_per_class + aleatoric_per_class + covariance_uncertainty
        
        return UncertaintyComponents(
            epistemic=epistemic_per_class,
            aleatoric=aleatoric_per_class,
            covariance=covariance_uncertainty,
            total=total_uncertainty,
            predictions=mc_predictions
        )
    
    def estimate_prediction_confidence(
        self,
        inputs: torch.Tensor,
        method: str = "entropy",
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Estimate prediction confidence using different methods.
        
        Args:
            inputs: Input tensor
            method: Confidence estimation method ('entropy', 'max_prob', 'margin')
            num_samples: Number of MC samples
            
        Returns:
            Confidence scores (batch_size,)
        """
        uncertainty_components = self.estimate_uncertainty(inputs, num_samples)
        mc_predictions = uncertainty_components.predictions
        
        # Convert complex predictions to probabilities
        prediction_probs = torch.abs(mc_predictions) ** 2  # Born rule
        prediction_probs = F.softmax(prediction_probs, dim=-1)
        
        # Average across MC samples
        mean_probs = torch.mean(prediction_probs, dim=0)  # (batch_size, num_classes)
        
        if method == "entropy":
            # Use entropy as confidence measure (lower entropy = higher confidence)
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
            max_entropy = torch.log(torch.tensor(mean_probs.shape[1], dtype=torch.float))
            confidence = 1 - entropy / max_entropy
            
        elif method == "max_prob":
            # Use maximum probability as confidence
            confidence = torch.max(mean_probs, dim=1)[0]
            
        elif method == "margin":
            # Use margin between top two predictions
            sorted_probs = torch.sort(mean_probs, dim=1, descending=True)[0]
            confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
            
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        
        return confidence
    
    def analyze_uncertainty_sources(
        self,
        inputs: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze different sources of uncertainty in detail.
        
        Args:
            inputs: Input tensor
            num_samples: Number of MC samples
            
        Returns:
            Dictionary with detailed uncertainty analysis
        """
        uncertainty_components = self.estimate_uncertainty(inputs, num_samples)
        
        # Compute relative contributions
        total_unc = uncertainty_components.total
        epistemic_ratio = uncertainty_components.epistemic / (total_unc + 1e-8)
        aleatoric_ratio = uncertainty_components.aleatoric / (total_unc + 1e-8)
        covariance_ratio = uncertainty_components.covariance / (total_unc + 1e-8)
        
        # Compute statistics
        analysis = {
            'epistemic_stats': {
                'mean': torch.mean(uncertainty_components.epistemic).item(),
                'std': torch.std(uncertainty_components.epistemic).item(),
                'max': torch.max(uncertainty_components.epistemic).item(),
                'min': torch.min(uncertainty_components.epistemic).item()
            },
            'aleatoric_stats': {
                'mean': torch.mean(uncertainty_components.aleatoric).item(),
                'std': torch.std(uncertainty_components.aleatoric).item(),
                'max': torch.max(uncertainty_components.aleatoric).item(),
                'min': torch.min(uncertainty_components.aleatoric).item()
            },
            'covariance_stats': {
                'mean': torch.mean(uncertainty_components.covariance).item(),
                'std': torch.std(uncertainty_components.covariance).item(),
                'max': torch.max(uncertainty_components.covariance).item(),
                'min': torch.min(uncertainty_components.covariance).item()
            },
            'relative_contributions': {
                'epistemic': torch.mean(epistemic_ratio).item(),
                'aleatoric': torch.mean(aleatoric_ratio).item(),
                'covariance': torch.mean(covariance_ratio).item()
            },
            'total_uncertainty': {
                'mean': torch.mean(total_unc).item(),
                'std': torch.std(total_unc).item()
            }
        }
        
        return analysis
    
    def check_convergence(
        self,
        inputs: torch.Tensor,
        max_samples: Optional[int] = None,
        convergence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Check convergence of uncertainty estimates with increasing MC samples.
        
        Args:
            inputs: Input tensor
            max_samples: Maximum number of samples to test
            convergence_threshold: MSE threshold for convergence
            
        Returns:
            Dictionary with convergence analysis
        """
        if max_samples is None:
            max_samples = min(self.config.num_mc_samples * 2, 100)
        
        if convergence_threshold is None:
            convergence_threshold = self.config.convergence_threshold
        
        # Test different numbers of samples
        sample_counts = list(range(self.config.min_samples, max_samples + 1, 5))
        uncertainties = []
        
        for num_samples in sample_counts:
            unc_components = self.estimate_uncertainty(inputs, num_samples)
            uncertainties.append({
                'num_samples': num_samples,
                'total_uncertainty': torch.mean(unc_components.total).item(),
                'epistemic': torch.mean(unc_components.epistemic).item(),
                'aleatoric': torch.mean(unc_components.aleatoric).item(),
                'covariance': torch.mean(unc_components.covariance).item()
            })
        
        # Check for convergence
        converged = False
        convergence_point = None
        
        if len(uncertainties) >= 2:
            for i in range(1, len(uncertainties)):
                current_unc = uncertainties[i]['total_uncertainty']
                prev_unc = uncertainties[i-1]['total_uncertainty']
                
                mse = (current_unc - prev_unc) ** 2
                if mse < convergence_threshold:
                    converged = True
                    convergence_point = uncertainties[i]['num_samples']
                    break
        
        return {
            'uncertainties': uncertainties,
            'converged': converged,
            'convergence_point': convergence_point,
            'recommended_samples': convergence_point or max_samples
        }


class EnsembleAUQ(AUQEstimator):
    """
    AUQ implementation using model ensembles instead of MC dropout.
    Provides alternative uncertainty estimation method.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        config: AUQConfig,
        device: Optional[torch.device] = None
    ):
        self.models = models
        self.config = config
        self.device = device or next(models[0].parameters()).device
        
        # Move all models to device
        for model in self.models:
            model.to(self.device)
            model.eval()
    
    def estimate_uncertainty(
        self,
        inputs: torch.Tensor,
        return_predictions: bool = False
    ) -> Union[UncertaintyComponents, Tuple[UncertaintyComponents, torch.Tensor]]:
        """
        Estimate uncertainty using ensemble of models.
        
        Args:
            inputs: Input tensor
            return_predictions: Whether to return ensemble predictions
            
        Returns:
            UncertaintyComponents from ensemble predictions
        """
        inputs = inputs.to(self.device)
        ensemble_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits, complex_amplitudes = outputs[0], outputs[1]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    complex_amplitudes = outputs.complex_amplitudes
                else:
                    logits = outputs
                    complex_amplitudes = None
                
                # Use complex amplitudes if available
                if complex_amplitudes is not None and torch.is_complex(complex_amplitudes):
                    ensemble_predictions.append(complex_amplitudes)
                else:
                    ensemble_predictions.append(torch.complex(logits, torch.zeros_like(logits)))
        
        # Stack predictions: (num_models, batch_size, num_classes)
        ensemble_predictions = torch.stack(ensemble_predictions, dim=0)
        
        # Compute uncertainty components
        uncertainty_components = self._compute_uncertainty_components(ensemble_predictions)
        
        if return_predictions:
            return uncertainty_components, ensemble_predictions
        else:
            return uncertainty_components


class AdaptiveAUQ(AUQEstimator):
    """
    Adaptive AUQ that adjusts the number of MC samples based on convergence.
    Provides efficient uncertainty estimation with automatic sample count selection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: AUQConfig,
        adaptive_threshold: float = 0.01,
        min_samples: int = 10,
        max_samples: int = 100,
        device: Optional[torch.device] = None
    ):
        super().__init__(model, config, device)
        self.adaptive_threshold = adaptive_threshold
        self.min_samples = min_samples
        self.max_samples = max_samples
    
    def estimate_uncertainty(
        self,
        inputs: torch.Tensor,
        return_predictions: bool = False
    ) -> Union[UncertaintyComponents, Tuple[UncertaintyComponents, torch.Tensor]]:
        """
        Estimate uncertainty with adaptive sampling.
        
        Automatically determines the optimal number of MC samples based on convergence.
        """
        inputs = inputs.to(self.device)
        
        # Start with minimum samples
        predictions = []
        current_samples = 0
        
        # Set model to training mode for dropout
        original_training = self.model.training
        self.model.train()
        
        with torch.no_grad():
            while current_samples < self.max_samples:
                # Add more samples
                batch_size = min(10, self.max_samples - current_samples)
                
                for _ in range(batch_size):
                    outputs = self.model(inputs)
                    
                    if isinstance(outputs, tuple):
                        logits, complex_amplitudes = outputs[0], outputs[1]
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        complex_amplitudes = outputs.complex_amplitudes
                    else:
                        logits = outputs
                        complex_amplitudes = None
                    
                    if complex_amplitudes is not None and torch.is_complex(complex_amplitudes):
                        predictions.append(complex_amplitudes)
                    else:
                        predictions.append(torch.complex(logits, torch.zeros_like(logits)))
                
                current_samples += batch_size
                
                # Check convergence if we have enough samples
                if current_samples >= self.min_samples:
                    # Compute uncertainty with current samples
                    current_predictions = torch.stack(predictions, dim=0)
                    current_uncertainty = self._compute_uncertainty_components(current_predictions)
                    
                    # Check convergence by comparing with previous uncertainty
                    if len(predictions) >= 20:  # Need at least 20 samples to check convergence
                        prev_predictions = torch.stack(predictions[:-10], dim=0)
                        prev_uncertainty = self._compute_uncertainty_components(prev_predictions)
                        
                        # Compute change in total uncertainty
                        uncertainty_change = torch.mean(torch.abs(
                            current_uncertainty.total - prev_uncertainty.total
                        )).item()
                        
                        if uncertainty_change < self.adaptive_threshold:
                            break
        
        # Restore original training mode
        self.model.train(original_training)
        
        # Final uncertainty computation
        final_predictions = torch.stack(predictions, dim=0)
        uncertainty_components = self._compute_uncertainty_components(final_predictions)
        
        # Add information about adaptive sampling
        uncertainty_components.num_samples_used = len(predictions)
        
        if return_predictions:
            return uncertainty_components, final_predictions
        else:
            return uncertainty_components


class AUQCalibrator:
    """
    Calibrator for AUQ uncertainty estimates.
    Improves calibration of uncertainty estimates using validation data.
    """
    
    def __init__(
        self,
        auq_estimator: AUQEstimator,
        calibration_method: str = "temperature_scaling"
    ):
        self.auq_estimator = auq_estimator
        self.calibration_method = calibration_method
        self.calibration_params = {}
        self.is_calibrated = False
    
    def calibrate(
        self,
        calibration_inputs: List[torch.Tensor],
        calibration_targets: List[torch.Tensor],
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Calibrate uncertainty estimates using validation data.
        
        Args:
            calibration_inputs: List of input tensors for calibration
            calibration_targets: List of target tensors
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with calibration metrics
        """
        # Split data into calibration and validation
        split_idx = int(len(calibration_inputs) * (1 - validation_split))
        
        cal_inputs = calibration_inputs[:split_idx]
        cal_targets = calibration_targets[:split_idx]
        val_inputs = calibration_inputs[split_idx:]
        val_targets = calibration_targets[split_idx:]
        
        if self.calibration_method == "temperature_scaling":
            self.calibration_params = self._temperature_scaling_calibration(
                cal_inputs, cal_targets
            )
        elif self.calibration_method == "platt_scaling":
            self.calibration_params = self._platt_scaling_calibration(
                cal_inputs, cal_targets
            )
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")
        
        self.is_calibrated = True
        
        # Evaluate calibration on validation set
        metrics = self._evaluate_calibration(val_inputs, val_targets)
        
        return metrics
    
    def _temperature_scaling_calibration(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Perform temperature scaling calibration."""
        # Collect all uncertainty estimates and correctness
        uncertainties = []
        correct_predictions = []
        
        for inp, target in zip(inputs, targets):
            unc_components = self.auq_estimator.estimate_uncertainty(inp.unsqueeze(0))
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.auq_estimator.model(inp.unsqueeze(0))
                if isinstance(outputs, tuple):
                    pred = torch.argmax(outputs[0], dim=-1)
                else:
                    pred = torch.argmax(outputs, dim=-1)
            
            total_unc = torch.mean(unc_components.total).item()
            is_correct = (pred.item() == target.item())
            
            uncertainties.append(total_unc)
            correct_predictions.append(is_correct)
        
        # Find optimal temperature parameter
        uncertainties = np.array(uncertainties)
        correct_predictions = np.array(correct_predictions)
        
        # Simple binary search for temperature (in practice, use proper optimization)
        best_temperature = 1.0
        best_ece = float('inf')
        
        for temp in np.linspace(0.1, 5.0, 50):
            calibrated_uncertainties = uncertainties / temp
            ece = self._compute_ece(calibrated_uncertainties, correct_predictions)
            
            if ece < best_ece:
                best_ece = ece
                best_temperature = temp
        
        return {'temperature': best_temperature}
    
    def _platt_scaling_calibration(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Perform Platt scaling calibration."""
        # This would implement Platt scaling (sigmoid calibration)
        # For brevity, returning dummy parameters
        return {'a': 1.0, 'b': 0.0}
    
    def _compute_ece(
        self,
        uncertainties: np.ndarray,
        correct_predictions: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_correct = correct_predictions[sorted_indices]
        
        # Create bins
        bin_size = len(uncertainties) // n_bins
        ece = 0.0
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(uncertainties)
            
            if end_idx > start_idx:
                bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
                bin_correct = sorted_correct[start_idx:end_idx]
                
                avg_uncertainty = np.mean(bin_uncertainties)
                avg_accuracy = np.mean(bin_correct)
                
                ece += np.abs(avg_uncertainty - avg_accuracy) * (end_idx - start_idx) / len(uncertainties)
        
        return ece
    
    def _evaluate_calibration(
        self,
        val_inputs: List[torch.Tensor],
        val_targets: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate calibration quality."""
        uncertainties = []
        correct_predictions = []
        
        for inp, target in zip(val_inputs, val_targets):
            unc_components = self.estimate_calibrated_uncertainty(inp.unsqueeze(0))
            
            with torch.no_grad():
                outputs = self.auq_estimator.model(inp.unsqueeze(0))
                if isinstance(outputs, tuple):
                    pred = torch.argmax(outputs[0], dim=-1)
                else:
                    pred = torch.argmax(outputs, dim=-1)
            
            total_unc = torch.mean(unc_components.total).item()
            is_correct = (pred.item() == target.item())
            
            uncertainties.append(total_unc)
            correct_predictions.append(is_correct)
        
        uncertainties = np.array(uncertainties)
        correct_predictions = np.array(correct_predictions)
        
        # Compute calibration metrics
        ece = self._compute_ece(uncertainties, correct_predictions)
        
        # Compute correlation between uncertainty and accuracy
        correlation = np.corrcoef(uncertainties, 1 - correct_predictions.astype(float))[0, 1]
        
        return {
            'ece': ece,
            'uncertainty_accuracy_correlation': correlation,
            'mean_uncertainty': np.mean(uncertainties),
            'mean_accuracy': np.mean(correct_predictions)
        }
    
    def estimate_calibrated_uncertainty(
        self,
        inputs: torch.Tensor,
        **kwargs
    ) -> UncertaintyComponents:
        """
        Estimate calibrated uncertainty.
        
        Args:
            inputs: Input tensor
            **kwargs: Additional arguments for uncertainty estimation
            
        Returns:
            Calibrated uncertainty components
        """
        if not self.is_calibrated:
            warnings.warn("AUQ calibrator has not been calibrated. Using uncalibrated estimates.")
            return self.auq_estimator.estimate_uncertainty(inputs, **kwargs)
        
        # Get raw uncertainty estimates
        uncertainty_components = self.auq_estimator.estimate_uncertainty(inputs, **kwargs)
        
        # Apply calibration
        if self.calibration_method == "temperature_scaling":
            temperature = self.calibration_params['temperature']
            
            # Scale uncertainties by temperature
            calibrated_epistemic = uncertainty_components.epistemic / temperature
            calibrated_aleatoric = uncertainty_components.aleatoric / temperature
            calibrated_covariance = uncertainty_components.covariance / temperature
            calibrated_total = uncertainty_components.total / temperature
            
            return UncertaintyComponents(
                epistemic=calibrated_epistemic,
                aleatoric=calibrated_aleatoric,
                covariance=calibrated_covariance,
                total=calibrated_total,
                predictions=uncertainty_components.predictions
            )
        
        elif self.calibration_method == "platt_scaling":
            # Apply Platt scaling transformation
            a = self.calibration_params['a']
            b = self.calibration_params['b']
            
            # Apply sigmoid transformation: 1 / (1 + exp(a * uncertainty + b))
            def platt_transform(uncertainty):
                return torch.sigmoid(-(a * uncertainty + b))
            
            return UncertaintyComponents(
                epistemic=platt_transform(uncertainty_components.epistemic),
                aleatoric=platt_transform(uncertainty_components.aleatoric),
                covariance=platt_transform(uncertainty_components.covariance),
                total=platt_transform(uncertainty_components.total),
                predictions=uncertainty_components.predictions
            )
        
        return uncertainty_components


class AUQVisualizer:
    """
    Visualizer for AUQ uncertainty results.
    Creates publication-quality visualizations of uncertainty decomposition.
    """
    
    def __init__(self, auq_estimator: AUQEstimator):
        self.auq_estimator = auq_estimator
    
    def visualize_uncertainty_decomposition(
        self,
        inputs: torch.Tensor,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> 'matplotlib.figure.Figure':
        """
        Visualize the three-component uncertainty decomposition.
        
        Args:
            inputs: Input tensor
            class_names: Optional class names for labeling
            save_path: Path to save figure
            title: Custom title
            
        Returns:
            Matplotlib figure
        """
        try:
            from utils.visualization import AUQVisualizer as VizUtils
        except ImportError:
            raise ImportError("Visualization utilities required")
        
        # Get uncertainty components
        uncertainty_components = self.auq_estimator.estimate_uncertainty(inputs)
        
        # Use visualization utilities
        viz_utils = VizUtils()
        
        # Create visualization
        fig = viz_utils.plot_uncertainty_decomposition(
            uncertainty_components.epistemic[0],  # First batch element
            uncertainty_components.aleatoric[0],
            uncertainty_components.covariance[0],
            class_names=class_names,
            title=title or "AUQ Uncertainty Decomposition"
        )
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_convergence(
        self,
        inputs: torch.Tensor,
        save_path: Optional[str] = None
    ) -> 'matplotlib.figure.Figure':
        """Visualize uncertainty convergence with increasing MC samples."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for visualization")
        
        # Check convergence
        convergence_analysis = self.auq_estimator.check_convergence(inputs)
        uncertainties = convergence_analysis['uncertainties']
        
        # Extract data for plotting
        sample_counts = [u['num_samples'] for u in uncertainties]
        total_uncertainties = [u['total_uncertainty'] for u in uncertainties]
        epistemic_uncertainties = [u['epistemic'] for u in uncertainties]
        aleatoric_uncertainties = [u['aleatoric'] for u in uncertainties]
        covariance_uncertainties = [u['covariance'] for u in uncertainties]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sample_counts, total_uncertainties, 'k-', linewidth=2, label='Total')
        ax.plot(sample_counts, epistemic_uncertainties, 'b--', label='Epistemic')
        ax.plot(sample_counts, aleatoric_uncertainties, 'r--', label='Aleatoric')
        ax.plot(sample_counts, covariance_uncertainties, 'g--', label='Covariance (Novel)')
        
        # Mark convergence point if found
        if convergence_analysis['converged']:
            ax.axvline(x=convergence_analysis['convergence_point'], 
                      color='red', linestyle=':', label='Convergence Point')
        
        ax.set_xlabel('Number of MC Samples')
        ax.set_ylabel('Uncertainty')
        ax.set_title('AUQ Uncertainty Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class AUQBenchmark:
    """
    Benchmark suite for evaluating AUQ performance and calibration quality.
    """
    
    def __init__(
        self,
        auq_estimator: AUQEstimator,
        test_inputs: List[torch.Tensor],
        test_targets: List[torch.Tensor]
    ):
        self.auq_estimator = auq_estimator
        self.test_inputs = test_inputs
        self.test_targets = test_targets
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive AUQ benchmark.
        
        Returns:
            Dictionary with benchmark results
        """
        results = {
            'calibration_metrics': [],
            'uncertainty_statistics': [],
            'computation_times': [],
            'convergence_analysis': [],
            'correlation_analysis': {}
        }
        
        print("Running AUQ benchmark...")
        
        all_uncertainties = []
        all_predictions = []
        all_targets = []
        
        for i, (inputs, target) in enumerate(zip(self.test_inputs, self.test_targets)):
            print(f"Processing sample {i+1}/{len(self.test_inputs)}")
            
            # Time the uncertainty computation
            start_time = time.time()
            uncertainty_components = self.auq_estimator.estimate_uncertainty(inputs.unsqueeze(0))
            computation_time = time.time() - start_time
            
            results['computation_times'].append(computation_time)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.auq_estimator.model(inputs.unsqueeze(0))
                if isinstance(outputs, tuple):
                    pred = torch.argmax(outputs[0], dim=-1)
                else:
                    pred = torch.argmax(outputs, dim=-1)
            
            # Store for later analysis
            all_uncertainties.append(uncertainty_components)
            all_predictions.append(pred.item())
            all_targets.append(target.item())
            
            # Uncertainty statistics
            stats = {
                'epistemic_mean': torch.mean(uncertainty_components.epistemic).item(),
                'aleatoric_mean': torch.mean(uncertainty_components.aleatoric).item(),
                'covariance_mean': torch.mean(uncertainty_components.covariance).item(),
                'total_mean': torch.mean(uncertainty_components.total).item(),
                'prediction_correct': pred.item() == target.item()
            }
            results['uncertainty_statistics'].append(stats)
            
            # Convergence analysis (for subset of samples)
            if i < 5:  # Only check convergence for first few samples
                convergence = self.auq_estimator.check_convergence(inputs.unsqueeze(0))
                results['convergence_analysis'].append(convergence)
        
        # Compute correlation between uncertainty and accuracy
        total_uncertainties = [s['total_mean'] for s in results['uncertainty_statistics']]
        correct_predictions = [s['prediction_correct'] for s in results['uncertainty_statistics']]
        
        if len(total_uncertainties) > 1:
            uncertainty_accuracy_corr = np.corrcoef(
                total_uncertainties, 
                [1-c for c in correct_predictions]  # Higher uncertainty should correlate with incorrectness
            )[0, 1]
            results['correlation_analysis']['uncertainty_accuracy'] = uncertainty_accuracy_corr
        
        # Aggregate results
        results['summary'] = self._aggregate_results(results)
        
        return results
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate benchmark results."""
        summary = {}
        
        # Computation time statistics
        if results['computation_times']:
            summary['mean_computation_time'] = np.mean(results['computation_times'])
            summary['std_computation_time'] = np.std(results['computation_times'])
        
        # Uncertainty statistics
        if results['uncertainty_statistics']:
            epistemic_values = [s['epistemic_mean'] for s in results['uncertainty_statistics']]
            aleatoric_values = [s['aleatoric_mean'] for s in results['uncertainty_statistics']]
            covariance_values = [s['covariance_mean'] for s in results['uncertainty_statistics']]
            total_values = [s['total_mean'] for s in results['uncertainty_statistics']]
            
            summary['mean_epistemic_uncertainty'] = np.mean(epistemic_values)
            summary['mean_aleatoric_uncertainty'] = np.mean(aleatoric_values)
            summary['mean_covariance_uncertainty'] = np.mean(covariance_values)
            summary['mean_total_uncertainty'] = np.mean(total_values)
            
            # Relative contributions
            total_sum = (summary['mean_epistemic_uncertainty'] + 
                        summary['mean_aleatoric_uncertainty'] + 
                        summary['mean_covariance_uncertainty'])
            
            if total_sum > 0:
                summary['epistemic_contribution'] = summary['mean_epistemic_uncertainty'] / total_sum
                summary['aleatoric_contribution'] = summary['mean_aleatoric_uncertainty'] / total_sum
                summary['covariance_contribution'] = summary['mean_covariance_uncertainty'] / total_sum
        
        # Convergence statistics
        if results['convergence_analysis']:
            converged_samples = [c['converged'] for c in results['convergence_analysis']]
            summary['convergence_rate'] = np.mean(converged_samples)
            
            convergence_points = [c['convergence_point'] for c in results['convergence_analysis'] if c['converged']]
            if convergence_points:
                summary['mean_convergence_point'] = np.mean(convergence_points)
        
        # Correlation analysis
        if 'uncertainty_accuracy' in results['correlation_analysis']:
            summary['uncertainty_accuracy_correlation'] = results['correlation_analysis']['uncertainty_accuracy']
        
        return summary
    
    def generate_report(self, results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Generate comprehensive benchmark report."""
        report = []
        report.append("AUQ Benchmark Report")
        report.append("=" * 50)
        report.append("")
        
        summary = results.get('summary', {})
        
        report.append("Uncertainty Decomposition:")
        report.append(f"  Mean Epistemic Uncertainty: {summary.get('mean_epistemic_uncertainty', 0):.6f}")
        report.append(f"  Mean Aleatoric Uncertainty: {summary.get('mean_aleatoric_uncertainty', 0):.6f}")
        report.append(f"  Mean Covariance Uncertainty: {summary.get('mean_covariance_uncertainty', 0):.6f} (Novel)")
        report.append(f"  Mean Total Uncertainty: {summary.get('mean_total_uncertainty', 0):.6f}")
        report.append("")
        
        report.append("Relative Contributions:")
        report.append(f"  Epistemic: {summary.get('epistemic_contribution', 0):.3f}")
        report.append(f"  Aleatoric: {summary.get('aleatoric_contribution', 0):.3f}")
        report.append(f"  Covariance: {summary.get('covariance_contribution', 0):.3f}")
        report.append("")
        
        report.append("Performance Metrics:")
        report.append(f"  Mean Computation Time: {summary.get('mean_computation_time', 0):.4f} ± {summary.get('std_computation_time', 0):.4f} seconds")
        report.append(f"  Convergence Rate: {summary.get('convergence_rate', 0):.3f}")
        if 'mean_convergence_point' in summary:
            report.append(f"  Mean Convergence Point: {summary['mean_convergence_point']:.1f} samples")
        report.append("")
        
        report.append("Calibration Quality:")
        if 'uncertainty_accuracy_correlation' in summary:
            report.append(f"  Uncertainty-Accuracy Correlation: {summary['uncertainty_accuracy_correlation']:.4f}")
        report.append("")
        
        # Configuration details
        report.append("Configuration:")
        report.append(f"  Number of MC Samples: {self.auq_estimator.config.num_mc_samples}")
        report.append(f"  MC Dropout Rate: {self.auq_estimator.config.mc_dropout_rate}")
        report.append(f"  Convergence Threshold: {self.auq_estimator.config.convergence_threshold}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


# Main interface functions

def estimate_uncertainty_with_auq(
    model: nn.Module,
    inputs: torch.Tensor,
    config: Optional[AUQConfig] = None,
    method: str = "mc_dropout",
    **kwargs
) -> UncertaintyComponents:
    """
    Main interface function for AUQ uncertainty estimation.
    
    Args:
        model: Complex-valued model
        inputs: Input tensor
        config: AUQ configuration (uses default if None)
        method: Uncertainty estimation method ('mc_dropout', 'ensemble', 'adaptive')
        **kwargs: Additional arguments for specific methods
        
    Returns:
        UncertaintyComponents with three-part decomposition
    """
    if config is None:
        config = AUQConfig()
    
    if method == "mc_dropout":
        estimator = AUQEstimator(model, config, **kwargs)
    elif method == "ensemble":
        models = kwargs.get('models', [model])
        estimator = EnsembleAUQ(models, config, **kwargs)
    elif method == "adaptive":
        estimator = AdaptiveAUQ(model, config, **kwargs)
    else:
        raise ValueError(f"Unknown AUQ method: {method}")
    
    return estimator.estimate_uncertainty(inputs)


def calibrate_auq_uncertainty(
    model: nn.Module,
    calibration_data: List[Tuple[torch.Tensor, torch.Tensor]],
    config: Optional[AUQConfig] = None,
    calibration_method: str = "temperature_scaling"
) -> AUQCalibrator:
    """
    Calibrate AUQ uncertainty estimates using validation data.
    
    Args:
        model: Complex-valued model
        calibration_data: List of (input, target) tuples for calibration
        config: AUQ configuration
        calibration_method: Calibration method to use
        
    Returns:
        Calibrated AUQ estimator
    """
    if config is None:
        config = AUQConfig()
    
    # Create base estimator
    base_estimator = AUQEstimator(model, config)
    
    # Create calibrator
    calibrator = AUQCalibrator(base_estimator, calibration_method)
    
    # Separate inputs and targets
    inputs = [data[0] for data in calibration_data]
    targets = [data[1] for data in calibration_data]
    
    # Perform calibration
    calibration_metrics = calibrator.calibrate(inputs, targets)
    
    print(f"Calibration completed. ECE: {calibration_metrics.get('ece', 'N/A'):.4f}")
    
    return calibrator


def benchmark_auq(
    model: nn.Module,
    test_data: List[Tuple[torch.Tensor, torch.Tensor]],
    config: Optional[AUQConfig] = None,
    save_report: bool = True,
    report_path: str = "auq_benchmark_report.txt"
) -> Dict[str, Any]:
    """
    Run AUQ benchmark on test data.
    
    Args:
        model: Model to benchmark
        test_data: List of (input, target) tuples
        config: AUQ configuration
        save_report: Whether to save benchmark report
        report_path: Path to save report
        
    Returns:
        Benchmark results dictionary
    """
    if config is None:
        config = AUQConfig()
    
    # Separate inputs and targets
    inputs = [data[0] for data in test_data]
    targets = [data[1] for data in test_data]
    
    # Create estimator and benchmark
    estimator = AUQEstimator(model, config)
    benchmark = AUQBenchmark(estimator, inputs, targets)
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    if save_report:
        benchmark.generate_report(results, report_path)
        print(f"AUQ benchmark report saved to {report_path}")
    
    return results


def analyze_uncertainty_contributions(
    uncertainty_components: UncertaintyComponents,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze the relative contributions of different uncertainty types.
    
    Args:
        uncertainty_components: Uncertainty components from AUQ
        class_names: Optional class names for analysis
        
    Returns:
        Dictionary with detailed analysis
    """
    # Convert to numpy for analysis
    epistemic = uncertainty_components.epistemic.cpu().numpy()
    aleatoric = uncertainty_components.aleatoric.cpu().numpy()
    covariance = uncertainty_components.covariance.cpu().numpy()
    total = uncertainty_components.total.cpu().numpy()
    
    # Compute relative contributions
    epistemic_ratio = epistemic / (total + 1e-8)
    aleatoric_ratio = aleatoric / (total + 1e-8)
    covariance_ratio = covariance / (total + 1e-8)
    
    analysis = {
        'overall_statistics': {
            'mean_epistemic': np.mean(epistemic),
            'mean_aleatoric': np.mean(aleatoric),
            'mean_covariance': np.mean(covariance),
            'mean_total': np.mean(total),
            'std_epistemic': np.std(epistemic),
            'std_aleatoric': np.std(aleatoric),
            'std_covariance': np.std(covariance),
            'std_total': np.std(total)
        },
        'relative_contributions': {
            'epistemic_fraction': np.mean(epistemic_ratio),
            'aleatoric_fraction': np.mean(aleatoric_ratio),
            'covariance_fraction': np.mean(covariance_ratio)
        },
        'per_class_analysis': {}
    }
    
    # Per-class analysis
    num_classes = epistemic.shape[-1]
    for class_idx in range(num_classes):
        class_name = class_names[class_idx] if class_names else f"Class_{class_idx}"
        
        analysis['per_class_analysis'][class_name] = {
            'epistemic': np.mean(epistemic[:, class_idx]),
            'aleatoric': np.mean(aleatoric[:, class_idx]),
            'covariance': np.mean(covariance[:, class_idx]),
            'total': np.mean(total[:, class_idx])
        }
    
    # Find most/least uncertain classes
    class_uncertainties = np.mean(total, axis=0)
    most_uncertain_idx = np.argmax(class_uncertainties)
    least_uncertain_idx = np.argmin(class_uncertainties)
    
    analysis['extremes'] = {
        'most_uncertain_class': {
            'index': most_uncertain_idx,
            'name': class_names[most_uncertain_idx] if class_names else f"Class_{most_uncertain_idx}",
            'uncertainty': class_uncertainties[most_uncertain_idx]
        },
        'least_uncertain_class': {
            'index': least_uncertain_idx,
            'name': class_names[least_uncertain_idx] if class_names else f"Class_{least_uncertain_idx}",
            'uncertainty': class_uncertainties[least_uncertain_idx]
        }
    }
    
    return analysis


# Utility functions for uncertainty analysis

def compute_prediction_intervals(
    uncertainty_components: UncertaintyComponents,
    confidence_level: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute prediction intervals based on uncertainty estimates.
    
    Args:
        uncertainty_components: Uncertainty estimates
        confidence_level: Confidence level for intervals
        
    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    from scipy.stats import norm
    
    # Use total uncertainty for interval computation
    total_uncertainty = uncertainty_components.total
    
    # Compute z-score for confidence level
    z_score = norm.ppf((1 + confidence_level) / 2)
    
    # Compute standard deviation from uncertainty
    std_dev = torch.sqrt(total_uncertainty)
    
    # Get mean predictions (from MC samples)
    mc_predictions = uncertainty_components.predictions
    mean_predictions = torch.mean(torch.abs(mc_predictions), dim=0)  # Use magnitude
    
    # Compute intervals
    margin = z_score * std_dev
    lower_bounds = mean_predictions - margin
    upper_bounds = mean_predictions + margin
    
    return lower_bounds, upper_bounds


def detect_out_of_distribution(
    uncertainty_components: UncertaintyComponents,
    threshold_method: str = "percentile",
    threshold_value: float = 95.0
) -> torch.Tensor:
    """
    Detect out-of-distribution samples using uncertainty.
    
    Args:
        uncertainty_components: Uncertainty estimates
        threshold_method: Method for setting threshold ('percentile', 'absolute')
        threshold_value: Threshold value
        
    Returns:
        Boolean tensor indicating OOD samples
    """
    total_uncertainty = uncertainty_components.total
    
    if threshold_method == "percentile":
        # Use percentile of total uncertainty
        threshold = torch.quantile(total_uncertainty.flatten(), threshold_value / 100.0)
        ood_mask = torch.max(total_uncertainty, dim=-1)[0] > threshold
        
    elif threshold_method == "absolute":
        # Use absolute threshold
        ood_mask = torch.max(total_uncertainty, dim=-1)[0] > threshold_value
        
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    return ood_mask