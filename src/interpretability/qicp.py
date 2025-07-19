"""
Quantum-Inspired Conformal Prediction (QICP) for Q-XAI framework.
Implements distribution-free prediction sets using Born rule-inspired nonconformity scores.
Provides formal coverage guarantees while leveraging complex-valued model outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

from config.model_config import QICPConfig
from utils.complex_math import amplitude_to_probability


@dataclass
class ConformalPredictionResult:
    """Results from conformal prediction."""
    prediction_sets: List[List[int]]  # Prediction sets for each sample
    coverage: float  # Empirical coverage
    average_set_size: float  # Average prediction set size
    threshold: float  # Nonconformity threshold used
    scores: np.ndarray  # Nonconformity scores for all samples
    individual_set_sizes: List[int]  # Size of each prediction set


class NonconformityScore(ABC):
    """Abstract base class for nonconformity score functions."""
    
    @abstractmethod
    def compute_score(
        self, 
        predictions: torch.Tensor, 
        true_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute nonconformity scores."""
        pass
    
    @abstractmethod
    def predict_with_score(
        self, 
        predictions: torch.Tensor, 
        threshold: float
    ) -> List[List[int]]:
        """Generate prediction sets using the threshold."""
        pass


class BornRuleScore(NonconformityScore):
    """
    Born rule-inspired nonconformity score for complex-valued models.
    Uses squared magnitude of complex amplitudes as confidence measure.
    
    Score: s_i = 1 - |f(x_i)_{y_i}|^2  (Equation 12 in paper)
    """
    
    def __init__(self, use_complex_amplitudes: bool = True):
        self.use_complex_amplitudes = use_complex_amplitudes
    
    def compute_score(
        self, 
        predictions: torch.Tensor, 
        true_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Born rule nonconformity scores.
        
        Args:
            predictions: Complex amplitudes (batch_size, num_classes) or probabilities
            true_labels: True class labels (batch_size,)
            
        Returns:
            Nonconformity scores (batch_size,)
        """
        batch_size = predictions.shape[0]
        
        if torch.is_complex(predictions) and self.use_complex_amplitudes:
            # Use Born rule: |amplitude|^2
            confidences = torch.abs(predictions) ** 2
        else:
            # Use probabilities directly (fallback for real-valued models)
            confidences = predictions
        
        # Normalize to ensure probabilities sum to 1
        confidences = F.softmax(confidences, dim=-1)
        
        # Extract confidence for true labels
        true_label_confidences = confidences[torch.arange(batch_size), true_labels]
        
        # Nonconformity score: 1 - confidence
        scores = 1.0 - true_label_confidences
        
        return scores
    
    def predict_with_score(
        self, 
        predictions: torch.Tensor, 
        threshold: float
    ) -> List[List[int]]:
        """
        Generate prediction sets using Born rule threshold.
        
        Args:
            predictions: Complex amplitudes or probabilities
            threshold: Nonconformity threshold
            
        Returns:
            List of prediction sets (one per sample)
        """
        if torch.is_complex(predictions) and self.use_complex_amplitudes:
            # Use Born rule
            confidences = torch.abs(predictions) ** 2
        else:
            confidences = predictions
        
        # Normalize
        confidences = F.softmax(confidences, dim=-1)
        
        prediction_sets = []
        
        for i in range(confidences.shape[0]):
            sample_confidences = confidences[i]
            
            # Include classes with confidence >= 1 - threshold
            confident_classes = torch.where(sample_confidences >= 1 - threshold)[0]
            prediction_sets.append(confident_classes.tolist())
        
        return prediction_sets


class SoftmaxScore(NonconformityScore):
    """Standard softmax-based nonconformity score for comparison."""
    
    def compute_score(
        self, 
        predictions: torch.Tensor, 
        true_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute softmax-based nonconformity scores."""
        # Convert to probabilities if complex
        if torch.is_complex(predictions):
            probabilities = amplitude_to_probability(predictions, dim=-1)
        else:
            probabilities = F.softmax(predictions, dim=-1)
        
        batch_size = probabilities.shape[0]
        true_label_probs = probabilities[torch.arange(batch_size), true_labels]
        
        return 1.0 - true_label_probs
    
    def predict_with_score(
        self, 
        predictions: torch.Tensor, 
        threshold: float
    ) -> List[List[int]]:
        """Generate prediction sets using softmax probabilities."""
        if torch.is_complex(predictions):
            probabilities = amplitude_to_probability(predictions, dim=-1)
        else:
            probabilities = F.softmax(predictions, dim=-1)
        
        prediction_sets = []
        
        for i in range(probabilities.shape[0]):
            sample_probs = probabilities[i]
            confident_classes = torch.where(sample_probs >= 1 - threshold)[0]
            prediction_sets.append(confident_classes.tolist())
        
        return prediction_sets


class QICPPredictor:
    """
    Main QICP class implementing quantum-inspired conformal prediction.
    
    Provides distribution-free coverage guarantees using Born rule nonconformity scores.
    Implements Algorithm 1 from the paper (lines 4-10, 25-26).
    """
    
    def __init__(
        self,
        config: QICPConfig,
        score_function: Optional[NonconformityScore] = None
    ):
        self.config = config
        self.score_function = score_function or BornRuleScore()
        
        # Calibration data
        self.calibration_scores = None
        self.threshold = None
        self.is_calibrated = False
        
        # Validation tracking
        self.coverage_history = []
        self.set_size_history = []
    
    def calibrate(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, float]:
        """
        Calibrate the conformal predictor using a calibration dataset.
        
        Args:
            model: Trained complex-valued model
            calibration_loader: DataLoader for calibration data
            device: Device to run on
            
        Returns:
            Dictionary with calibration statistics
        """
        model.eval()
        all_scores = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_loader):
                # Handle different batch formats
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(device)
                    labels = batch['label'].to(device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                else:
                    raise ValueError("Unsupported batch format")
                
                # Get model predictions
                outputs = model(inputs, return_complex_amplitudes=True)
                
                if isinstance(outputs, tuple):
                    # (logits, complex_amplitudes, intermediate)
                    _, complex_amplitudes, _ = outputs
                    predictions = complex_amplitudes
                else:
                    predictions = outputs
                
                # Compute nonconformity scores
                scores = self.score_function.compute_score(predictions, labels)
                all_scores.extend(scores.cpu().numpy())
        
        self.calibration_scores = np.array(all_scores)
        
        # Compute threshold for desired coverage level
        alpha = 1 - self.config.confidence_level
        n = len(self.calibration_scores)
        
        # Use empirical quantile
        quantile_level = (n + 1) * (1 - alpha) / n
        quantile_level = min(quantile_level, 1.0)  # Ensure <= 1
        
        self.threshold = np.quantile(self.calibration_scores, quantile_level)
        self.is_calibrated = True
        
        # Compute calibration statistics
        calibration_stats = {
            'num_calibration_samples': n,
            'threshold': self.threshold,
            'target_coverage': self.config.confidence_level,
            'mean_score': np.mean(self.calibration_scores),
            'std_score': np.std(self.calibration_scores),
            'min_score': np.min(self.calibration_scores),
            'max_score': np.max(self.calibration_scores)
        }
        
        return calibration_stats
    
    def predict(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu'),
        return_scores: bool = False
    ) -> ConformalPredictionResult:
        """
        Generate conformal prediction sets for test data.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to run on
            return_scores: Whether to return nonconformity scores
            
        Returns:
            ConformalPredictionResult with prediction sets and statistics
        """
        if not self.is_calibrated:
            raise ValueError("Predictor must be calibrated before making predictions")
        
        model.eval()
        all_prediction_sets = []
        all_true_labels = []
        all_scores = [] if return_scores else None
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle batch format
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(device)
                    labels = batch['label'].to(device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                else:
                    raise ValueError("Unsupported batch format")
                
                # Get model predictions
                outputs = model(inputs, return_complex_amplitudes=True)
                
                if isinstance(outputs, tuple):
                    _, complex_amplitudes, _ = outputs
                    predictions = complex_amplitudes
                else:
                    predictions = outputs
                
                # Generate prediction sets
                batch_prediction_sets = self.score_function.predict_with_score(
                    predictions, self.threshold
                )
                all_prediction_sets.extend(batch_prediction_sets)
                all_true_labels.extend(labels.cpu().numpy())
                
                # Compute scores if requested
                if return_scores:
                    scores = self.score_function.compute_score(predictions, labels)
                    all_scores.extend(scores.cpu().numpy())
        
        # Compute coverage and statistics
        coverage = self._compute_coverage(all_prediction_sets, all_true_labels)
        average_set_size = np.mean([len(pred_set) for pred_set in all_prediction_sets])
        individual_set_sizes = [len(pred_set) for pred_set in all_prediction_sets]
        
        # Update history
        self.coverage_history.append(coverage)
        self.set_size_history.append(average_set_size)
        
        result = ConformalPredictionResult(
            prediction_sets=all_prediction_sets,
            coverage=coverage,
            average_set_size=average_set_size,
            threshold=self.threshold,
            scores=np.array(all_scores) if all_scores else None,
            individual_set_sizes=individual_set_sizes
        )
        
        return result
    
    def predict_single(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        device: torch.device = torch.device('cpu')
    ) -> List[int]:
        """
        Generate prediction set for a single input.
        
        Args:
            model: Trained model
            inputs: Single input tensor
            device: Device to run on
            
        Returns:
            Prediction set (list of class indices)
        """
        if not self.is_calibrated:
            raise ValueError("Predictor must be calibrated before making predictions")
        
        model.eval()
        
        with torch.no_grad():
            if inputs.dim() == 2:  # Add batch dimension
                inputs = inputs.unsqueeze(0)
            
            inputs = inputs.to(device)
            
            # Get model predictions
            outputs = model(inputs, return_complex_amplitudes=True)
            
            if isinstance(outputs, tuple):
                _, complex_amplitudes, _ = outputs
                predictions = complex_amplitudes
            else:
                predictions = outputs
            
            # Generate prediction set
            prediction_sets = self.score_function.predict_with_score(
                predictions, self.threshold
            )
            
            return prediction_sets[0]  # Return first (and only) prediction set
    
    def _compute_coverage(
        self, 
        prediction_sets: List[List[int]], 
        true_labels: List[int]
    ) -> float:
        """Compute empirical coverage of prediction sets."""
        if len(prediction_sets) != len(true_labels):
            raise ValueError("Number of prediction sets must match number of true labels")
        
        covered = 0
        total = len(prediction_sets)
        
        for pred_set, true_label in zip(prediction_sets, true_labels):
            if true_label in pred_set:
                covered += 1
        
        return covered / total if total > 0 else 0.0
    
    def validate_coverage(
        self,
        model: nn.Module,
        validation_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, float]:
        """
        Validate coverage on a separate validation set.
        
        Args:
            model: Trained model
            validation_loader: Validation data loader
            device: Device to run on
            
        Returns:
            Dictionary with validation statistics
        """
        if not self.config.validate_coverage:
            warnings.warn("Coverage validation is disabled in config")
        
        result = self.predict(model, validation_loader, device)
        
        # Compute detailed statistics
        validation_stats = {
            'empirical_coverage': result.coverage,
            'target_coverage': self.config.confidence_level,
            'coverage_gap': abs(result.coverage - self.config.confidence_level),
            'average_set_size': result.average_set_size,
            'set_size_std': np.std(result.individual_set_sizes),
            'min_set_size': min(result.individual_set_sizes),
            'max_set_size': max(result.individual_set_sizes),
            'threshold_used': self.threshold
        }
        
        return validation_stats
    
    def adaptive_threshold_adjustment(
        self,
        current_coverage: float,
        target_coverage: float,
        adjustment_factor: float = 0.1
    ) -> float:
        """
        Adaptively adjust threshold based on observed coverage.
        
        Args:
            current_coverage: Currently observed coverage
            target_coverage: Target coverage level
            adjustment_factor: How much to adjust threshold
            
        Returns:
            New threshold value
        """
        if not self.config.adaptive_threshold:
            return self.threshold
        
        coverage_gap = current_coverage - target_coverage
        
        # If coverage is too high, increase threshold (smaller sets)
        # If coverage is too low, decrease threshold (larger sets)
        threshold_adjustment = -coverage_gap * adjustment_factor
        
        new_threshold = self.threshold + threshold_adjustment
        new_threshold = np.clip(new_threshold, 0.0, 1.0)
        
        self.threshold = new_threshold
        return new_threshold


class MultiClassQICP(QICPPredictor):
    """
    Extended QICP for multi-class scenarios with class-specific thresholds.
    Useful when different classes have varying uncertainty characteristics.
    """
    
    def __init__(
        self,
        config: QICPConfig,
        num_classes: int,
        score_function: Optional[NonconformityScore] = None
    ):
        super().__init__(config, score_function)
        self.num_classes = num_classes
        self.class_thresholds = None
        self.class_calibration_scores = None
    
    def calibrate(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, Any]:
        """Calibrate with class-specific thresholds."""
        model.eval()
        
        # Collect scores for each class
        class_scores = {i: [] for i in range(self.num_classes)}
        
        with torch.no_grad():
            for batch in calibration_loader:
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(device)
                    labels = batch['label'].to(device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                
                outputs = model(inputs, return_complex_amplitudes=True)
                
                if isinstance(outputs, tuple):
                    _, complex_amplitudes, _ = outputs
                    predictions = complex_amplitudes
                else:
                    predictions = outputs
                
                scores = self.score_function.compute_score(predictions, labels)
                
                # Group scores by class
                for score, label in zip(scores.cpu().numpy(), labels.cpu().numpy()):
                    class_scores[label].append(score)
        
        # Compute class-specific thresholds
        alpha = 1 - self.config.confidence_level
        self.class_thresholds = {}
        self.class_calibration_scores = class_scores
        
        for class_idx in range(self.num_classes):
            if len(class_scores[class_idx]) > 0:
                scores_array = np.array(class_scores[class_idx])
                n = len(scores_array)
                quantile_level = (n + 1) * (1 - alpha) / n
                quantile_level = min(quantile_level, 1.0)
                
                self.class_thresholds[class_idx] = np.quantile(scores_array, quantile_level)
            else:
                # Fallback for classes with no calibration samples
                self.class_thresholds[class_idx] = 0.5
        
        self.is_calibrated = True
        
        # Compute statistics
        all_scores = []
        for scores in class_scores.values():
            all_scores.extend(scores)
        
        calibration_stats = {
            'num_calibration_samples': len(all_scores),
            'class_thresholds': self.class_thresholds,
            'target_coverage': self.config.confidence_level,
            'samples_per_class': {i: len(scores) for i, scores in class_scores.items()},
            'mean_threshold': np.mean(list(self.class_thresholds.values())),
            'std_threshold': np.std(list(self.class_thresholds.values()))
        }
        
        return calibration_stats


class QICPEvaluator:
    """
    Comprehensive evaluator for QICP performance.
    Provides detailed analysis of coverage, efficiency, and reliability.
    """
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_comprehensive(
        self,
        qicp_predictor: QICPPredictor,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        baseline_predictors: Optional[Dict[str, QICPPredictor]] = None,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of QICP performance.
        
        Args:
            qicp_predictor: Main QICP predictor to evaluate
            model: Trained model
            test_loader: Test data loader
            baseline_predictors: Optional baseline predictors for comparison
            device: Device to run on
            
        Returns:
            Comprehensive evaluation results
        """
        # Main QICP evaluation
        main_result = qicp_predictor.predict(model, test_loader, device, return_scores=True)
        
        evaluation = {
            'qicp_results': {
                'coverage': main_result.coverage,
                'average_set_size': main_result.average_set_size,
                'threshold': main_result.threshold,
                'set_size_distribution': {
                    'min': min(main_result.individual_set_sizes),
                    'max': max(main_result.individual_set_sizes),
                    'std': np.std(main_result.individual_set_sizes),
                    'median': np.median(main_result.individual_set_sizes)
                }
            }
        }
        
        # Baseline comparisons
        if baseline_predictors:
            evaluation['baseline_comparisons'] = {}
            
            for name, baseline in baseline_predictors.items():
                if not baseline.is_calibrated:
                    warnings.warn(f"Baseline {name} is not calibrated, skipping")
                    continue
                
                baseline_result = baseline.predict(model, test_loader, device)
                
                evaluation['baseline_comparisons'][name] = {
                    'coverage': baseline_result.coverage,
                    'average_set_size': baseline_result.average_set_size,
                    'efficiency_gain': (baseline_result.average_set_size - main_result.average_set_size) / baseline_result.average_set_size
                }
        
        # Coverage analysis across different confidence levels
        evaluation['coverage_analysis'] = self._analyze_coverage_levels(
            qicp_predictor, model, test_loader, device
        )
        
        # Conditional coverage analysis
        evaluation['conditional_coverage'] = self._analyze_conditional_coverage(
            main_result, model, test_loader, device
        )
        
        self.evaluation_history.append(evaluation)
        return evaluation
    
    def _analyze_coverage_levels(
        self,
        predictor: QICPPredictor,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze coverage at different confidence levels."""
        confidence_levels = [0.8, 0.85, 0.9, 0.95, 0.99]
        original_confidence = predictor.config.confidence_level
        original_threshold = predictor.threshold
        
        coverage_analysis = {}
        
        for conf_level in confidence_levels:
            # Temporarily update confidence level
            predictor.config.confidence_level = conf_level
            
            # Recompute threshold
            alpha = 1 - conf_level
            n = len(predictor.calibration_scores)
            quantile_level = (n + 1) * (1 - alpha) / n
            quantile_level = min(quantile_level, 1.0)
            predictor.threshold = np.quantile(predictor.calibration_scores, quantile_level)
            
            # Evaluate
            result = predictor.predict(model, test_loader, device)
            
            coverage_analysis[f'confidence_{conf_level}'] = {
                'empirical_coverage': result.coverage,
                'average_set_size': result.average_set_size,
                'threshold': predictor.threshold
            }
        
        # Restore original settings
        predictor.config.confidence_level = original_confidence
        predictor.threshold = original_threshold
        
        return coverage_analysis
    
    def _analyze_conditional_coverage(
        self,
        result: ConformalPredictionResult,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Analyze coverage conditional on model confidence."""
        if result.scores is None:
            return {"error": "Scores not available for conditional analysis"}
        
        # Bin samples by nonconformity scores
        n_bins = 5
        score_bins = np.quantile(result.scores, np.linspace(0, 1, n_bins + 1))
        
        conditional_coverage = {}
        
        # Collect true labels
        true_labels = []
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    labels = batch['label']
                elif isinstance(batch, (list, tuple)):
                    labels = batch[1]
                true_labels.extend(labels.numpy())
        
        for i in range(n_bins):
            bin_mask = (result.scores >= score_bins[i]) & (result.scores < score_bins[i + 1])
            if i == n_bins - 1:  # Include upper bound in last bin
                bin_mask = (result.scores >= score_bins[i]) & (result.scores <= score_bins[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_prediction_sets = [result.prediction_sets[j] for j in range(len(result.prediction_sets)) if bin_mask[j]]
                bin_true_labels = [true_labels[j] for j in range(len(true_labels)) if bin_mask[j]]
                
                bin_coverage = sum(1 for pred_set, true_label in zip(bin_prediction_sets, bin_true_labels) 
                                 if true_label in pred_set) / len(bin_prediction_sets)
                
                bin_avg_size = np.mean([len(pred_set) for pred_set in bin_prediction_sets])
                
                conditional_coverage[f'bin_{i}'] = {
                    'score_range': [score_bins[i], score_bins[i + 1]],
                    'coverage': bin_coverage,
                    'average_set_size': bin_avg_size,
                    'num_samples': np.sum(bin_mask)
                }
        
        return conditional_coverage


# Factory functions and utilities

def create_qicp_predictor(
    config: QICPConfig,
    score_type: str = "born_rule",
    num_classes: Optional[int] = None,
    **kwargs
) -> QICPPredictor:
    """
    Factory function to create QICP predictors.
    
    Args:
        config: QICP configuration
        score_type: Type of nonconformity score ('born_rule', 'softmax')
        num_classes: Number of classes (for multi-class variant)
        **kwargs: Additional arguments
        
    Returns:
        QICP predictor instance
    """
    # Create score function
    if score_type == "born_rule":
        score_function = BornRuleScore(**kwargs)
    elif score_type == "softmax":
        score_function = SoftmaxScore()
    else:
        raise ValueError(f"Unknown score type: {score_type}")
    
    # Create predictor
    if num_classes is not None and num_classes > 2:
        return MultiClassQICP(config, num_classes, score_function)
    else:
        return QICPPredictor(config, score_function)


def compare_nonconformity_scores(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: QICPConfig,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Compare different nonconformity score functions.
    
    Args:
        model: Trained model
        calibration_loader: Calibration data
        test_loader: Test data
        config: QICP configuration
        device: Device to run on
        
    Returns:
        Comparison results
    """
    score_types = ["born_rule", "softmax"]
    results = {}
    
    for score_type in score_types:
        predictor = create_qicp_predictor(config, score_type)
        
        # Calibrate
        calibration_stats = predictor.calibrate(model, calibration_loader, device)
        
        # Evaluate
        prediction_result = predictor.predict(model, test_loader, device)
        
        results[score_type] = {
            'calibration': calibration_stats,
            'evaluation': {
                'coverage': prediction_result.coverage,
                'average_set_size': prediction_result.average_set_size,
                'threshold': prediction_result.threshold
            }
        }
    
    # Compute efficiency comparison
    born_rule_size = results['born_rule']['evaluation']['average_set_size']
    softmax_size = results['softmax']['evaluation']['average_set_size']
    
    results['comparison'] = {
        'efficiency_improvement': (softmax_size - born_rule_size) / softmax_size,
        'born_rule_more_efficient': born_rule_size < softmax_size
    }
    
    return results


def qicp_robustness_test(
    predictor: QICPPredictor,
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3],
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Test QICP robustness under input perturbations.
    
    Args:
        predictor: Calibrated QICP predictor
        model: Trained model
        test_loader: Test data loader
        noise_levels: List of noise standard deviations to test
        device: Device to run on
        
    Returns:
        Robustness test results
    """
    robustness_results = {}
    
    for noise_level in noise_levels:
        # Create noisy test loader
        noisy_results = []
        model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(device)
                    labels = batch['label'].to(device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                
                # Add noise
                if noise_level > 0:
                    noise = torch.randn_like(inputs) * noise_level
                    noisy_inputs = inputs + noise
                else:
                    noisy_inputs = inputs
                
                # Get predictions
                outputs = model(noisy_inputs, return_complex_amplitudes=True)
                
                if isinstance(outputs, tuple):
                    _, complex_amplitudes, _ = outputs
                    predictions = complex_amplitudes
                else:
                    predictions = outputs
                
                # Generate prediction sets
                prediction_sets = predictor.score_function.predict_with_score(
                    predictions, predictor.threshold
                )
                
                # Check coverage for this batch
                for pred_set, true_label in zip(prediction_sets, labels.cpu().numpy()):
                    noisy_results.append({
                        'covered': true_label in pred_set,
                        'set_size': len(pred_set)
                    })
        
        # Compute statistics for this noise level
        coverage = np.mean([r['covered'] for r in noisy_results])
        avg_set_size = np.mean([r['set_size'] for r in noisy_results])
        
        robustness_results[f'noise_{noise_level}'] = {
            'coverage': coverage,
            'average_set_size': avg_set_size,
            'coverage_drop': robustness_results.get('noise_0.0', {}).get('coverage', coverage) - coverage
        }
    
    return robustness_results


class QICPVisualizer:
    """
    Visualization utilities for QICP results and analysis.
    """
    
    def __init__(self):
        pass
    
    def plot_coverage_vs_confidence(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Plot coverage vs confidence level."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for visualization")
        
        if 'coverage_analysis' not in evaluation_results:
            raise ValueError("Coverage analysis not found in results")
        
        coverage_data = evaluation_results['coverage_analysis']
        
        confidence_levels = []
        empirical_coverages = []
        set_sizes = []
        
        for key, value in coverage_data.items():
            if key.startswith('confidence_'):
                conf_level = float(key.split('_')[1])
                confidence_levels.append(conf_level)
                empirical_coverages.append(value['empirical_coverage'])
                set_sizes.append(value['average_set_size'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coverage plot
        ax1.plot(confidence_levels, empirical_coverages, 'bo-', linewidth=2, markersize=6, label='Empirical Coverage')
        ax1.plot(confidence_levels, confidence_levels, 'r--', linewidth=2, label='Target Coverage')
        ax1.set_xlabel('Target Confidence Level')
        ax1.set_ylabel('Empirical Coverage')
        ax1.set_title('QICP Coverage Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([min(confidence_levels) - 0.01, max(confidence_levels) + 0.01])
        ax1.set_ylim([min(empirical_coverages) - 0.01, 1.01])
        
        # Set size plot
        ax2.plot(confidence_levels, set_sizes, 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel('Target Confidence Level')
        ax2.set_ylabel('Average Set Size')
        ax2.set_title('Prediction Set Size vs Confidence')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_set_size_distribution(
        self,
        result: ConformalPredictionResult,
        save_path: Optional[str] = None
    ):
        """Plot distribution of prediction set sizes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for visualization")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(result.individual_set_sizes, bins=range(1, max(result.individual_set_sizes) + 2), 
                alpha=0.7, edgecolor='black', density=True)
        ax1.set_xlabel('Prediction Set Size')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Prediction Set Sizes')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_size = result.average_set_size
        median_size = np.median(result.individual_set_sizes)
        std_size = np.std(result.individual_set_sizes)
        
        stats_text = f'Mean: {mean_size:.2f}\nMedian: {median_size:.1f}\nStd: {std_size:.2f}'
        ax1.text(0.7, 0.8, stats_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # Cumulative distribution
        sorted_sizes = np.sort(result.individual_set_sizes)
        y_values = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
        ax2.plot(sorted_sizes, y_values, 'g-', linewidth=2)
        ax2.set_xlabel('Prediction Set Size')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_nonconformity_scores(
        self,
        predictor: QICPPredictor,
        result: ConformalPredictionResult,
        save_path: Optional[str] = None
    ):
        """Plot nonconformity score distribution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for visualization")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot calibration scores
        if predictor.calibration_scores is not None:
            ax.hist(predictor.calibration_scores, bins=50, alpha=0.5, 
                   label='Calibration Scores', density=True, color='blue')
        
        # Plot test scores if available
        if result.scores is not None:
            ax.hist(result.scores, bins=50, alpha=0.5, 
                   label='Test Scores', density=True, color='red')
        
        # Mark threshold
        ax.axvline(x=predictor.threshold, color='black', linestyle='--', 
                  linewidth=2, label=f'Threshold: {predictor.threshold:.3f}')
        
        ax.set_xlabel('Nonconformity Score')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Nonconformity Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Main interface functions

def run_qicp_evaluation(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: Optional[QICPConfig] = None,
    score_type: str = "born_rule",
    device: torch.device = torch.device('cpu'),
    return_detailed: bool = True
) -> Dict[str, Any]:
    """
    Main interface for running QICP evaluation.
    
    Args:
        model: Trained complex-valued model
        calibration_loader: Calibration data loader
        test_loader: Test data loader
        config: QICP configuration (uses default if None)
        score_type: Nonconformity score type
        device: Device to run on
        return_detailed: Whether to return detailed analysis
        
    Returns:
        Evaluation results dictionary
    """
    if config is None:
        config = QICPConfig()
    
    # Create predictor
    predictor = create_qicp_predictor(config, score_type)
    
    # Calibrate
    print("Calibrating QICP predictor...")
    calibration_stats = predictor.calibrate(model, calibration_loader, device)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    result = predictor.predict(model, test_loader, device, return_scores=True)
    
    # Basic results
    evaluation = {
        'calibration_stats': calibration_stats,
        'test_results': {
            'coverage': result.coverage,
            'average_set_size': result.average_set_size,
            'threshold': result.threshold,
            'target_coverage': config.confidence_level,
            'coverage_gap': abs(result.coverage - config.confidence_level)
        }
    }
    
    # Detailed analysis if requested
    if return_detailed:
        evaluator = QICPEvaluator()
        
        # Compare with baseline
        baseline_predictors = {
            'softmax': create_qicp_predictor(config, "softmax")
        }
        
        # Calibrate baseline
        baseline_predictors['softmax'].calibrate(model, calibration_loader, device)
        
        # Comprehensive evaluation
        detailed_eval = evaluator.evaluate_comprehensive(
            predictor, model, test_loader, baseline_predictors, device
        )
        
        evaluation['detailed_analysis'] = detailed_eval
        
        # Efficiency comparison
        if 'baseline_comparisons' in detailed_eval:
            softmax_results = detailed_eval['baseline_comparisons'].get('softmax', {})
            if 'efficiency_gain' in softmax_results:
                evaluation['efficiency_improvement'] = softmax_results['efficiency_gain']
    
    print(f"QICP Evaluation Complete:")
    print(f"  Coverage: {result.coverage:.3f} (target: {config.confidence_level:.3f})")
    print(f"  Average set size: {result.average_set_size:.2f}")
    print(f"  Threshold: {result.threshold:.3f}")
    
    return evaluation


def qicp_quick_test(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    config: Optional[QICPConfig] = None,
    split_ratio: float = 0.5,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Quick test of QICP with automatic calibration/test split.
    
    Args:
        model: Trained model
        data_loader: Single data loader (will be split)
        config: QICP configuration
        split_ratio: Fraction to use for calibration
        device: Device to run on
        
    Returns:
        Basic results dictionary
    """
    if config is None:
        config = QICPConfig()
    
    # Split data
    all_data = list(data_loader)
    split_idx = int(len(all_data) * split_ratio)
    
    # Create temporary data loaders
    calibration_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    # Convert back to data loaders (simplified)
    from torch.utils.data import DataLoader, TensorDataset
    
    # Extract tensors for calibration
    cal_inputs = torch.cat([batch[0] if isinstance(batch, (list, tuple)) else batch['spectrogram'] 
                           for batch in calibration_data])
    cal_labels = torch.cat([batch[1] if isinstance(batch, (list, tuple)) else batch['label'] 
                           for batch in calibration_data])
    
    # Extract tensors for test
    test_inputs = torch.cat([batch[0] if isinstance(batch, (list, tuple)) else batch['spectrogram'] 
                            for batch in test_data])
    test_labels = torch.cat([batch[1] if isinstance(batch, (list, tuple)) else batch['label'] 
                            for batch in test_data])
    
    # Create new data loaders
    cal_dataset = TensorDataset(cal_inputs, cal_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)
    
    cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Run evaluation
    results = run_qicp_evaluation(
        model, cal_loader, test_loader, config, device=device, return_detailed=False
    )
    
    # Return simplified results
    return {
        'coverage': results['test_results']['coverage'],
        'average_set_size': results['test_results']['average_set_size'],
        'coverage_gap': results['test_results']['coverage_gap'],
        'threshold': results['test_results']['threshold']
    }