"""
Evaluation metrics for Q-XAI framework.
Includes classification metrics, interpretability evaluation, uncertainty calibration, and conformal prediction metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as sk_metrics
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr


class ClassificationMetrics:
    """Standard classification metrics for acoustic scene classification."""
    
    @staticmethod
    def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute classification accuracy."""
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=-1)
        
        correct = (predictions == targets).float().sum()
        total = targets.numel()
        
        return (correct / total).item()
    
    @staticmethod
    def compute_top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Compute top-k accuracy."""
        if predictions.dim() == 1:
            return ClassificationMetrics.compute_accuracy(predictions, targets)
        
        _, top_k_preds = torch.topk(predictions, k, dim=-1)
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
        
        correct = (top_k_preds == targets_expanded).any(dim=-1).float().sum()
        total = targets.numel()
        
        return (correct / total).item()
    
    @staticmethod
    def compute_f1_scores(
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        num_classes: int,
        average: str = 'macro'
    ) -> Union[float, np.ndarray]:
        """Compute F1 scores."""
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=-1)
        
        # Convert to numpy for sklearn
        y_true = targets.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        return sk_metrics.f1_score(y_true, y_pred, average=average, labels=np.arange(num_classes))
    
    @staticmethod
    def compute_precision_recall(
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        num_classes: int,
        average: str = 'macro'
    ) -> Tuple[float, float]:
        """Compute precision and recall."""
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=-1)
        
        y_true = targets.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        precision = sk_metrics.precision_score(y_true, y_pred, average=average, labels=np.arange(num_classes))
        recall = sk_metrics.recall_score(y_true, y_pred, average=average, labels=np.arange(num_classes))
        
        return precision, recall
    
    @staticmethod
    def compute_confusion_matrix(
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        num_classes: int
    ) -> np.ndarray:
        """Compute confusion matrix."""
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=-1)
        
        y_true = targets.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        return sk_metrics.confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    
    @staticmethod
    def compute_classification_report(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive classification report."""
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=-1)
        
        y_true = targets.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        return sk_metrics.classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )


class InterpretabilityMetrics:
    """Metrics for evaluating interpretability quality (QISA)."""
    
    @staticmethod
    def compute_deletion_curve(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        attributions: torch.Tensor,
        targets: torch.Tensor,
        steps: int = 20,
        baseline_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute deletion curve for attribution faithfulness evaluation.
        
        Args:
            model: The model to evaluate
            inputs: Input samples (batch_size, seq_len, features)
            attributions: Attribution maps (batch_size, seq_len, features)
            targets: Ground truth labels
            steps: Number of deletion steps
            baseline_value: Value to replace deleted features with
            
        Returns:
            Tuple of (deletion_percentages, accuracies)
        """
        model.eval()
        
        # Get original predictions
        with torch.no_grad():
            original_outputs = model(inputs)
            if isinstance(original_outputs, dict):
                original_outputs = original_outputs['logits']
            original_acc = ClassificationMetrics.compute_accuracy(original_outputs, targets)
        
        # Flatten attributions for sorting
        batch_size, seq_len, features = inputs.shape
        flat_attributions = attributions.view(batch_size, -1)
        flat_inputs = inputs.view(batch_size, -1, features)
        
        # Sort by attribution magnitude (descending)
        sorted_indices = torch.argsort(flat_attributions, dim=1, descending=True)
        
        deletion_percentages = np.linspace(0, 1, steps)
        accuracies = []
        
        for pct in deletion_percentages:
            # Create modified inputs
            modified_inputs = flat_inputs.clone()
            
            # Number of features to delete
            num_to_delete = int(pct * flat_inputs.shape[1])
            
            if num_to_delete > 0:
                for b in range(batch_size):
                    # Get indices of most important features for this sample
                    delete_indices = sorted_indices[b, :num_to_delete]
                    # Set to baseline value
                    modified_inputs[b, delete_indices] = baseline_value
            
            # Reshape back to original shape
            modified_inputs = modified_inputs.view(batch_size, seq_len, features)
            
            # Get predictions
            with torch.no_grad():
                modified_outputs = model(modified_inputs)
                if isinstance(modified_outputs, dict):
                    modified_outputs = modified_outputs['logits']
                acc = ClassificationMetrics.compute_accuracy(modified_outputs, targets)
                accuracies.append(acc)
        
        return deletion_percentages, np.array(accuracies)
    
    @staticmethod
    def compute_insertion_curve(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        attributions: torch.Tensor,
        targets: torch.Tensor,
        steps: int = 20,
        baseline_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute insertion curve (reverse of deletion)."""
        model.eval()
        
        batch_size, seq_len, features = inputs.shape
        flat_attributions = attributions.view(batch_size, -1)
        flat_inputs = inputs.view(batch_size, -1, features)
        
        # Sort by attribution magnitude (descending)
        sorted_indices = torch.argsort(flat_attributions, dim=1, descending=True)
        
        insertion_percentages = np.linspace(0, 1, steps)
        accuracies = []
        
        for pct in insertion_percentages:
            # Start with all baseline values
            modified_inputs = torch.full_like(flat_inputs, baseline_value)
            
            # Number of features to insert (keep)
            num_to_keep = int(pct * flat_inputs.shape[1])
            
            if num_to_keep > 0:
                for b in range(batch_size):
                    # Get indices of most important features
                    keep_indices = sorted_indices[b, :num_to_keep]
                    # Keep original values for these features
                    modified_inputs[b, keep_indices] = flat_inputs[b, keep_indices]
            
            # Reshape back
            modified_inputs = modified_inputs.view(batch_size, seq_len, features)
            
            # Get predictions
            with torch.no_grad():
                modified_outputs = model(modified_inputs)
                if isinstance(modified_outputs, dict):
                    modified_outputs = modified_outputs['logits']
                acc = ClassificationMetrics.compute_accuracy(modified_outputs, targets)
                accuracies.append(acc)
        
        return insertion_percentages, np.array(accuracies)
    
    @staticmethod
    def compute_audc_auic(
        deletion_curve: np.ndarray,
        insertion_curve: np.ndarray,
        percentages: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Area Under Deletion Curve (AUDC) and Area Under Insertion Curve (AUIC).
        
        Higher AUDC indicates better faithfulness (model degrades more when important features are removed).
        Higher AUIC indicates better faithfulness (model improves more when important features are added).
        """
        audc = np.trapz(deletion_curve, percentages)
        auic = np.trapz(insertion_curve, percentages)
        
        return audc, auic
    
    @staticmethod
    def compute_attribution_stability(
        attributions1: torch.Tensor,
        attributions2: torch.Tensor,
        method: str = 'pearson'
    ) -> float:
        """
        Compute stability of attributions between two similar inputs.
        
        Args:
            attributions1: First attribution map
            attributions2: Second attribution map
            method: Correlation method ('pearson', 'spearman', 'cosine')
        """
        # Flatten attributions
        attr1_flat = attributions1.flatten().cpu().numpy()
        attr2_flat = attributions2.flatten().cpu().numpy()
        
        if method == 'pearson':
            correlation, _ = pearsonr(attr1_flat, attr2_flat)
        elif method == 'spearman':
            correlation, _ = spearmanr(attr1_flat, attr2_flat)
        elif method == 'cosine':
            # Cosine similarity
            dot_product = np.dot(attr1_flat, attr2_flat)
            norm1 = np.linalg.norm(attr1_flat)
            norm2 = np.linalg.norm(attr2_flat)
            correlation = dot_product / (norm1 * norm2 + 1e-8)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return correlation
    
    @staticmethod
    def compute_sparsity(attributions: torch.Tensor, threshold: float = 0.1) -> float:
        """Compute sparsity of attribution maps."""
        flat_attr = attributions.flatten()
        max_attr = torch.max(torch.abs(flat_attr))
        
        # Count features below threshold
        sparse_features = torch.sum(torch.abs(flat_attr) < threshold * max_attr)
        total_features = flat_attr.numel()
        
        return (sparse_features / total_features).item()


class UncertaintyMetrics:
    """Metrics for evaluating uncertainty quantification (AUQ)."""
    
    @staticmethod
    def compute_ece(
        probabilities: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            probabilities: Predicted probabilities (batch_size, num_classes)
            predictions: Predicted class indices (batch_size,)
            targets: True class indices (batch_size,)
            n_bins: Number of calibration bins
        """
        # Get confidence (max probability) for each prediction
        confidences = torch.max(probabilities, dim=1)[0]
        accuracies = (predictions == targets).float()
        
        # Convert to numpy
        confidences = confidences.cpu().numpy()
        accuracies = accuracies.cpu().numpy()
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(confidences)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def compute_reliability_diagram(
        probabilities: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reliability diagram data.
        
        Returns:
            Tuple of (bin_centers, bin_accuracies, bin_counts)
        """
        confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
        accuracies = (predictions == targets).float().cpu().numpy()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        return bin_centers, np.array(bin_accuracies), np.array(bin_counts)
    
    @staticmethod
    def compute_brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Brier score for probability calibration."""
        # Convert targets to one-hot
        num_classes = probabilities.shape[1]
        targets_onehot = F.one_hot(targets, num_classes).float()
        
        # Compute Brier score
        brier = torch.mean(torch.sum((probabilities - targets_onehot) ** 2, dim=1))
        
        return brier.item()
    
    @staticmethod
    def compute_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        """Compute prediction entropy (measure of uncertainty)."""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probs_safe = torch.clamp(probabilities, eps, 1 - eps)
        
        entropy = -torch.sum(probs_safe * torch.log(probs_safe), dim=1)
        
        return entropy
    
    @staticmethod
    def compute_mutual_information(
        predictions_ensemble: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mutual information for ensemble uncertainty.
        
        Args:
            predictions_ensemble: Ensemble predictions (num_models, batch_size, num_classes)
        """
        # Average predictions across ensemble
        mean_pred = torch.mean(predictions_ensemble, dim=0)
        
        # Entropy of mean prediction
        entropy_mean = UncertaintyMetrics.compute_entropy(mean_pred)
        
        # Mean entropy of individual predictions
        individual_entropies = []
        for i in range(predictions_ensemble.shape[0]):
            entropy_i = UncertaintyMetrics.compute_entropy(predictions_ensemble[i])
            individual_entropies.append(entropy_i)
        
        mean_entropy = torch.mean(torch.stack(individual_entropies), dim=0)
        
        # Mutual information = entropy(mean) - mean(entropy)
        mutual_info = entropy_mean - mean_entropy
        
        return mutual_info


class ConformalPredictionMetrics:
    """Metrics for evaluating conformal prediction (QICP)."""
    
    @staticmethod
    def compute_coverage(prediction_sets: List[List[int]], targets: torch.Tensor) -> float:
        """
        Compute empirical coverage of conformal prediction sets.
        
        Args:
            prediction_sets: List of prediction sets for each sample
            targets: True labels
        """
        targets_np = targets.cpu().numpy()
        
        covered = 0
        total = len(prediction_sets)
        
        for i, pred_set in enumerate(prediction_sets):
            if targets_np[i] in pred_set:
                covered += 1
        
        return covered / total if total > 0 else 0.0
    
    @staticmethod
    def compute_average_set_size(prediction_sets: List[List[int]]) -> float:
        """Compute average size of prediction sets."""
        if not prediction_sets:
            return 0.0
        
        sizes = [len(pred_set) for pred_set in prediction_sets]
        return np.mean(sizes)
    
    @staticmethod
    def compute_conditional_coverage(
        prediction_sets: List[List[int]],
        targets: torch.Tensor,
        confidences: torch.Tensor,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute conditional coverage across confidence bins.
        
        Returns:
            Tuple of (bin_centers, coverage_per_bin)
        """
        targets_np = targets.cpu().numpy()
        confidences_np = confidences.cpu().numpy()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        coverage_per_bin = []
        
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences_np > bin_lower) & (confidences_np <= bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_indices = np.where(in_bin)[0]
                covered_in_bin = sum(
                    1 for idx in bin_indices 
                    if targets_np[idx] in prediction_sets[idx]
                )
                coverage = covered_in_bin / len(bin_indices)
                coverage_per_bin.append(coverage)
            else:
                coverage_per_bin.append(0.0)
        
        return bin_centers, np.array(coverage_per_bin)
    
    @staticmethod
    def compute_efficiency(
        prediction_sets: List[List[int]],
        baseline_sets: List[List[int]]
    ) -> float:
        """
        Compute efficiency compared to baseline method.
        
        Efficiency = (baseline_avg_size - method_avg_size) / baseline_avg_size
        """
        method_avg_size = ConformalPredictionMetrics.compute_average_set_size(prediction_sets)
        baseline_avg_size = ConformalPredictionMetrics.compute_average_set_size(baseline_sets)
        
        if baseline_avg_size == 0:
            return 0.0
        
        efficiency = (baseline_avg_size - method_avg_size) / baseline_avg_size
        return efficiency


class RobustnessMetrics:
    """Metrics for evaluating model robustness."""
    
    @staticmethod
    def compute_robustness_curve(
        clean_accuracy: float,
        noisy_accuracies: Dict[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute robustness curve showing accuracy vs noise level.
        
        Args:
            clean_accuracy: Accuracy on clean data
            noisy_accuracies: Dict mapping noise levels to accuracies
        """
        noise_levels = np.array(list(noisy_accuracies.keys()))
        accuracies = np.array(list(noisy_accuracies.values()))
        
        # Add clean accuracy at noise level 0
        noise_levels = np.concatenate([[0], noise_levels])
        accuracies = np.concatenate([[clean_accuracy], accuracies])
        
        # Sort by noise level
        sort_indices = np.argsort(noise_levels)
        noise_levels = noise_levels[sort_indices]
        accuracies = accuracies[sort_indices]
        
        return noise_levels, accuracies
    
    @staticmethod
    def compute_area_under_robustness_curve(
        noise_levels: np.ndarray,
        accuracies: np.ndarray
    ) -> float:
        """Compute area under robustness curve (higher is better)."""
        return np.trapz(accuracies, noise_levels)
    
    @staticmethod
    def compute_robustness_drop(
        clean_accuracy: float,
        adversarial_accuracy: float
    ) -> float:
        """Compute robustness drop percentage."""
        return (clean_accuracy - adversarial_accuracy) / clean_accuracy * 100


def compute_statistical_significance(
    results1: np.ndarray,
    results2: np.ndarray,
    test: str = 'paired_ttest',
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Compute statistical significance between two sets of results.
    
    Args:
        results1: First set of results (e.g., from multiple runs)
        results2: Second set of results
        test: Statistical test to use
        alpha: Significance level
        
    Returns:
        Tuple of (p_value, is_significant)
    """
    if test == 'paired_ttest':
        statistic, p_value = stats.ttest_rel(results1, results2)
    elif test == 'independent_ttest':
        statistic, p_value = stats.ttest_ind(results1, results2)
    elif test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(results1, results2)
    elif test == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(results1, results2)
    else:
        raise ValueError(f"Unknown statistical test: {test}")
    
    is_significant = p_value < alpha
    
    return p_value, is_significant


class ComprehensiveEvaluator:
    """Comprehensive evaluation combining all metrics."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names
    
    def evaluate_classification(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Comprehensive classification evaluation."""
        results = {}
        
        # Basic metrics
        results['accuracy'] = ClassificationMetrics.compute_accuracy(predictions, targets)
        results['top5_accuracy'] = ClassificationMetrics.compute_top_k_accuracy(predictions, targets, k=5)
        
        # F1 scores
        results['macro_f1'] = ClassificationMetrics.compute_f1_scores(
            predictions, targets, self.num_classes, 'macro'
        )
        results['weighted_f1'] = ClassificationMetrics.compute_f1_scores(
            predictions, targets, self.num_classes, 'weighted'
        )
        
        # Precision and recall
        precision, recall = ClassificationMetrics.compute_precision_recall(
            predictions, targets, self.num_classes, 'macro'
        )
        results['macro_precision'] = precision
        results['macro_recall'] = recall
        
        # Confusion matrix
        results['confusion_matrix'] = ClassificationMetrics.compute_confusion_matrix(
            predictions, targets, self.num_classes
        )
        
        # Classification report
        results['classification_report'] = ClassificationMetrics.compute_classification_report(
            predictions, targets, self.class_names
        )
        
        # Calibration metrics if probabilities provided
        if probabilities is not None:
            pred_labels = torch.argmax(predictions, dim=-1) if predictions.dim() > 1 else predictions
            results['ece'] = UncertaintyMetrics.compute_ece(probabilities, pred_labels, targets)
            results['brier_score'] = UncertaintyMetrics.compute_brier_score(probabilities, targets)
        
        return results
    
    def evaluate_interpretability(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        attributions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """Comprehensive interpretability evaluation."""
        results = {}
        
        # Faithfulness metrics
        del_pcts, del_accs = InterpretabilityMetrics.compute_deletion_curve(
            model, inputs, attributions, targets
        )
        ins_pcts, ins_accs = InterpretabilityMetrics.compute_insertion_curve(
            model, inputs, attributions, targets
        )
        
        audc, auic = InterpretabilityMetrics.compute_audc_auic(del_accs, ins_accs, del_pcts)
        
        results['audc'] = audc
        results['auic'] = auic
        results['deletion_curve'] = (del_pcts, del_accs)
        results['insertion_curve'] = (ins_pcts, ins_accs)
        
        # Sparsity
        results['sparsity'] = InterpretabilityMetrics.compute_sparsity(attributions)
        
        return results