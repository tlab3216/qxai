"""
Loss functions for Q-XAI framework.
Implements specialized loss functions for complex-valued transformers and uncertainty quantification.

This module provides:
1. Complex-aware cross-entropy loss using Born rule (|amplitude|²)
2. Focal loss for class imbalance in acoustic scene classification
3. Uncertainty-aware losses for AUQ training
4. Regularization losses for complex networks
5. Multi-component losses combining classification and uncertainty

Based on the Q-XAI paper: "Interpretable Complex-Valued Transformers for Acoustic Scene Classification"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any
import warnings


class ComplexCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for complex-valued model outputs using Born rule.
    
    Converts complex amplitudes to probabilities via |amplitude|² (Born rule)
    then applies standard cross-entropy loss. This is the main classification
    loss used in the Q-XAI framework.
    
    Args:
        weight: Class weights for handling imbalanced datasets
        ignore_index: Index to ignore in loss computation
        reduction: Reduction method ('mean', 'sum', 'none')
        label_smoothing: Amount of label smoothing to apply
        temperature: Temperature scaling for probability calibration
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        temperature: float = 1.0
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        
    def forward(
        self, 
        complex_logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss from complex logits.
        
        Args:
            complex_logits: Complex-valued model outputs (batch_size, num_classes)
            targets: Target class indices (batch_size,)
            
        Returns:
            Cross-entropy loss value
        """
        # Convert complex amplitudes to probabilities using Born rule
        # P(class) = |amplitude|² (Equation 12 in paper)
        probabilities = torch.abs(complex_logits) ** 2
        
        # Apply temperature scaling for calibration
        probabilities = probabilities / self.temperature
        
        # Normalize to ensure probabilities sum to 1
        probabilities = F.softmax(probabilities, dim=-1)
        
        # Add small epsilon to prevent log(0)
        eps = 1e-8
        probabilities = torch.clamp(probabilities, eps, 1 - eps)
        
        # Compute cross-entropy loss
        if self.label_smoothing > 0:
            # Apply label smoothing
            num_classes = probabilities.size(-1)
            one_hot = F.one_hot(targets, num_classes).float()
            smooth_targets = (1 - self.label_smoothing) * one_hot + \
                           self.label_smoothing / num_classes
            
            # Compute loss with smooth targets
            log_probs = torch.log(probabilities)
            loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(
                torch.log(probabilities),
                targets,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction='none'
            )
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance in acoustic scene classification.
    
    Focal loss down-weights easy examples and focuses learning on hard negatives.
    Particularly useful for datasets like ESC-50 with diverse class frequencies.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        weight: Class weights tensor
        reduction: Reduction method
        label_smoothing: Label smoothing factor
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (real or complex)
            targets: Target class indices
            
        Returns:
            Focal loss value
        """
        # Handle complex inputs
        if torch.is_complex(inputs):
            # Convert complex to probabilities using Born rule
            inputs = torch.abs(inputs) ** 2
        
        # Get probabilities
        probs = F.softmax(inputs, dim=-1)
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.weight, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # Get probability of true class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha
        if self.weight is not None:
            alpha_weight = self.weight[targets]
        
        # Final focal loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class UncertaintyLoss(nn.Module):
    """
    Loss function for training uncertainty quantification in AUQ framework.
    
    Combines classification loss with uncertainty regularization terms to
    encourage well-calibrated uncertainty estimates. Based on the three-component
    uncertainty decomposition in the Q-XAI paper.
    
    Args:
        classification_weight: Weight for classification loss term
        epistemic_weight: Weight for epistemic uncertainty regularization
        aleatoric_weight: Weight for aleatoric uncertainty regularization
        covariance_weight: Weight for novel covariance uncertainty term
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        epistemic_weight: float = 0.1,
        aleatoric_weight: float = 0.1,
        covariance_weight: float = 0.05
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.epistemic_weight = epistemic_weight
        self.aleatoric_weight = aleatoric_weight
        self.covariance_weight = covariance_weight
        
        # Base classification loss
        self.classification_loss = ComplexCrossEntropyLoss()
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty_components: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-aware loss.
        
        Args:
            predictions: Model predictions (complex or real)
            targets: Target labels
            uncertainty_components: Dict with 'epistemic', 'aleatoric', 'covariance'
            
        Returns:
            Dictionary with loss components
        """
        # Main classification loss
        cls_loss = self.classification_loss(predictions, targets)
        
        total_loss = self.classification_weight * cls_loss
        loss_dict = {'classification': cls_loss, 'total': total_loss}
        
        # Add uncertainty regularization if components provided
        if uncertainty_components is not None:
            
            # Epistemic uncertainty regularization
            # Encourage model to be confident on training data
            if 'epistemic' in uncertainty_components:
                epistemic_unc = uncertainty_components['epistemic']
                # Penalize high epistemic uncertainty on training data
                epistemic_reg = torch.mean(epistemic_unc)
                total_loss += self.epistemic_weight * epistemic_reg
                loss_dict['epistemic_reg'] = epistemic_reg
            
            # Aleatoric uncertainty regularization
            # Encourage appropriate aleatoric uncertainty estimation
            if 'aleatoric' in uncertainty_components:
                aleatoric_unc = uncertainty_components['aleatoric']
                # Encourage reasonable aleatoric uncertainty levels
                aleatoric_reg = torch.var(aleatoric_unc)  # Penalize extreme variance
                total_loss += self.aleatoric_weight * aleatoric_reg
                loss_dict['aleatoric_reg'] = aleatoric_reg
            
            # Covariance uncertainty regularization (novel component)
            # Encourage stable phase relationships when model is confident
            if 'covariance' in uncertainty_components:
                covariance_unc = uncertainty_components['covariance']
                # Penalize unstable phase relationships
                covariance_reg = torch.mean(covariance_unc)
                total_loss += self.covariance_weight * covariance_reg
                loss_dict['covariance_reg'] = covariance_reg
        
        loss_dict['total'] = total_loss
        return loss_dict


class ComplexRegularizationLoss(nn.Module):
    """
    Regularization losses specific to complex-valued networks.
    
    Implements various regularization techniques to improve training stability
    and generalization of complex-valued transformers.
    """
    
    def __init__(
        self,
        magnitude_reg_weight: float = 0.01,
        phase_reg_weight: float = 0.01,
        unitarity_reg_weight: float = 0.0
    ):
        super().__init__()
        self.magnitude_reg_weight = magnitude_reg_weight
        self.phase_reg_weight = phase_reg_weight
        self.unitarity_reg_weight = unitarity_reg_weight
        
    def forward(
        self, 
        complex_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses for complex parameters.
        
        Args:
            complex_params: Dictionary of complex-valued parameters
            
        Returns:
            Dictionary with regularization loss components
        """
        reg_losses = {}
        total_reg = torch.tensor(0.0, device=next(iter(complex_params.values())).device)
        
        for name, param in complex_params.items():
            if torch.is_complex(param):
                
                # Magnitude regularization - prevent magnitude explosion
                if self.magnitude_reg_weight > 0:
                    magnitude = torch.abs(param)
                    mag_reg = torch.mean(magnitude ** 2)
                    reg_losses[f'{name}_magnitude_reg'] = mag_reg
                    total_reg += self.magnitude_reg_weight * mag_reg
                
                # Phase regularization - encourage phase diversity
                if self.phase_reg_weight > 0:
                    phase = torch.angle(param)
                    # Penalize phase clustering around specific values
                    phase_var = torch.var(phase)
                    phase_reg = torch.exp(-phase_var)  # Encourage high variance
                    reg_losses[f'{name}_phase_reg'] = phase_reg
                    total_reg += self.phase_reg_weight * phase_reg
                
                # Unitarity regularization for attention weights
                if self.unitarity_reg_weight > 0 and 'attention' in name.lower():
                    # Encourage unitary-like properties in attention matrices
                    if param.dim() >= 2:
                        # For matrices, penalize deviation from unitarity
                        matrix = param.view(-1, param.size(-1))
                        gram = torch.matmul(matrix, torch.conj(matrix).transpose(-2, -1))
                        identity = torch.eye(gram.size(-1), device=param.device, dtype=param.dtype)
                        unitarity_reg = torch.norm(gram - identity) ** 2
                        reg_losses[f'{name}_unitarity_reg'] = unitarity_reg
                        total_reg += self.unitarity_reg_weight * unitarity_reg
        
        reg_losses['total_regularization'] = total_reg
        return reg_losses


class QICPCalibrationLoss(nn.Module):
    """
    Loss function for improving conformal prediction calibration.
    
    Encourages the model to produce well-calibrated confidence scores
    for the QICP (Quantum-Inspired Conformal Prediction) component.
    """
    
    def __init__(
        self,
        temperature_init: float = 1.0,
        learn_temperature: bool = True
    ):
        super().__init__()
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature_init))
        else:
            self.register_buffer('temperature', torch.tensor(temperature_init))
        
    def forward(
        self,
        complex_logits: torch.Tensor,
        targets: torch.Tensor,
        return_calibrated: bool = False
    ) -> torch.Tensor:
        """
        Compute calibration loss using temperature scaling.
        
        Args:
            complex_logits: Complex model outputs
            targets: True labels
            return_calibrated: Whether to return calibrated probabilities
            
        Returns:
            Calibration loss (and optionally calibrated probabilities)
        """
        # Convert to probabilities using Born rule
        probabilities = torch.abs(complex_logits) ** 2
        
        # Apply temperature scaling
        calibrated_logits = probabilities / self.temperature
        calibrated_probs = F.softmax(calibrated_logits, dim=-1)
        
        # Compute negative log-likelihood
        nll_loss = F.nll_loss(torch.log(calibrated_probs + 1e-8), targets)
        
        if return_calibrated:
            return nll_loss, calibrated_probs
        else:
            return nll_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification, uncertainty, and interpretability objectives.
    
    Implements the complete loss function for the Q-XAI framework training,
    balancing multiple objectives for optimal performance.
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
        regularization_weight: float = 0.01,
        calibration_weight: float = 0.05,
        adaptive_weighting: bool = True
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.uncertainty_weight = uncertainty_weight
        self.regularization_weight = regularization_weight
        self.calibration_weight = calibration_weight
        self.adaptive_weighting = adaptive_weighting
        
        # Individual loss components
        self.classification_loss = ComplexCrossEntropyLoss()
        self.uncertainty_loss = UncertaintyLoss()
        self.regularization_loss = ComplexRegularizationLoss()
        self.calibration_loss = QICPCalibrationLoss()
        
        # Adaptive weighting parameters
        if adaptive_weighting:
            self.log_vars = nn.Parameter(torch.zeros(4))  # For 4 loss components
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty_components: Optional[Dict[str, torch.Tensor]] = None,
        complex_params: Optional[Dict[str, torch.Tensor]] = None,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss for Q-XAI training.
        
        Args:
            predictions: Model predictions
            targets: Target labels
            uncertainty_components: AUQ uncertainty components
            complex_params: Complex model parameters for regularization
            epoch: Current training epoch for adaptive weighting
            
        Returns:
            Dictionary with all loss components
        """
        # Individual loss components
        cls_loss = self.classification_loss(predictions, targets)
        
        # Uncertainty loss
        unc_loss_dict = self.uncertainty_loss(
            predictions, targets, uncertainty_components
        )
        unc_loss = unc_loss_dict['total']
        
        # Regularization loss
        reg_loss = torch.tensor(0.0, device=predictions.device)
        if complex_params is not None:
            reg_loss_dict = self.regularization_loss(complex_params)
            reg_loss = reg_loss_dict['total_regularization']
        
        # Calibration loss
        cal_loss = self.calibration_loss(predictions, targets)
        
        # Compute weights
        if self.adaptive_weighting:
            # Adaptive weighting based on uncertainty
            weights = torch.exp(-self.log_vars)
            precision_loss = torch.sum(self.log_vars)
            
            weighted_cls = weights[0] * cls_loss
            weighted_unc = weights[1] * unc_loss
            weighted_reg = weights[2] * reg_loss
            weighted_cal = weights[3] * cal_loss
            
            total_loss = weighted_cls + weighted_unc + weighted_reg + weighted_cal + precision_loss
        else:
            # Fixed weighting
            total_loss = (
                self.classification_weight * cls_loss +
                self.uncertainty_weight * unc_loss +
                self.regularization_weight * reg_loss +
                self.calibration_weight * cal_loss
            )
        
        # Compile results
        loss_dict = {
            'total': total_loss,
            'classification': cls_loss,
            'uncertainty': unc_loss,
            'regularization': reg_loss,
            'calibration': cal_loss
        }
        
        # Add individual uncertainty components
        for key, value in unc_loss_dict.items():
            if key != 'total':
                loss_dict[f'uncertainty_{key}'] = value
        
        return loss_dict


def create_loss_function(
    loss_type: str = 'complex_cross_entropy',
    num_classes: int = 10,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create appropriate loss function for Q-XAI training.
    
    Args:
        loss_type: Type of loss function to create
        num_classes: Number of classes in the dataset
        class_weights: Optional class weights for imbalanced datasets
        **kwargs: Additional arguments for specific loss functions
        
    Returns:
        Configured loss function
    """
    if loss_type == 'complex_cross_entropy':
        return ComplexCrossEntropyLoss(weight=class_weights, **kwargs)
    
    elif loss_type == 'focal':
        return FocalLoss(weight=class_weights, **kwargs)
    
    elif loss_type == 'uncertainty':
        return UncertaintyLoss(**kwargs)
    
    elif loss_type == 'multi_task':
        return MultiTaskLoss(**kwargs)
    
    elif loss_type == 'qicp_calibration':
        return QICPCalibrationLoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    method: str = 'inverse_frequency'
) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.
    
    Args:
        targets: Target labels tensor
        num_classes: Total number of classes
        method: Weighting method ('inverse_frequency', 'effective_number')
        
    Returns:
        Class weights tensor
    """
    if method == 'inverse_frequency':
        # Inverse frequency weighting
        class_counts = torch.bincount(targets, minlength=num_classes)
        total_samples = len(targets)
        weights = total_samples / (num_classes * class_counts.float())
        
    elif method == 'effective_number':
        # Effective number of samples weighting
        beta = 0.9999
        class_counts = torch.bincount(targets, minlength=num_classes)
        effective_num = 1.0 - torch.pow(beta, class_counts.float())
        weights = (1.0 - beta) / effective_num
        
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return weights


# Example usage and testing functions
if __name__ == "__main__":
    # Test the loss functions
    batch_size = 16
    num_classes = 10
    embed_dim = 256
    
    # Create dummy data
    complex_logits = torch.randn(batch_size, num_classes, dtype=torch.complex64)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test ComplexCrossEntropyLoss
    print("Testing ComplexCrossEntropyLoss...")
    cls_loss_fn = ComplexCrossEntropyLoss()
    cls_loss = cls_loss_fn(complex_logits, targets)
    print(f"Classification loss: {cls_loss.item():.4f}")
    
    # Test FocalLoss
    print("\nTesting FocalLoss...")
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    focal_loss = focal_loss_fn(complex_logits, targets)
    print(f"Focal loss: {focal_loss.item():.4f}")
    
    # Test UncertaintyLoss
    print("\nTesting UncertaintyLoss...")
    uncertainty_components = {
        'epistemic': torch.rand(batch_size, num_classes) * 0.1,
        'aleatoric': torch.rand(batch_size, num_classes) * 0.1,
        'covariance': torch.rand(batch_size, num_classes) * 0.05
    }
    unc_loss_fn = UncertaintyLoss()
    unc_loss_dict = unc_loss_fn(complex_logits, targets, uncertainty_components)
    print(f"Uncertainty loss components: {unc_loss_dict}")
    
    # Test MultiTaskLoss
    print("\nTesting MultiTaskLoss...")
    complex_params = {
        'attention_weights': torch.randn(8, embed_dim, embed_dim, dtype=torch.complex64),
        'linear_weights': torch.randn(embed_dim, num_classes, dtype=torch.complex64)
    }
    multi_loss_fn = MultiTaskLoss()
    multi_loss_dict = multi_loss_fn(
        complex_logits, targets, uncertainty_components, complex_params
    )
    print(f"Multi-task loss: {multi_loss_dict['total'].item():.4f}")
    
    print("\nAll loss function tests completed successfully!")