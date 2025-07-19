"""
Complex mathematical operations for Q-XAI framework.
Implements Wirtinger calculus and complex-valued operations for interpretable complex neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Union, List, Dict, Any


class WirtingerGradient:
    """
    Implements Wirtinger calculus for complex-valued functions.
    
    For a function f: C -> R that depends on complex variable z = x + iy,
    the Wirtinger derivatives are:
    ∂f/∂z = 1/2 * (∂f/∂x - i∂f/∂y)
    ∂f/∂z* = 1/2 * (∂f/∂x + i∂f/∂y)
    
    This is essential for Q-XAI's QISA component.
    """
    
    @staticmethod
    def compute_wirtinger_derivatives(
        loss: torch.Tensor,
        complex_tensor: torch.Tensor,
        create_graph: bool = False,
        retain_graph: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Wirtinger derivatives of loss w.r.t. complex tensor.
        
        Args:
            loss: Real-valued scalar loss
            complex_tensor: Complex tensor to compute gradients for
            create_graph: Whether to create computation graph for higher-order derivatives
            retain_graph: Whether to retain computation graph
            
        Returns:
            Tuple of (∂L/∂z, ∂L/∂z*) derivatives
        """
        if not complex_tensor.requires_grad:
            raise ValueError("Complex tensor must require gradients")
        
        # Extract real and imaginary parts
        real_part = complex_tensor.real
        imag_part = complex_tensor.imag
        
        # Ensure parts require gradients
        if not real_part.requires_grad:
            real_part.requires_grad_(True)
        if not imag_part.requires_grad:
            imag_part.requires_grad_(True)
        
        # Compute gradients w.r.t. real and imaginary parts
        grad_outputs = torch.ones_like(loss)
        
        grad_real = torch.autograd.grad(
            outputs=loss,
            inputs=real_part,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        grad_imag = torch.autograd.grad(
            outputs=loss,
            inputs=imag_part,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=True
        )[0]
        
        # Handle case where gradients are None
        if grad_real is None:
            grad_real = torch.zeros_like(real_part)
        if grad_imag is None:
            grad_imag = torch.zeros_like(imag_part)
        
        # Compute Wirtinger derivatives
        # ∂f/∂z = 1/2 * (∂f/∂x - i∂f/∂y)
        grad_z = 0.5 * torch.complex(grad_real, -grad_imag)
        
        # ∂f/∂z* = 1/2 * (∂f/∂x + i∂f/∂y)
        grad_z_conj = 0.5 * torch.complex(grad_real, grad_imag)
        
        return grad_z, grad_z_conj
    
    @staticmethod
    def compute_attribution_magnitude(
        grad_z: torch.Tensor,
        grad_z_conj: torch.Tensor,
        method: str = "squared_magnitude"
    ) -> torch.Tensor:
        """
        Compute attribution magnitude from Wirtinger derivatives.
        
        Args:
            grad_z: Holomorphic gradient ∂f/∂z
            grad_z_conj: Anti-holomorphic gradient ∂f/∂z*
            method: Attribution computation method
            
        Returns:
            Real-valued attribution tensor
        """
        if method == "squared_magnitude":
            # ||∂f/∂z||² + ||∂f/∂z*||² (Equation 8 in paper)
            attribution = torch.abs(grad_z)**2 + torch.abs(grad_z_conj)**2
            
        elif method == "magnitude":
            # ||∂f/∂z|| + ||∂f/∂z*||
            attribution = torch.abs(grad_z) + torch.abs(grad_z_conj)
            
        elif method == "real_part":
            # Re(∂f/∂z) + Re(∂f/∂z*)
            attribution = grad_z.real + grad_z_conj.real
            
        elif method == "euclidean_norm":
            # ||[∂f/∂z, ∂f/∂z*]||₂ (Euclidean norm of gradient vector)
            attribution = torch.sqrt(torch.abs(grad_z)**2 + torch.abs(grad_z_conj)**2)
            
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        return attribution


def complex_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Complex ReLU (CReLU) activation function.
    Applies ReLU to both real and imaginary parts independently.
    This is a non-holomorphic function that breaks Cauchy-Riemann conditions.
    """
    return torch.complex(
        torch.relu(x.real),
        torch.relu(x.imag)
    )


def complex_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Complex GELU activation function.
    Applies GELU to both real and imaginary parts independently.
    """
    return torch.complex(
        F.gelu(x.real),
        F.gelu(x.imag)
    )


def complex_silu(x: torch.Tensor) -> torch.Tensor:
    """
    Complex SiLU (Swish) activation function.
    Applies SiLU to both real and imaginary parts independently.
    """
    return torch.complex(
        F.silu(x.real),
        F.silu(x.imag)
    )


def complex_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Complex hyperbolic tangent.
    This is actually holomorphic, unlike the split activations above.
    """
    # tanh(a + bi) = (tanh(a) + i*tan(b)) / (1 + i*tanh(a)*tan(b))
    real_part = x.real
    imag_part = x.imag
    
    tanh_real = torch.tanh(real_part)
    tan_imag = torch.tan(imag_part)
    
    denominator = 1 + 1j * tanh_real * tan_imag
    numerator = tanh_real + 1j * tan_imag
    
    return numerator / denominator


def hermitian_product(q: torch.Tensor, k: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute Hermitian inner product between query and key tensors.
    
    Args:
        q: Query tensor of shape (..., d)
        k: Key tensor of shape (..., d)
        dim: Dimension along which to compute the product
        
    Returns:
        Hermitian product q^H @ k = conj(q) @ k
    """
    # Ensure tensors are complex
    if not torch.is_complex(q):
        q = q.to(torch.complex64)
    if not torch.is_complex(k):
        k = k.to(torch.complex64)
    
    # Compute q^H @ k = conj(q) @ k
    hermitian_prod = torch.sum(torch.conj(q) * k, dim=dim)
    
    return hermitian_prod


def complex_softmax(x: torch.Tensor, dim: int = -1, method: str = "real_part") -> torch.Tensor:
    """
    Complex softmax using different methods.
    
    Args:
        x: Complex input tensor
        dim: Dimension to apply softmax
        method: Method for complex softmax computation
        
    Returns:
        Complex softmax output
    """
    if method == "real_part":
        # Use real part for softmax (Equation 5 in paper)
        real_part = x.real
        softmax_weights = F.softmax(real_part, dim=dim)
        
        # Apply weights while preserving phase structure
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        
        # Scale magnitude by softmax weights
        scaled_magnitude = softmax_weights * magnitude
        
        return scaled_magnitude * torch.exp(1j * phase)
    
    elif method == "magnitude":
        # Use magnitude for softmax
        magnitude = torch.abs(x)
        softmax_weights = F.softmax(magnitude, dim=dim)
        
        phase = torch.angle(x)
        return softmax_weights * torch.exp(1j * phase)
    
    elif method == "holomorphic":
        # Truly holomorphic complex softmax
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    
    else:
        raise ValueError(f"Unknown complex softmax method: {method}")


def amplitude_to_probability(complex_amplitudes: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Convert complex amplitudes to probabilities using Born rule.
    P(class) = |amplitude|^2 (Equation 12 in paper)
    
    Args:
        complex_amplitudes: Complex amplitude tensor
        dim: Dimension to normalize probabilities
        
    Returns:
        Real-valued probability tensor
    """
    # Compute squared magnitudes (Born rule)
    probabilities = torch.abs(complex_amplitudes) ** 2
    
    # Normalize to ensure probabilities sum to 1
    prob_sum = torch.sum(probabilities, dim=dim, keepdim=True)
    normalized_probs = probabilities / (prob_sum + 1e-8)
    
    return normalized_probs


def complex_dropout(
    x: torch.Tensor, 
    p: float = 0.5, 
    training: bool = True,
    coherent: bool = True
) -> torch.Tensor:
    """
    Complex dropout that can preserve phase relationships.
    
    Args:
        x: Complex input tensor
        p: Dropout probability
        training: Whether in training mode
        coherent: Whether to apply same mask to real and imaginary parts
        
    Returns:
        Dropout-applied complex tensor
    """
    if not training or p == 0:
        return x
    
    if coherent:
        # Apply same dropout mask to both real and imaginary parts
        # This preserves phase relationships
        mask = torch.bernoulli(torch.full_like(x.real, 1 - p))
        keep_prob = 1 - p
        return x * mask.unsqueeze(-1) / keep_prob
    else:
        # Apply independent dropout to real and imaginary parts
        real_mask = torch.bernoulli(torch.full_like(x.real, 1 - p))
        imag_mask = torch.bernoulli(torch.full_like(x.imag, 1 - p))
        
        keep_prob = 1 - p
        real_part = x.real * real_mask / keep_prob
        imag_part = x.imag * imag_mask / keep_prob
        
        return torch.complex(real_part, imag_part)


def compute_covariance_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance uncertainty between real and imaginary parts.
    This is the novel σ²_cov term in AUQ (Equation 11 in paper).
    
    Args:
        predictions: Complex predictions of shape (M, batch_size, num_classes)
        
    Returns:
        Covariance uncertainty tensor of shape (batch_size, num_classes)
    """
    M, batch_size, num_classes = predictions.shape
    
    real_parts = predictions.real  # (M, batch_size, num_classes)
    imag_parts = predictions.imag  # (M, batch_size, num_classes)
    
    # Compute covariance for each sample and class
    covariances = torch.zeros(batch_size, num_classes, device=predictions.device)
    
    for b in range(batch_size):
        for c in range(num_classes):
            real_vals = real_parts[:, b, c]  # (M,)
            imag_vals = imag_parts[:, b, c]  # (M,)
            
            # Center the values
            real_centered = real_vals - torch.mean(real_vals)
            imag_centered = imag_vals - torch.mean(imag_vals)
            
            # Compute covariance
            cov = torch.mean(real_centered * imag_centered)
            covariances[b, c] = torch.abs(cov)
    
    return covariances


def complex_batch_norm(
    x: torch.Tensor,
    running_mean: Optional[torch.Tensor] = None,
    running_var: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Complex batch normalization.
    Normalizes magnitude while preserving phase relationships.
    """
    # Get magnitude and phase
    magnitude = torch.abs(x)
    phase = torch.angle(x)
    
    # Apply batch normalization to magnitude
    normalized_magnitude = F.batch_norm(
        magnitude,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps
    )
    
    # Reconstruct complex tensor
    return normalized_magnitude * torch.exp(1j * phase)


def complex_layer_norm(
    x: torch.Tensor,
    normalized_shape: Union[int, List[int], torch.Size],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Complex layer normalization.
    Normalizes magnitude while preserving phase relationships.
    """
    # Get magnitude and phase
    magnitude = torch.abs(x)
    phase = torch.angle(x)
    
    # Apply layer normalization to magnitude
    normalized_magnitude = F.layer_norm(
        magnitude,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps
    )
    
    # Reconstruct complex tensor
    return normalized_magnitude * torch.exp(1j * phase)


class ComplexLinear(nn.Module):
    """
    Complex linear layer that properly handles complex matrix multiplication.
    Implements: y = Wx + b where W and b are complex.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        init_method: str = "complex_glorot"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method
        
        # Initialize complex weights as separate real and imaginary parameters
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using complex-aware initialization."""
        if self.init_method == "complex_glorot":
            # Complex Glorot (Xavier) initialization
            # For complex networks, we need to account for the complex nature
            fan_in = self.in_features
            fan_out = self.out_features
            
            # For complex parameters, the effective fan is doubled
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            
            with torch.no_grad():
                self.weight_real.uniform_(-bound, bound)
                self.weight_imag.uniform_(-bound, bound)
                
                if self.bias_real is not None:
                    self.bias_real.uniform_(-bound, bound)
                    self.bias_imag.uniform_(-bound, bound)
                    
        elif self.init_method == "complex_normal":
            # Complex normal initialization
            std = 1.0 / math.sqrt(self.in_features)
            
            with torch.no_grad():
                self.weight_real.normal_(0, std)
                self.weight_imag.normal_(0, std)
                
                if self.bias_real is not None:
                    self.bias_real.normal_(0, std)
                    self.bias_imag.normal_(0, std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complex linear transformation.
        
        Args:
            x: Complex input tensor (..., in_features)
            
        Returns:
            Complex output tensor (..., out_features)
        """
        # Construct complex weight and bias
        weight = torch.complex(self.weight_real, self.weight_imag)
        
        if self.bias_real is not None:
            bias = torch.complex(self.bias_real, self.bias_imag)
        else:
            bias = None
        
        # Perform complex linear transformation
        return F.linear(x, weight, bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_real is not None}'


def complex_matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Efficient complex matrix multiplication.
    (a_r + i*a_i) @ (b_r + i*b_i) = (a_r@b_r - a_i@b_i) + i*(a_r@b_i + a_i@b_r)
    
    This can be more efficient than PyTorch's built-in complex matmul for certain cases.
    """
    a_real, a_imag = a.real, a.imag
    b_real, b_imag = b.real, b.imag
    
    # Real part: a_r @ b_r - a_i @ b_i
    real_part = torch.matmul(a_real, b_real) - torch.matmul(a_imag, b_imag)
    
    # Imaginary part: a_r @ b_i + a_i @ b_r
    imag_part = torch.matmul(a_real, b_imag) + torch.matmul(a_imag, b_real)
    
    return torch.complex(real_part, imag_part)


def complex_eigendecomposition(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute eigendecomposition of complex matrix.
    
    Args:
        matrix: Complex square matrix
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # Use PyTorch's built-in eigendecomposition
    eigenvals, eigenvecs = torch.linalg.eig(matrix)
    
    return eigenvals, eigenvecs


def stable_complex_division(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable complex division.
    
    Args:
        numerator: Complex numerator tensor
        denominator: Complex denominator tensor
        eps: Small value to prevent division by zero
        
    Returns:
        Result of numerator / denominator
    """
    # For complex division: (a + bi) / (c + di) = ((ac + bd) + i(bc - ad)) / (c² + d²)
    a, b = numerator.real, numerator.imag
    c, d = denominator.real, denominator.imag
    
    # Denominator magnitude squared
    denom_mag_sq = c**2 + d**2 + eps
    
    # Real and imaginary parts of result
    real_part = (a * c + b * d) / denom_mag_sq
    imag_part = (b * c - a * d) / denom_mag_sq
    
    return torch.complex(real_part, imag_part)


def quantum_interference_attention_score(
    query: torch.Tensor, 
    key: torch.Tensor, 
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Compute quantum-inspired interference attention scores.
    Implements the attention mechanism from Equations 4-5 in the paper.
    
    Args:
        query: Complex query tensor (..., d)
        key: Complex key tensor (..., d)
        scale: Optional scaling factor (default: 1/sqrt(d))
        
    Returns:
        Complex attention scores
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    
    # Compute Hermitian product: q^H @ k
    attention_scores = hermitian_product(query, key, dim=-1)
    
    # Apply scaling
    attention_scores = attention_scores * scale
    
    return attention_scores


def apply_quantum_interference(
    amplitudes: torch.Tensor, 
    phase_shifts: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply quantum-like interference to complex amplitudes.
    
    Args:
        amplitudes: Complex amplitude tensor
        phase_shifts: Optional phase shifts to apply
        
    Returns:
        Interfered amplitudes
    """
    if phase_shifts is not None:
        # Apply phase shifts
        amplitudes = amplitudes * torch.exp(1j * phase_shifts)
    
    # Interference is naturally captured by complex arithmetic
    # When amplitudes are added, their phases determine constructive/destructive interference
    return amplitudes


class ComplexParameterizedFunction(nn.Module):
    """
    A parameterized complex function for learning complex transformations.
    Can be used for learnable complex activations or transformations.
    """
    
    def __init__(self, num_params: int = 2):
        super().__init__()
        # Learnable complex parameters
        self.params_real = nn.Parameter(torch.randn(num_params))
        self.params_imag = nn.Parameter(torch.randn(num_params))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable complex transformation."""
        params = torch.complex(self.params_real, self.params_imag)
        
        # Example: learnable complex polynomial
        result = x.clone()
        for i, param in enumerate(params):
            result = result + param * (x ** (i + 1))
        
        return result