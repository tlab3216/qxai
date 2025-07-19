"""
Complex-valued neural network layers for Q-XAI framework.
Implements building blocks for complex-valued transformers including complex linear layers,
normalization, activation functions, and positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Union, List, Dict, Any

from utils.complex_math import (
    complex_relu, complex_gelu, complex_silu, complex_tanh,
    ComplexLinear, complex_layer_norm, complex_dropout
)


class ComplexEmbedding(nn.Module):
    """
    Complex-valued embedding layer that converts real inputs to complex representations.
    Implements Equation 2 from the paper: H = 1/√2 * (XW_real + iXW_imag) + (b_real + ib_imag)
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        init_method: str = "complex_glorot",
        scaling_factor: float = 1.0 / math.sqrt(2)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.scaling_factor = scaling_factor
        
        # Real and imaginary weight matrices
        self.weight_real = nn.Parameter(torch.empty(input_dim, embed_dim))
        self.weight_imag = nn.Parameter(torch.empty(input_dim, embed_dim))
        
        # Real and imaginary bias vectors
        self.bias_real = nn.Parameter(torch.empty(embed_dim))
        self.bias_imag = nn.Parameter(torch.empty(embed_dim))
        
        self.init_method = init_method
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using complex-aware initialization."""
        if self.init_method == "complex_glorot":
            # Complex Glorot initialization for quantum-inspired networks
            fan_in = self.input_dim
            fan_out = self.embed_dim
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            
            with torch.no_grad():
                self.weight_real.uniform_(-bound, bound)
                self.weight_imag.uniform_(-bound, bound)
                self.bias_real.uniform_(-bound, bound)
                self.bias_imag.uniform_(-bound, bound)
                
        elif self.init_method == "quantum_inspired":
            # Quantum-inspired initialization with phase diversity
            std = 1.0 / math.sqrt(self.input_dim)
            
            with torch.no_grad():
                # Initialize with random phases
                phases = torch.rand_like(self.weight_real) * 2 * math.pi
                magnitudes = torch.normal(0, std, self.weight_real.shape)
                
                self.weight_real.copy_(magnitudes * torch.cos(phases))
                self.weight_imag.copy_(magnitudes * torch.sin(phases))
                
                # Bias initialization
                bias_phases = torch.rand_like(self.bias_real) * 2 * math.pi
                bias_magnitudes = torch.normal(0, std, self.bias_real.shape)
                
                self.bias_real.copy_(bias_magnitudes * torch.cos(bias_phases))
                self.bias_imag.copy_(bias_magnitudes * torch.sin(bias_phases))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert real input to complex embedding.
        
        Args:
            x: Real input tensor (..., input_dim)
            
        Returns:
            Complex embedding tensor (..., embed_dim)
        """
        # Construct complex weights and bias
        weight = torch.complex(self.weight_real, self.weight_imag)
        bias = torch.complex(self.bias_real, self.bias_imag)
        
        # Linear transformation: x @ weight + bias
        output = F.linear(x, weight.T, bias)
        
        # Apply scaling factor (1/√2 in paper)
        output = output * self.scaling_factor
        
        return output


class ComplexLayerNorm(nn.Module):
    """
    Complex layer normalization that preserves phase relationships.
    Normalizes magnitude while maintaining phase structure.
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            # Complex weight parameters
            self.weight_real = nn.Parameter(torch.ones(normalized_shape))
            self.weight_imag = nn.Parameter(torch.zeros(normalized_shape))
            
            if bias:
                self.bias_real = nn.Parameter(torch.zeros(normalized_shape))
                self.bias_imag = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('bias_real', None)
                self.register_parameter('bias_imag', None)
        else:
            self.register_parameter('weight_real', None)
            self.register_parameter('weight_imag', None)
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply complex layer normalization.
        
        Args:
            x: Complex input tensor
            
        Returns:
            Normalized complex tensor
        """
        # Separate magnitude and phase
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        
        # Normalize magnitude using standard layer norm
        normalized_magnitude = F.layer_norm(
            magnitude, 
            self.normalized_shape, 
            weight=None, 
            bias=None, 
            eps=self.eps
        )
        
        # Reconstruct complex tensor
        normalized = normalized_magnitude * torch.exp(1j * phase)
        
        # Apply learnable affine transformation if enabled
        if self.elementwise_affine:
            weight = torch.complex(self.weight_real, self.weight_imag)
            normalized = normalized * weight
            
            if self.bias_real is not None:
                bias = torch.complex(self.bias_real, self.bias_imag)
                normalized = normalized + bias
        
        return normalized


class ComplexActivation(nn.Module):
    """
    Complex activation functions for non-holomorphic operations.
    These break Cauchy-Riemann conditions, requiring Wirtinger calculus for gradients.
    """
    
    def __init__(self, activation_type: str = "complex_relu"):
        super().__init__()
        self.activation_type = activation_type
        
        # Validate activation type
        valid_activations = ["complex_relu", "complex_gelu", "complex_silu", "complex_tanh"]
        if activation_type not in valid_activations:
            raise ValueError(f"Unknown activation: {activation_type}. Valid options: {valid_activations}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex activation function."""
        if self.activation_type == "complex_relu":
            return complex_relu(x)
        elif self.activation_type == "complex_gelu":
            return complex_gelu(x)
        elif self.activation_type == "complex_silu":
            return complex_silu(x)
        elif self.activation_type == "complex_tanh":
            return complex_tanh(x)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")


class ComplexFeedForward(nn.Module):
    """
    Complex feed-forward network with non-holomorphic activation.
    Standard transformer FFN adapted for complex inputs.
    """
    
    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        activation: str = "complex_gelu",
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        
        # Two linear layers
        self.linear1 = ComplexLinear(embed_dim, ff_dim, bias=bias)
        self.linear2 = ComplexLinear(ff_dim, embed_dim, bias=bias)
        
        # Activation function
        self.activation = ComplexActivation(activation)
        
        # Dropout
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complex FFN.
        
        Args:
            x: Complex input tensor (..., embed_dim)
            
        Returns:
            Complex output tensor (..., embed_dim)
        """
        # First linear layer
        x = self.linear1(x)
        
        # Non-holomorphic activation
        x = self.activation(x)
        
        # Dropout
        if self.training:
            x = complex_dropout(x, p=self.dropout, training=True, coherent=True)
        
        # Second linear layer
        x = self.linear2(x)
        
        return x


class ComplexPositionalEncoding(nn.Module):
    """
    Complex positional encoding for sequential inputs.
    Extends sinusoidal encoding to complex domain for phase-aware position representation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_length: int = 5000,
        encoding_type: str = "complex_sinusoidal",
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.encoding_type = encoding_type
        
        if encoding_type == "complex_sinusoidal":
            self.register_buffer('pe', self._create_complex_sinusoidal_encoding())
        elif encoding_type == "learnable":
            self.pe = nn.Parameter(torch.randn(max_length, embed_dim, dtype=torch.complex64))
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_complex_sinusoidal_encoding(self) -> torch.Tensor:
        """Create complex sinusoidal positional encoding."""
        pe = torch.zeros(self.max_length, self.embed_dim, dtype=torch.complex64)
        
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        
        # Create different frequency scales for real and imaginary parts
        div_term_real = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * 
            -(math.log(10000.0) / self.embed_dim)
        )
        div_term_imag = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * 
            -(math.log(8000.0) / self.embed_dim)  # Slightly different frequency for phase diversity
        )
        
        # Real part: standard sinusoidal
        pe_real = torch.zeros(self.max_length, self.embed_dim)
        pe_real[:, 0::2] = torch.sin(position * div_term_real)
        pe_real[:, 1::2] = torch.cos(position * div_term_real)
        
        # Imaginary part: shifted sinusoidal for phase diversity
        pe_imag = torch.zeros(self.max_length, self.embed_dim)
        pe_imag[:, 0::2] = torch.sin(position * div_term_imag + math.pi/4)  # Phase shift
        pe_imag[:, 1::2] = torch.cos(position * div_term_imag + math.pi/4)
        
        # Combine into complex encoding
        pe = torch.complex(pe_real, pe_imag)
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Complex input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Input with positional encoding added
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_length}")
        
        # Add positional encoding
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        
        # Apply dropout to real and imaginary parts separately but coherently
        if self.training:
            x = complex_dropout(x, p=self.dropout.p, training=True, coherent=True)
        
        return x


class ComplexMultiHeadProjection(nn.Module):
    """
    Complex multi-head projection for query, key, value computation.
    Projects input to multiple attention heads in complex space.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        projection_type: str = "linear"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        
        self.projection_type = projection_type
        
        if projection_type == "linear":
            self.projection = ComplexLinear(embed_dim, embed_dim, bias=bias)
        elif projection_type == "unitary":
            # Unitary projection for quantum-inspired attention
            self.projection = ComplexUnitaryLayer(embed_dim, embed_dim)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input for multi-head attention.
        
        Args:
            x: Complex input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Projected tensor reshaped for multi-head attention
            (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        projected = self.projection(x)  # (batch_size, seq_len, embed_dim)
        
        # Reshape for multi-head attention
        projected = projected.view(batch_size, seq_len, self.num_heads, self.head_dim)
        projected = projected.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        return projected


class ComplexUnitaryLayer(nn.Module):
    """
    Complex unitary transformation layer.
    Preserves quantum properties by maintaining unitary operations.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        parameterization: str = "cayley"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.parameterization = parameterization
        
        if parameterization == "cayley":
            # Cayley parameterization: U = (I + iA)(I - iA)^(-1)
            self.A = nn.Parameter(torch.randn(output_dim, input_dim))
        elif parameterization == "householder":
            # Householder reflection parameterization
            self.v = nn.Parameter(torch.randn(output_dim, input_dim, dtype=torch.complex64))
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply unitary transformation."""
        if self.parameterization == "cayley":
            # Construct unitary matrix using Cayley transform
            I = torch.eye(self.output_dim, device=self.A.device, dtype=torch.complex64)
            iA = 1j * torch.complex(self.A, torch.zeros_like(self.A))
            
            U = torch.linalg.solve(I - iA, I + iA)
            
        elif self.parameterization == "householder":
            # Householder reflection: U = I - 2vv^H / (v^H v)
            v = self.v
            v_conj = torch.conj(v)
            
            # Normalize
            norm_sq = torch.sum(v_conj * v, dim=-1, keepdim=True)
            
            U = (torch.eye(self.output_dim, device=v.device, dtype=torch.complex64) - 
                 2 * torch.matmul(v, v_conj) / (norm_sq + 1e-8))
        
        # Apply transformation
        return F.linear(x, U)


class ComplexResidualConnection(nn.Module):
    """
    Complex residual connection with layer normalization.
    Implements residual connections for complex-valued layers.
    """
    
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        prenorm: bool = True
    ):
        super().__init__()
        self.norm = ComplexLayerNorm(embed_dim)
        self.dropout = dropout
        self.prenorm = prenorm
    
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Apply residual connection around sublayer.
        
        Args:
            x: Complex input tensor
            sublayer: Sublayer to apply (e.g., attention or FFN)
            
        Returns:
            Output with residual connection
        """
        if self.prenorm:
            # Pre-normalization: norm -> sublayer -> dropout -> residual
            normalized = self.norm(x)
            output = sublayer(normalized)
            if self.training:
                output = complex_dropout(output, p=self.dropout, training=True, coherent=True)
            return x + output
        else:
            # Post-normalization: sublayer -> dropout -> residual -> norm
            output = sublayer(x)
            if self.training:
                output = complex_dropout(output, p=self.dropout, training=True, coherent=True)
            return self.norm(x + output)


class ComplexGatedLinearUnit(nn.Module):
    """
    Complex Gated Linear Unit (GLU) for improved expressiveness.
    Implements gating mechanism in complex domain.
    """
    
    def __init__(
        self,
        input_dim: int,
        gate_activation: str = "complex_sigmoid"
    ):
        super().__init__()
        self.input_dim = input_dim
        
        # Linear projections for value and gate
        self.value_proj = ComplexLinear(input_dim, input_dim)
        self.gate_proj = ComplexLinear(input_dim, input_dim)
        
        # Gate activation
        if gate_activation == "complex_sigmoid":
            self.gate_activation = lambda x: torch.sigmoid(x.real) + 1j * torch.sigmoid(x.imag)
        elif gate_activation == "complex_tanh":
            self.gate_activation = complex_tanh
        else:
            raise ValueError(f"Unknown gate activation: {gate_activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex gated linear unit."""
        value = self.value_proj(x)
        gate = self.gate_activation(self.gate_proj(x))
        
        return value * gate


class ComplexDropPath(nn.Module):
    """
    Complex drop path (stochastic depth) for regularization.
    Randomly drops entire residual paths during training.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop path to complex tensor."""
        if not self.training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with diff dim tensors, not just 2D
        
        # Create random tensor for path dropping
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        
        output = x.div(keep_prob) * random_tensor
        
        return output


class ComplexAttentionPooling(nn.Module):
    """
    Complex attention-based pooling for sequence-to-vector conversion.
    Useful for classification tasks.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        temperature: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Query vector for attention pooling
        self.query = nn.Parameter(torch.randn(num_heads, embed_dim // num_heads, dtype=torch.complex64))
        
        # Key and value projections
        self.key_proj = ComplexMultiHeadProjection(embed_dim, num_heads)
        self.value_proj = ComplexMultiHeadProjection(embed_dim, num_heads)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool sequence using complex attention.
        
        Args:
            x: Complex input sequence (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Pooled representation (batch_size, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to keys and values
        keys = self.key_proj(x)    # (batch_size, num_heads, seq_len, head_dim)
        values = self.value_proj(x)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Expand query for batch
        queries = self.query.unsqueeze(0).unsqueeze(2)  # (1, num_heads, 1, head_dim)
        queries = queries.expand(batch_size, -1, -1, -1)
        
        # Compute attention scores using Hermitian product
        scores = torch.sum(torch.conj(queries) * keys, dim=-1) / self.temperature  # (batch_size, num_heads, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax on real part (as in paper Equation 5)
        attention_weights = F.softmax(scores.real, dim=-1)
        
        # Apply attention to values
        pooled = torch.sum(attention_weights.unsqueeze(-1) * values, dim=2)  # (batch_size, num_heads, head_dim)
        
        # Concatenate heads
        pooled = pooled.view(batch_size, self.embed_dim)
        
        return pooled


def create_complex_layer(
    layer_type: str,
    embed_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create complex layers.
    
    Args:
        layer_type: Type of layer to create
        embed_dim: Embedding dimension
        **kwargs: Additional layer-specific arguments
        
    Returns:
        Created complex layer
    """
    if layer_type == "embedding":
        input_dim = kwargs.get('input_dim', embed_dim)
        return ComplexEmbedding(input_dim, embed_dim, **kwargs)
    
    elif layer_type == "layer_norm":
        return ComplexLayerNorm(embed_dim, **kwargs)
    
    elif layer_type == "activation":
        activation_type = kwargs.get('activation_type', 'complex_relu')
        return ComplexActivation(activation_type)
    
    elif layer_type == "feedforward":
        ff_dim = kwargs.get('ff_dim', 4 * embed_dim)
        return ComplexFeedForward(embed_dim, ff_dim, **kwargs)
    
    elif layer_type == "positional_encoding":
        return ComplexPositionalEncoding(embed_dim, **kwargs)
    
    elif layer_type == "residual":
        return ComplexResidualConnection(embed_dim, **kwargs)
    
    elif layer_type == "glu":
        return ComplexGatedLinearUnit(embed_dim, **kwargs)
    
    elif layer_type == "attention_pooling":
        return ComplexAttentionPooling(embed_dim, **kwargs)
    
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


# Utility functions for complex layer operations

def init_complex_weights(
    tensor: torch.Tensor,
    method: str = "complex_glorot",
    gain: float = 1.0
) -> None:
    """Initialize complex weights in-place."""
    if method == "complex_glorot":
        fan_in = tensor.shape[-2] if tensor.dim() >= 2 else tensor.shape[-1]
        fan_out = tensor.shape[-1]
        bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
        
        with torch.no_grad():
            tensor.real.uniform_(-bound, bound)
            tensor.imag.uniform_(-bound, bound)
            
    elif method == "complex_normal":
        std = gain / math.sqrt(tensor.shape[-1])
        with torch.no_grad():
            tensor.real.normal_(0, std)
            tensor.imag.normal_(0, std)
            
    elif method == "quantum_inspired":
        # Random phase initialization
        with torch.no_grad():
            phases = torch.rand_like(tensor.real) * 2 * math.pi
            magnitudes = torch.normal(0, gain / math.sqrt(tensor.shape[-1]), tensor.real.shape)
            
            tensor.real.copy_(magnitudes * torch.cos(phases))
            tensor.imag.copy_(magnitudes * torch.sin(phases))


def check_complex_gradients(model: nn.Module) -> Dict[str, Any]:
    """
    Check complex gradient flow in model.
    Useful for debugging complex-valued training.
    """
    gradient_info = {
        'has_gradients': {},
        'gradient_norms': {},
        'gradient_phases': {},
        'total_parameters': 0,
        'parameters_with_gradients': 0
    }
    
    for name, param in model.named_parameters():
        gradient_info['total_parameters'] += param.numel()
        
        if param.grad is not None:
            gradient_info['parameters_with_gradients'] += param.numel()
            gradient_info['has_gradients'][name] = True
            
            if torch.is_complex(param.grad):
                # Complex gradient analysis
                grad_norm = torch.norm(param.grad).item()
                grad_phase = torch.angle(param.grad)
                
                gradient_info['gradient_norms'][name] = grad_norm
                gradient_info['gradient_phases'][name] = {
                    'mean_phase': torch.mean(grad_phase).item(),
                    'phase_std': torch.std(grad_phase).item()
                }
            else:
                # Real gradient
                gradient_info['gradient_norms'][name] = torch.norm(param.grad).item()
                gradient_info['gradient_phases'][name] = None
        else:
            gradient_info['has_gradients'][name] = False
    
    return gradient_info