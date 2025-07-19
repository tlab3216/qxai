"""
Quantum-inspired attention mechanisms for Q-XAI framework.
Implements complex-valued attention with interference patterns as described in the paper.
Includes Hermitian products, phase-aware attention, and quantum-inspired operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Union, List, Dict, Any

from utils.complex_math import (
    hermitian_product, complex_softmax, amplitude_to_probability,
    quantum_interference_attention_score, complex_dropout
)
from models.complex_layers import ComplexLinear, ComplexMultiHeadProjection


class QuantumInspiredAttention(nn.Module):
    """
    Quantum-inspired attention mechanism implementing Equations 4-5 from the paper.
    
    Uses Hermitian inner products and interference patterns to model
    quantum-like attention in complex-valued transformers.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        attention_method: str = "hermitian_interference",
        temperature_scaling: bool = True,
        phase_attention: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.attention_method = attention_method
        self.temperature_scaling = temperature_scaling
        self.phase_attention = phase_attention
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        
        # Scaling factor (1/√d in paper)
        self.scale = 1.0 / math.sqrt(self.head_dim) if temperature_scaling else 1.0
        
        # Query, Key, Value projections
        self.q_proj = ComplexLinear(embed_dim, embed_dim, bias=bias)
        self.k_proj = ComplexLinear(embed_dim, embed_dim, bias=bias)
        self.v_proj = ComplexLinear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = ComplexLinear(embed_dim, embed_dim, bias=bias)
        
        # Phase attention parameters (if enabled)
        if phase_attention:
            self.phase_weight = nn.Parameter(torch.tensor(0.5))  # Learnable phase weighting
        
        # Attention dropout
        self.attn_dropout = dropout
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass implementing quantum-inspired attention.
        
        Args:
            query: Query tensor (batch_size, tgt_len, embed_dim)
            key: Key tensor (batch_size, src_len, embed_dim)
            value: Value tensor (batch_size, src_len, embed_dim)
            attn_mask: Attention mask
            key_padding_mask: Key padding mask
            need_weights: Whether to return attention weights
            average_attn_weights: Whether to average attention weights across heads
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, tgt_len, embed_dim = query.shape
        src_len = key.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # (batch_size, tgt_len, embed_dim)
        K = self.k_proj(key)    # (batch_size, src_len, embed_dim)
        V = self.v_proj(value)  # (batch_size, src_len, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute quantum-inspired attention
        attended_values, attn_weights = self._quantum_attention(
            Q, K, V, attn_mask, key_padding_mask
        )
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, embed_dim
        )
        
        # Output projection
        output = self.out_proj(attended_values)
        
        # Process attention weights for return
        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)  # Average across heads
            return output, attn_weights
        else:
            return output, None
    
    def _quantum_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core quantum-inspired attention computation.
        Implements Equations 4-5 from the paper.
        """
        batch_size, num_heads, tgt_len, head_dim = Q.shape
        src_len = K.shape[2]
        
        if self.attention_method == "hermitian_interference":
            # Method from paper: Hermitian product + interference
            attended_values, attn_weights = self._hermitian_interference_attention(
                Q, K, V, attn_mask, key_padding_mask
            )
            
        elif self.attention_method == "born_rule":
            # Alternative: Born rule-based attention
            attended_values, attn_weights = self._born_rule_attention(
                Q, K, V, attn_mask, key_padding_mask
            )
            
        elif self.attention_method == "quantum_superposition":
            # Quantum superposition-based attention
            attended_values, attn_weights = self._superposition_attention(
                Q, K, V, attn_mask, key_padding_mask
            )
        else:
            raise ValueError(f"Unknown attention method: {self.attention_method}")
        
        return attended_values, attn_weights
    
    def _hermitian_interference_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the main attention mechanism from Equations 4-5 in the paper.
        
        S_ij = (Q_i^H @ K_j) / √d  (Equation 4)
        A_ij = exp(Re(S_ij)) / Σ_k exp(Re(S_ik))  (Equation 5)
        """
        # Compute Hermitian inner products (Equation 4)
        # Q: (batch_size, num_heads, tgt_len, head_dim)
        # K: (batch_size, num_heads, src_len, head_dim)
        
        # Compute Q^H @ K for all pairs
        attn_scores = torch.zeros(
            Q.shape[0], Q.shape[1], Q.shape[2], K.shape[2],
            dtype=torch.complex64, device=Q.device
        )
        
        for i in range(Q.shape[2]):  # For each query position
            for j in range(K.shape[2]):  # For each key position
                # Hermitian product: Q_i^H @ K_j
                q_i = Q[:, :, i, :]  # (batch_size, num_heads, head_dim)
                k_j = K[:, :, j, :]  # (batch_size, num_heads, head_dim)
                
                # Compute conjugate(q_i) * k_j and sum over head_dim
                hermitian_prod = torch.sum(torch.conj(q_i) * k_j, dim=-1)  # (batch_size, num_heads)
                attn_scores[:, :, i, j] = hermitian_prod
        
        # Apply scaling (√d in denominator)
        attn_scores = attn_scores * self.scale
        
        # Extract real part for attention weights (Equation 5)
        real_scores = attn_scores.real  # (batch_size, num_heads, tgt_len, src_len)
        
        # Apply masks
        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, src_len)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
            real_scores = real_scores.masked_fill(key_padding_mask, float('-inf'))
        
        if attn_mask is not None:
            real_scores = real_scores + attn_mask
        
        # Softmax on real part (Equation 5)
        attn_weights = F.softmax(real_scores, dim=-1)  # (batch_size, num_heads, tgt_len, src_len)
        
        # Apply attention dropout
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout)
        
        # Apply attention to values
        attended_values = torch.matmul(attn_weights, V)  # (batch_size, num_heads, tgt_len, head_dim)
        
        # Add phase attention if enabled
        if self.phase_attention:
            attended_values = self._apply_phase_attention(
                attended_values, attn_scores, attn_weights
            )
        
        return attended_values, attn_weights
    
    def _born_rule_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Born rule-based attention: P(i,j) = |Q_i^H @ K_j|^2
        Uses squared magnitude for attention weights.
        """
        # Compute Hermitian products
        Q_expanded = Q.unsqueeze(3)  # (batch, heads, tgt_len, 1, head_dim)
        K_expanded = K.unsqueeze(2)  # (batch, heads, 1, src_len, head_dim)
        
        # Hermitian product for all pairs
        hermitian_products = torch.sum(
            torch.conj(Q_expanded) * K_expanded, dim=-1
        )  # (batch, heads, tgt_len, src_len)
        
        # Born rule: |amplitude|^2
        attn_scores = torch.abs(hermitian_products) ** 2
        
        # Apply scaling
        attn_scores = attn_scores * self.scale
        
        # Apply masks
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, 0.0)
        
        # Normalize to get probabilities
        attn_weights = attn_scores / (torch.sum(attn_scores, dim=-1, keepdim=True) + 1e-8)
        
        # Apply attention dropout
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout)
        
        # Apply to values
        attended_values = torch.matmul(attn_weights, V)
        
        return attended_values, attn_weights
    
    def _superposition_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantum superposition-based attention.
        Creates superposition states from keys and queries.
        """
        # Create superposition of query and key states
        # |ψ⟩ = α|q⟩ + β|k⟩
        
        # Normalize queries and keys
        Q_norm = Q / (torch.norm(Q, dim=-1, keepdim=True) + 1e-8)
        K_norm = K / (torch.norm(K, dim=-1, keepdim=True) + 1e-8)
        
        # Superposition coefficients (learnable)
        alpha = 1.0 / math.sqrt(2)
        beta = 1.0 / math.sqrt(2)
        
        # Compute superposition overlap
        superposition_scores = torch.zeros(
            Q.shape[0], Q.shape[1], Q.shape[2], K.shape[2],
            device=Q.device
        )
        
        for i in range(Q.shape[2]):
            for j in range(K.shape[2]):
                q_i = Q_norm[:, :, i, :]
                k_j = K_norm[:, :, j, :]
                
                # Superposition state overlap
                overlap = torch.abs(torch.sum(
                    torch.conj(alpha * q_i + beta * k_j) * (alpha * q_i + beta * k_j),
                    dim=-1
                ))
                
                superposition_scores[:, :, i, j] = overlap
        
        # Convert to attention weights
        attn_weights = F.softmax(superposition_scores * self.scale, dim=-1)
        
        # Apply masks and dropout as before
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, 0.0)
        
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout)
        
        # Apply to values
        attended_values = torch.matmul(attn_weights, V)
        
        return attended_values, attn_weights
    
    def _apply_phase_attention(
        self,
        attended_values: torch.Tensor,
        attn_scores: torch.Tensor,
        attn_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply phase-aware attention modulation.
        Uses phase information from attention scores to modulate outputs.
        """
        # Extract phase from complex attention scores
        phase_info = torch.angle(attn_scores)  # (batch, heads, tgt_len, src_len)
        
        # Compute phase-weighted attention
        phase_weights = torch.cos(phase_info) * self.phase_weight
        
        # Apply phase modulation
        phase_modulation = torch.matmul(
            attn_weights * phase_weights, 
            torch.ones_like(attended_values)
        )
        
        # Modulate the attended values
        modulated_values = attended_values * (1.0 + phase_modulation)
        
        return modulated_values


class MultiHeadQuantumAttention(nn.Module):
    """
    Multi-head quantum-inspired attention with advanced features.
    Includes relative position encoding and cross-attention capabilities.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        relative_attention: bool = False,
        max_relative_positions: int = 128
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.relative_attention = relative_attention
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Core quantum attention
        self.quantum_attention = QuantumInspiredAttention(
            embed_dim, num_heads, dropout, bias
        )
        
        # Relative position encoding (if enabled)
        if relative_attention:
            self.max_relative_positions = max_relative_positions
            self.relative_position_k = ComplexLinear(
                max_relative_positions, self.head_dim, bias=False
            )
            self.relative_position_v = ComplexLinear(
                max_relative_positions, self.head_dim, bias=False
            )
        
        # Additional key and value biases (if enabled)
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros(1, 1, 1, embed_dim, dtype=torch.complex64))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, 1, embed_dim, dtype=torch.complex64))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        static_k: Optional[torch.Tensor] = None,
        static_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head quantum attention.
        
        Args:
            query: Query tensor
            key: Key tensor (defaults to query for self-attention)
            value: Value tensor (defaults to key)
            key_padding_mask: Mask for padded keys
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average weights across heads
            static_k: Static key for incremental decoding
            static_v: Static value for incremental decoding
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Set defaults for self-attention
        if key is None:
            key = query
        if value is None:
            value = key
        
        # Handle static keys/values for incremental decoding
        if static_k is not None:
            key = static_k
        if static_v is not None:
            value = static_v
        
        # Add bias keys and values if enabled
        if self.bias_k is not None:
            key = torch.cat([
                key, 
                self.bias_k.expand(key.shape[0], -1, -1, -1)
            ], dim=1)
            
        if self.bias_v is not None:
            value = torch.cat([
                value,
                self.bias_v.expand(value.shape[0], -1, -1, -1)
            ], dim=1)
            
            # Update padding mask for bias
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([
                    key_padding_mask,
                    torch.zeros(key_padding_mask.shape[0], 1, 
                              dtype=key_padding_mask.dtype, 
                              device=key_padding_mask.device)
                ], dim=1)
        
        # Add zero attention if enabled
        if self.add_zero_attn:
            zero_attn_shape = (key.shape[0], 1, key.shape[2])
            key = torch.cat([
                key,
                torch.zeros(zero_attn_shape, dtype=key.dtype, device=key.device)
            ], dim=1)
            
            value = torch.cat([
                value,
                torch.zeros(zero_attn_shape, dtype=value.dtype, device=value.device)
            ], dim=1)
            
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([
                    key_padding_mask,
                    torch.zeros(key_padding_mask.shape[0], 1,
                              dtype=key_padding_mask.dtype,
                              device=key_padding_mask.device)
                ], dim=1)
        
        # Apply relative position encoding if enabled
        if self.relative_attention:
            attn_mask = self._add_relative_position_bias(
                query, key, attn_mask
            )
        
        # Forward through quantum attention
        output, attn_weights = self.quantum_attention(
            query, key, value, attn_mask, key_padding_mask,
            need_weights, average_attn_weights
        )
        
        return output, attn_weights
    
    def _add_relative_position_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Add relative position bias to attention mask."""
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        
        # Generate relative position matrix
        positions_q = torch.arange(seq_len_q, device=query.device)
        positions_k = torch.arange(seq_len_k, device=key.device)
        
        relative_positions = positions_q.unsqueeze(1) - positions_k.unsqueeze(0)
        
        # Clip to maximum range
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_positions // 2,
            self.max_relative_positions // 2
        )
        
        # Convert to positive indices
        relative_positions = relative_positions + self.max_relative_positions // 2
        
        # Create one-hot encoding for position embeddings
        relative_positions_one_hot = F.one_hot(
            relative_positions, self.max_relative_positions
        ).float()
        
        # Get relative position embeddings
        relative_bias = self.relative_position_k(relative_positions_one_hot)
        relative_bias = relative_bias.real  # Use real part for bias
        
        # Reshape for attention
        relative_bias = relative_bias.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        
        # Add to attention mask
        if attn_mask is None:
            attn_mask = relative_bias
        else:
            attn_mask = attn_mask + relative_bias
        
        return attn_mask


class CrossModalQuantumAttention(nn.Module):
    """
    Quantum-inspired cross-modal attention for multi-modal inputs.
    Handles attention between different modalities (e.g., audio and text).
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        modality_fusion: str = "additive"
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.modality_fusion = modality_fusion
        
        # Modality-specific projections
        self.query_proj = ComplexLinear(query_dim, embed_dim)
        self.key_proj = ComplexLinear(key_dim, embed_dim)
        self.value_proj = ComplexLinear(key_dim, embed_dim)
        
        # Core attention mechanism
        self.attention = QuantumInspiredAttention(
            embed_dim, num_heads, dropout
        )
        
        # Modality fusion parameters
        if modality_fusion == "gated":
            self.fusion_gate = ComplexLinear(embed_dim * 2, embed_dim)
        elif modality_fusion == "learned":
            self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
    
    def forward(
        self,
        query_modality: torch.Tensor,
        key_modality: torch.Tensor,
        value_modality: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Cross-modal attention forward pass."""
        if value_modality is None:
            value_modality = key_modality
        
        # Project modalities to common embedding space
        query_proj = self.query_proj(query_modality)
        key_proj = self.key_proj(key_modality)
        value_proj = self.value_proj(value_modality)
        
        # Apply quantum attention
        attended_output, attn_weights = self.attention(
            query_proj, key_proj, value_proj, **kwargs
        )
        
        # Apply modality fusion
        if self.modality_fusion == "additive":
            output = query_proj + attended_output
        elif self.modality_fusion == "gated":
            # Gated fusion
            gate_input = torch.cat([query_proj, attended_output], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(gate_input).real)
            output = gate * query_proj + (1 - gate) * attended_output
        elif self.modality_fusion == "learned":
            # Learned weighted combination
            weights = F.softmax(self.fusion_weights, dim=0)
            output = weights[0] * query_proj + weights[1] * attended_output
        else:
            output = attended_output
        
        return output, attn_weights


class QuantumAttentionBlock(nn.Module):
    """
    Complete quantum attention block with residual connections and normalization.
    Integrates quantum attention with the transformer architecture.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "complex_gelu",
        prenorm: bool = True,
        attention_type: str = "quantum"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.prenorm = prenorm
        
        # Attention layer
        if attention_type == "quantum":
            self.self_attn = MultiHeadQuantumAttention(
                embed_dim, num_heads, dropout
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Feed-forward network
        from models.complex_layers import ComplexFeedForward
        self.ffn = ComplexFeedForward(
            embed_dim, ff_dim, activation, dropout
        )
        
        # Layer normalization
        from models.complex_layers import ComplexLayerNorm
        self.norm1 = ComplexLayerNorm(embed_dim)
        self.norm2 = ComplexLayerNorm(embed_dim)
        
        # Dropout
        self.dropout = dropout
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through quantum attention block."""
        # Self-attention with residual connection
        if self.prenorm:
            # Pre-norm
            normed = self.norm1(x)
            attn_out, _ = self.self_attn(
                normed, normed, normed,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
            if self.training:
                attn_out = complex_dropout(attn_out, p=self.dropout, training=True)
            x = x + attn_out
            
            # Feed-forward with residual connection
            normed = self.norm2(x)
            ff_out = self.ffn(normed)
            if self.training:
                ff_out = complex_dropout(ff_out, p=self.dropout, training=True)
            x = x + ff_out
        else:
            # Post-norm
            attn_out, _ = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
            if self.training:
                attn_out = complex_dropout(attn_out, p=self.dropout, training=True)
            x = self.norm1(x + attn_out)
            
            ff_out = self.ffn(x)
            if self.training:
                ff_out = complex_dropout(ff_out, p=self.dropout, training=True)
            x = self.norm2(x + ff_out)
        
        return x


# Utility functions for quantum attention

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal (lower triangular) mask for autoregressive attention."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))


def create_padding_mask(
    lengths: torch.Tensor, 
    max_len: Optional[int] = None
) -> torch.Tensor:
    """Create padding mask from sequence lengths."""
    if max_len is None:
        max_len = lengths.max().item()
    
    batch_size = lengths.shape[0]
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) >= lengths.unsqueeze(1)
    
    return mask


def visualize_attention_patterns(
    attention_weights: torch.Tensor,
    head_idx: int = 0,
    layer_name: str = "attention"
) -> Dict[str, Any]:
    """
    Analyze attention patterns for visualization and debugging.
    
    Args:
        attention_weights: Attention weight tensor (batch, heads, seq_len, seq_len)
        head_idx: Which attention head to analyze
        layer_name: Name of the layer for identification
        
    Returns:
        Dictionary with attention analysis results
    """
    # Extract specific head
    if attention_weights.dim() == 4:
        head_attn = attention_weights[0, head_idx].cpu().numpy()  # (seq_len, seq_len)
    else:
        head_attn = attention_weights.cpu().numpy()
    
    # Compute attention statistics
    attention_stats = {
        'layer_name': layer_name,
        'head_idx': head_idx,
        'attention_entropy': compute_attention_entropy(head_attn),
        'attention_sparsity': compute_attention_sparsity(head_attn),
        'diagonal_attention': np.mean(np.diag(head_attn)),
        'max_attention': np.max(head_attn),
        'min_attention': np.min(head_attn),
        'attention_variance': np.var(head_attn),
        'attention_matrix': head_attn
    }
    
    return attention_stats


def compute_attention_entropy(attention_matrix: np.ndarray) -> float:
    """Compute entropy of attention distribution."""
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    attention_safe = attention_matrix + eps
    
    # Compute entropy for each query position
    entropies = -np.sum(attention_safe * np.log(attention_safe), axis=1)
    
    # Return average entropy
    return np.mean(entropies)


def compute_attention_sparsity(attention_matrix: np.ndarray, threshold: float = 0.01) -> float:
    """Compute sparsity of attention (fraction of weights below threshold)."""
    total_weights = attention_matrix.size
    sparse_weights = np.sum(attention_matrix < threshold)
    
    return sparse_weights / total_weights


def quantum_interference_score(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    interference_type: str = "constructive"
) -> torch.Tensor:
    """
    Compute quantum interference scores between query and key states.
    
    Args:
        query_states: Complex query states
        key_states: Complex key states  
        interference_type: "constructive", "destructive", or "both"
        
    Returns:
        Interference scores
    """
    # Compute complex inner products
    inner_products = torch.sum(
        torch.conj(query_states.unsqueeze(-2)) * key_states.unsqueeze(-3),
        dim=-1
    )
    
    # Extract magnitude and phase
    magnitude = torch.abs(inner_products)
    phase = torch.angle(inner_products)
    
    if interference_type == "constructive":
        # Constructive interference: cos(phase) > 0
        interference = magnitude * torch.clamp(torch.cos(phase), min=0)
    elif interference_type == "destructive":
        # Destructive interference: cos(phase) < 0
        interference = magnitude * torch.clamp(-torch.cos(phase), min=0)
    elif interference_type == "both":
        # Both types: full interference pattern
        interference = magnitude * torch.cos(phase)
    else:
        raise ValueError(f"Unknown interference type: {interference_type}")
    
    return interference


class AdaptiveQuantumAttention(nn.Module):
    """
    Adaptive quantum attention that learns to balance different attention mechanisms.
    Can dynamically choose between Hermitian, Born rule, and superposition attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        num_attention_types: int = 3,
        temperature: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_attention_types = num_attention_types
        self.temperature = temperature
        
        # Individual attention mechanisms
        self.attention_mechanisms = nn.ModuleList([
            QuantumInspiredAttention(
                embed_dim, num_heads, dropout, 
                attention_method=method
            ) for method in ["hermitian_interference", "born_rule", "quantum_superposition"]
        ])
        
        # Attention type selection network
        self.attention_selector = ComplexLinear(embed_dim, num_attention_types)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with adaptive attention mechanism selection."""
        batch_size, seq_len, embed_dim = query.shape
        
        # Compute attention type weights based on query
        query_pooled = torch.mean(query, dim=1)  # Pool over sequence
        attention_logits = self.attention_selector(query_pooled).real  # Use real part
        attention_weights = F.softmax(attention_logits / self.temperature, dim=-1)
        
        # Compute outputs from all attention mechanisms
        outputs = []
        all_attn_weights = []
        
        for i, attention_module in enumerate(self.attention_mechanisms):
            output, attn_weights = attention_module(query, key, value, **kwargs)
            outputs.append(output)
            all_attn_weights.append(attn_weights)
        
        # Weighted combination of outputs
        final_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            weight = attention_weights[:, i].unsqueeze(1).unsqueeze(2)  # Broadcast to sequence
            final_output += weight * output
        
        # Average attention weights (for visualization)
        if all_attn_weights[0] is not None:
            final_attn_weights = torch.zeros_like(all_attn_weights[0])
            for i, attn_weights in enumerate(all_attn_weights):
                weight = attention_weights[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                final_attn_weights += weight * attn_weights
        else:
            final_attn_weights = None
        
        return final_output, final_attn_weights


class QuantumAttentionWithMemory(nn.Module):
    """
    Quantum attention with external memory mechanism.
    Maintains quantum states in external memory for long-term dependencies.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        memory_size: int,
        dropout: float = 0.1,
        memory_update_rate: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.memory_update_rate = memory_update_rate
        
        # Core attention mechanism
        self.attention = QuantumInspiredAttention(embed_dim, num_heads, dropout)
        
        # External memory (complex-valued)
        self.register_buffer(
            'memory_states',
            torch.zeros(memory_size, embed_dim, dtype=torch.complex64)
        )
        
        # Memory interaction layers
        self.memory_query_proj = ComplexLinear(embed_dim, embed_dim)
        self.memory_update_proj = ComplexLinear(embed_dim, embed_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        update_memory: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with memory interaction."""
        batch_size, seq_len, embed_dim = query.shape
        
        # Standard attention
        attended_output, attn_weights = self.attention(query, key, value, **kwargs)
        
        # Memory interaction
        memory_query = self.memory_query_proj(query)
        
        # Attention to memory
        memory_expanded = self.memory_states.unsqueeze(0).expand(batch_size, -1, -1)
        memory_output, memory_attn = self.attention(
            memory_query, memory_expanded, memory_expanded, **kwargs
        )
        
        # Combine with standard attention
        combined_output = attended_output + 0.1 * memory_output  # Small memory contribution
        
        # Update memory (during training)
        if update_memory and self.training:
            self._update_memory(query.detach())
        
        return combined_output, attn_weights
    
    def _update_memory(self, query_states: torch.Tensor):
        """Update external memory with current query states."""
        # Average query states over batch and sequence
        query_summary = torch.mean(query_states, dim=(0, 1))  # (embed_dim,)
        
        # Update memory with exponential moving average
        memory_update = self.memory_update_proj(query_summary.unsqueeze(0)).squeeze(0)
        
        # Select random memory slot to update
        update_idx = torch.randint(0, self.memory_size, (1,)).item()
        
        self.memory_states[update_idx] = (
            (1 - self.memory_update_rate) * self.memory_states[update_idx] + 
            self.memory_update_rate * memory_update
        )


class HierarchicalQuantumAttention(nn.Module):
    """
    Hierarchical quantum attention that operates at multiple scales.
    Implements coarse-to-fine attention refinement.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_scales: int = 3,
        dropout: float = 0.1,
        scale_factors: Optional[List[int]] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        
        if scale_factors is None:
            scale_factors = [1, 2, 4]  # Different pooling factors
        self.scale_factors = scale_factors[:num_scales]
        
        # Multi-scale attention modules
        self.scale_attentions = nn.ModuleList([
            QuantumInspiredAttention(embed_dim, num_heads, dropout)
            for _ in range(num_scales)
        ])
        
        # Scale fusion network
        self.scale_fusion = ComplexLinear(embed_dim * num_scales, embed_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Multi-scale attention forward pass."""
        batch_size, seq_len, embed_dim = query.shape
        
        scale_outputs = []
        scale_attentions = []
        
        for i, (attention_module, scale_factor) in enumerate(
            zip(self.scale_attentions, self.scale_factors)
        ):
            if scale_factor == 1:
                # Full resolution
                scale_query, scale_key, scale_value = query, key, value
            else:
                # Downsample for coarser scales
                scale_query = self._downsample(query, scale_factor)
                scale_key = self._downsample(key, scale_factor)
                scale_value = self._downsample(value, scale_factor)
            
            # Apply attention at this scale
            scale_output, scale_attn = attention_module(
                scale_query, scale_key, scale_value, **kwargs
            )
            
            # Upsample back to original resolution if needed
            if scale_factor > 1:
                scale_output = self._upsample(scale_output, seq_len)
            
            scale_outputs.append(scale_output)
            scale_attentions.append(scale_attn)
        
        # Fuse multi-scale outputs
        concatenated = torch.cat(scale_outputs, dim=-1)
        fused_output = self.scale_fusion(concatenated)
        
        # Average attention weights for return
        avg_attention = None
        if scale_attentions[0] is not None:
            avg_attention = torch.mean(torch.stack([
                attn for attn in scale_attentions if attn is not None
            ]), dim=0)
        
        return fused_output, avg_attention
    
    def _downsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Downsample sequence by averaging consecutive elements."""
        batch_size, seq_len, embed_dim = x.shape
        
        # Pad sequence if necessary
        pad_len = (factor - seq_len % factor) % factor
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len += pad_len
        
        # Reshape and average
        x_reshaped = x.view(batch_size, seq_len // factor, factor, embed_dim)
        downsampled = torch.mean(x_reshaped, dim=2)
        
        return downsampled
    
    def _upsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Upsample sequence to target length using interpolation."""
        # Use linear interpolation in the time dimension
        x_permuted = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        upsampled = F.interpolate(
            x_permuted, size=target_len, mode='linear', align_corners=False
        )
        return upsampled.permute(0, 2, 1)  # Back to (batch, seq_len, embed_dim)


# Additional utility functions

def compute_quantum_entanglement(
    attention_weights: torch.Tensor,
    threshold: float = 0.1
) -> torch.Tensor:
    """
    Compute a measure of quantum entanglement in attention patterns.
    High entanglement indicates strong non-local correlations.
    """
    # Normalize attention weights
    normalized_attn = attention_weights / (torch.sum(attention_weights, dim=-1, keepdim=True) + 1e-8)
    
    # Compute mutual information as entanglement measure
    # H(X) - H(X|Y) where X,Y are query and key positions
    
    # Marginal entropy (query positions)
    query_marginal = torch.sum(normalized_attn, dim=-1)  # Sum over keys
    query_entropy = -torch.sum(
        query_marginal * torch.log(query_marginal + 1e-8), dim=-1
    )
    
    # Conditional entropy H(X|Y)
    conditional_entropy = -torch.sum(
        normalized_attn * torch.log(normalized_attn + 1e-8), dim=(-2, -1)
    )
    
    # Mutual information (entanglement measure)
    entanglement = query_entropy - conditional_entropy
    
    return entanglement


def analyze_attention_locality(
    attention_weights: torch.Tensor,
    window_sizes: List[int] = [3, 5, 7, 11]
) -> Dict[str, float]:
    """
    Analyze locality patterns in attention weights.
    Returns fraction of attention mass within different window sizes.
    """
    seq_len = attention_weights.shape[-1]
    locality_stats = {}
    
    for window_size in window_sizes:
        local_mass = 0.0
        total_positions = 0
        
        for i in range(seq_len):
            # Define local window around position i
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            
            # Sum attention within window
            local_attn = torch.sum(attention_weights[..., i, start:end], dim=-1)
            local_mass += torch.mean(local_attn).item()
            total_positions += 1
        
        locality_stats[f'window_{window_size}'] = local_mass / total_positions
    
    return locality_stats


def quantum_attention_regularization(
    attention_weights: torch.Tensor,
    target_sparsity: float = 0.1,
    target_entropy: float = 2.0
) -> torch.Tensor:
    """
    Compute regularization loss for quantum attention.
    Encourages sparsity and controlled entropy.
    """
    # Sparsity regularization (L1 penalty)
    sparsity_loss = torch.mean(torch.abs(attention_weights))
    
    # Entropy regularization
    eps = 1e-8
    normalized_attn = attention_weights / (torch.sum(attention_weights, dim=-1, keepdim=True) + eps)
    entropy = -torch.sum(normalized_attn * torch.log(normalized_attn + eps), dim=-1)
    entropy_loss = torch.mean((entropy - target_entropy) ** 2)
    
    # Combine losses
    total_loss = sparsity_loss + 0.1 * entropy_loss
    
    return total_loss