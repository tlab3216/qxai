"""
Complex-Valued Transformer for Q-XAI framework.
Main transformer architecture implementing the quantum-inspired transformer described in the paper.
Integrates complex layers, quantum attention, and provides the backbone for QISA, AUQ, and QICP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Union, List, Dict, Any, NamedTuple

from models.complex_layers import (
    ComplexEmbedding, ComplexLayerNorm, ComplexFeedForward, 
    ComplexPositionalEncoding, ComplexResidualConnection,
    ComplexAttentionPooling, ComplexDropPath
)
from models.quantum_attention import (
    QuantumAttentionBlock, MultiHeadQuantumAttention,
    AdaptiveQuantumAttention, HierarchicalQuantumAttention
)
from utils.complex_math import (
    ComplexLinear, complex_dropout, amplitude_to_probability
)
from config.model_config import ComplexTransformerConfig


class TransformerOutput(NamedTuple):
    """Output structure for complex transformer."""
    logits: torch.Tensor  # Final classification logits
    complex_amplitudes: torch.Tensor  # Complex amplitudes for Born rule
    hidden_states: Optional[List[torch.Tensor]] = None  # Hidden states from all layers
    attention_weights: Optional[List[torch.Tensor]] = None  # Attention weights from all layers
    embeddings: Optional[torch.Tensor] = None  # Input embeddings after positional encoding


class ComplexTransformerEncoder(nn.Module):
    """
    Complex-valued transformer encoder implementing the core architecture from the paper.
    Processes complex spectrograms through quantum-inspired attention layers.
    """
    
    def __init__(
        self,
        config: ComplexTransformerConfig,
        return_attention_weights: bool = False,
        return_hidden_states: bool = False
    ):
        super().__init__()
        self.config = config
        self.return_attention_weights = return_attention_weights
        self.return_hidden_states = return_hidden_states
        
        # Input embedding layer (Equation 2 in paper)
        self.embedding = ComplexEmbedding(
            input_dim=config.input_dim,
            embed_dim=config.embed_dim,
            init_method="quantum_inspired"
        )
        
        # Positional encoding
        self.pos_encoding = ComplexPositionalEncoding(
            embed_dim=config.embed_dim,
            max_length=config.max_seq_length,
            encoding_type=config.pos_encoding_type,
            dropout=config.pos_encoding_dropout
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            QuantumAttentionBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                activation=config.activation,
                prenorm=True,
                attention_type="quantum"
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = ComplexLayerNorm(config.embed_dim)
        
        # Dropout paths for stochastic depth
        if config.path_dropout > 0:
            self.drop_paths = nn.ModuleList([
                ComplexDropPath(config.path_dropout * (i + 1) / config.num_layers)
                for i in range(config.num_layers)
            ])
        else:
            self.drop_paths = [None] * config.num_layers
    
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> TransformerOutput:
        """
        Forward pass through complex transformer encoder.
        
        Args:
            x: Complex spectrogram input (batch_size, seq_len, input_dim)
            src_key_padding_mask: Padding mask for input sequences
            attn_mask: Attention mask for self-attention
            
        Returns:
            TransformerOutput with logits, amplitudes, and optional intermediate states
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Store intermediate states if requested
        hidden_states = [] if self.return_hidden_states else None
        attention_weights = [] if self.return_attention_weights else None
        
        # Input embedding (Equation 2)
        embedded = self.embedding(x)  # Convert to complex embeddings
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        if self.return_hidden_states:
            hidden_states.append(embedded)
        
        # Pass through transformer layers
        hidden = embedded
        
        for i, (layer, drop_path) in enumerate(zip(self.layers, self.drop_paths)):
            # Apply transformer layer
            layer_output = layer(
                hidden,
                attn_mask=attn_mask,
                key_padding_mask=src_key_padding_mask
            )
            
            # Apply drop path if configured
            if drop_path is not None:
                layer_output = drop_path(layer_output)
            
            hidden = layer_output
            
            # Store intermediate states
            if self.return_hidden_states:
                hidden_states.append(hidden)
            
            # Store attention weights (would need modification to layer to return them)
            if self.return_attention_weights:
                # For now, we'll store None - this would require modifying QuantumAttentionBlock
                attention_weights.append(None)
        
        # Final normalization
        hidden = self.final_norm(hidden)
        
        return TransformerOutput(
            logits=hidden,  # Will be processed by classification head
            complex_amplitudes=hidden,  # Complex amplitudes for Born rule
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            embeddings=embedded
        )


class ComplexTransformerClassifier(nn.Module):
    """
    Complete Q-XAI transformer for acoustic scene classification.
    Implements the full architecture described in the paper.
    """
    
    def __init__(
        self,
        config: ComplexTransformerConfig,
        num_classes: Optional[int] = None,
        pooling_method: str = "attention",
        return_intermediate: bool = False
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes or config.num_classes
        self.pooling_method = pooling_method
        self.return_intermediate = return_intermediate
        
        # Core transformer encoder
        self.encoder = ComplexTransformerEncoder(
            config=config,
            return_attention_weights=return_intermediate,
            return_hidden_states=return_intermediate
        )
        
        # Pooling layer for sequence-to-vector conversion
        if pooling_method == "attention":
            self.pooling = ComplexAttentionPooling(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads // 2,  # Fewer heads for pooling
                temperature=1.0
            )
        elif pooling_method == "mean":
            self.pooling = None  # Will use mean pooling
        elif pooling_method == "cls_token":
            # Add a special CLS token for classification
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, config.embed_dim, dtype=torch.complex64)
            )
            self.pooling = None
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
        
        # Classification head
        self.classifier = ComplexLinear(
            in_features=config.embed_dim,
            out_features=self.num_classes,
            bias=True,
            init_method="complex_normal"
        )
        
        # Dropout for classification
        self.classifier_dropout = config.dropout
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with quantum-inspired method."""
        if isinstance(module, ComplexLinear):
            # Complex Glorot initialization
            fan_in = module.in_features
            fan_out = module.out_features
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            
            with torch.no_grad():
                module.weight_real.uniform_(-bound, bound)
                module.weight_imag.uniform_(-bound, bound)
                
                if module.bias_real is not None:
                    module.bias_real.uniform_(-bound, bound)
                    module.bias_imag.uniform_(-bound, bound)
        
        elif isinstance(module, ComplexEmbedding):
            # Already handled in ComplexEmbedding initialization
            pass
    
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_complex_amplitudes: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through complete Q-XAI transformer.
        
        Args:
            x: Input complex spectrogram (batch_size, seq_len, input_dim)
            src_key_padding_mask: Padding mask
            attn_mask: Attention mask
            return_complex_amplitudes: Whether to return complex amplitudes for QICP
            
        Returns:
            If return_complex_amplitudes=True: (logits, complex_amplitudes, intermediate_outputs)
            Else: logits only
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Handle CLS token if used
        if self.pooling_method == "cls_token":
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Update padding mask for CLS token
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        
        # Pass through transformer encoder
        encoder_output = self.encoder(
            x=x,
            src_key_padding_mask=src_key_padding_mask,
            attn_mask=attn_mask
        )
        
        # Extract sequence representations
        sequence_output = encoder_output.logits  # (batch_size, seq_len, embed_dim)
        
        # Apply pooling to get fixed-size representation
        if self.pooling_method == "attention":
            # Use attention pooling
            pooled_output = self.pooling(sequence_output, mask=src_key_padding_mask)
        elif self.pooling_method == "mean":
            # Mean pooling with masking
            if src_key_padding_mask is not None:
                # Mask out padded positions
                mask_expanded = (~src_key_padding_mask).unsqueeze(-1).float()
                masked_output = sequence_output * mask_expanded
                pooled_output = torch.sum(masked_output, dim=1) / torch.sum(mask_expanded, dim=1)
            else:
                pooled_output = torch.mean(sequence_output, dim=1)
        elif self.pooling_method == "cls_token":
            # Use CLS token representation
            pooled_output = sequence_output[:, 0, :]  # First token is CLS
        
        # Apply dropout before classification
        if self.training:
            pooled_output = complex_dropout(
                pooled_output, p=self.classifier_dropout, training=True, coherent=True
            )
        
        # Classification head - outputs complex amplitudes
        complex_amplitudes = self.classifier(pooled_output)  # (batch_size, num_classes)
        
        # Convert complex amplitudes to logits using Born rule (|amplitude|²)
        logits = amplitude_to_probability(complex_amplitudes, dim=-1)
        
        # Return based on requirements
        if return_complex_amplitudes or self.return_intermediate:
            intermediate_outputs = {
                'sequence_output': sequence_output,
                'pooled_output': pooled_output,
                'hidden_states': encoder_output.hidden_states,
                'attention_weights': encoder_output.attention_weights,
                'embeddings': encoder_output.embeddings
            }
            return logits, complex_amplitudes, intermediate_outputs
        else:
            return logits
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """Extract attention weights for interpretability analysis."""
        # Temporarily enable attention weight return
        original_return_attention = self.encoder.return_attention_weights
        self.encoder.return_attention_weights = True
        
        with torch.no_grad():
            _, _, intermediate = self.forward(x, return_complex_amplitudes=True, **kwargs)
            attention_weights = intermediate.get('attention_weights', [])
        
        # Restore original setting
        self.encoder.return_attention_weights = original_return_attention
        
        # Filter by layer and head if specified
        if layer_idx is not None:
            attention_weights = [attention_weights[layer_idx]]
        
        if head_idx is not None:
            attention_weights = [
                attn[:, head_idx:head_idx+1] if attn is not None else None
                for attn in attention_weights
            ]
        
        return attention_weights
    
    def get_complex_representations(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
        **kwargs
    ) -> torch.Tensor:
        """Extract complex representations from specified layer."""
        original_return_hidden = self.encoder.return_hidden_states
        self.encoder.return_hidden_states = True
        
        with torch.no_grad():
            _, _, intermediate = self.forward(x, return_complex_amplitudes=True, **kwargs)
            hidden_states = intermediate.get('hidden_states', [])
        
        self.encoder.return_hidden_states = original_return_hidden
        
        if hidden_states and len(hidden_states) > abs(layer_idx):
            return hidden_states[layer_idx]
        else:
            return None


class MultiScaleComplexTransformer(ComplexTransformerClassifier):
    """
    Multi-scale complex transformer for processing spectrograms at different resolutions.
    Useful for capturing both fine-grained and coarse-grained acoustic patterns.
    """
    
    def __init__(
        self,
        config: ComplexTransformerConfig,
        num_classes: Optional[int] = None,
        scale_factors: List[int] = [1, 2, 4],
        fusion_method: str = "learned_weights"
    ):
        super().__init__(config, num_classes, return_intermediate=True)
        
        self.scale_factors = scale_factors
        self.fusion_method = fusion_method
        
        # Create separate encoders for different scales
        self.scale_encoders = nn.ModuleList([
            ComplexTransformerEncoder(config) for _ in scale_factors
        ])
        
        # Scale-specific pooling
        self.scale_poolings = nn.ModuleList([
            ComplexAttentionPooling(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads // 2
            ) for _ in scale_factors
        ])
        
        # Fusion mechanism
        if fusion_method == "learned_weights":
            self.fusion_weights = nn.Parameter(torch.ones(len(scale_factors)))
        elif fusion_method == "attention":
            self.fusion_attention = ComplexLinear(
                config.embed_dim, len(scale_factors)
            )
        
        # Final classifier
        self.multi_scale_classifier = ComplexLinear(
            config.embed_dim, num_classes or config.num_classes
        )
    
    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """Multi-scale forward pass."""
        batch_size, seq_len, input_dim = x.shape
        
        scale_outputs = []
        scale_amplitudes = []
        
        # Process at each scale
        for scale_factor, encoder, pooling in zip(
            self.scale_factors, self.scale_encoders, self.scale_poolings
        ):
            # Downsample input if necessary
            if scale_factor == 1:
                scale_input = x
            else:
                # Simple downsampling by averaging
                pad_len = (scale_factor - seq_len % scale_factor) % scale_factor
                if pad_len > 0:
                    scale_input = F.pad(x, (0, 0, 0, pad_len))
                else:
                    scale_input = x
                
                scale_input = scale_input.view(
                    batch_size, -1, scale_factor, input_dim
                ).mean(dim=2)
            
            # Encode at this scale
            encoder_output = encoder(scale_input, **kwargs)
            
            # Pool to fixed size
            pooled = pooling(encoder_output.logits)
            
            scale_outputs.append(pooled)
            scale_amplitudes.append(pooled)
        
        # Fuse multi-scale representations
        if self.fusion_method == "learned_weights":
            # Weighted combination
            weights = F.softmax(self.fusion_weights, dim=0)
            fused_output = sum(w * output for w, output in zip(weights, scale_outputs))
        
        elif self.fusion_method == "attention":
            # Attention-based fusion
            stacked_outputs = torch.stack(scale_outputs, dim=1)  # (batch, scales, embed_dim)
            
            # Compute attention weights
            attn_logits = self.fusion_attention(stacked_outputs).real  # Use real part
            attn_weights = F.softmax(attn_logits, dim=1)  # (batch, scales, scales)
            
            # Apply attention
            fused_output = torch.sum(
                attn_weights.unsqueeze(-1) * stacked_outputs, dim=1
            )
        
        elif self.fusion_method == "concatenation":
            # Simple concatenation (would need different classifier)
            fused_output = torch.cat(scale_outputs, dim=-1)
        
        else:
            # Simple averaging
            fused_output = torch.mean(torch.stack(scale_outputs), dim=0)
        
        # Final classification
        complex_amplitudes = self.multi_scale_classifier(fused_output)
        logits = amplitude_to_probability(complex_amplitudes, dim=-1)
        
        return logits, complex_amplitudes, {
            'scale_outputs': scale_outputs,
            'fused_output': fused_output
        }


class ComplexTransformerWithAdaptiveAttention(ComplexTransformerClassifier):
    """
    Complex transformer with adaptive attention mechanism selection.
    Can dynamically choose between different quantum attention types.
    """
    
    def __init__(
        self,
        config: ComplexTransformerConfig,
        num_classes: Optional[int] = None,
        adaptation_layers: List[int] = None
    ):
        # Modify config to use adaptive attention
        super().__init__(config, num_classes)
        
        if adaptation_layers is None:
            adaptation_layers = list(range(config.num_layers))
        
        # Replace specified layers with adaptive attention
        for layer_idx in adaptation_layers:
            if layer_idx < len(self.encoder.layers):
                # Replace the attention in the quantum attention block
                original_block = self.encoder.layers[layer_idx]
                
                # Create new adaptive attention
                adaptive_attention = AdaptiveQuantumAttention(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout
                )
                
                # Replace the attention module (this would require modifying QuantumAttentionBlock)
                # For now, we'll replace the entire block
                self.encoder.layers[layer_idx] = QuantumAttentionBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                    activation=config.activation,
                    attention_type="adaptive"  # Would need to implement this
                )


class ComplexTransformerEnsemble(nn.Module):
    """
    Ensemble of complex transformers for improved uncertainty quantification.
    Useful for the AUQ component of Q-XAI.
    """
    
    def __init__(
        self,
        config: ComplexTransformerConfig,
        num_models: int = 5,
        diversity_method: str = "dropout",
        num_classes: Optional[int] = None
    ):
        super().__init__()
        self.num_models = num_models
        self.diversity_method = diversity_method
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            ComplexTransformerClassifier(config, num_classes)
            for _ in range(num_models)
        ])
        
        # Add diversity if requested
        if diversity_method == "different_init":
            # Models already have different random initialization
            pass
        elif diversity_method == "different_dropout":
            # Set different dropout rates
            for i, model in enumerate(self.models):
                dropout_rate = config.dropout * (0.5 + 0.5 * i / num_models)
                self._set_dropout_rate(model, dropout_rate)
    
    def _set_dropout_rate(self, model: nn.Module, dropout_rate: float):
        """Recursively set dropout rate in model."""
        for module in model.modules():
            if hasattr(module, 'dropout') and isinstance(module.dropout, float):
                module.dropout = dropout_rate
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_predictions: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Ensemble forward pass."""
        predictions = []
        complex_amplitudes_list = []
        
        for model in self.models:
            logits, complex_amplitudes, _ = model(
                x, return_complex_amplitudes=True, **kwargs
            )
            predictions.append(logits)
            complex_amplitudes_list.append(complex_amplitudes)
        
        # Average predictions
        averaged_logits = torch.mean(torch.stack(predictions), dim=0)
        averaged_amplitudes = torch.mean(torch.stack(complex_amplitudes_list), dim=0)
        
        if return_all_predictions:
            return averaged_logits, averaged_amplitudes, predictions, complex_amplitudes_list
        else:
            return averaged_logits, averaged_amplitudes


def create_complex_transformer(
    config: ComplexTransformerConfig,
    model_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function to create different variants of complex transformers.
    
    Args:
        config: Model configuration
        model_type: Type of model to create
        **kwargs: Additional arguments for specific model types
        
    Returns:
        Complex transformer model
    """
    if model_type == "standard":
        return ComplexTransformerClassifier(config, **kwargs)
    
    elif model_type == "multi_scale":
        return MultiScaleComplexTransformer(config, **kwargs)
    
    elif model_type == "adaptive":
        return ComplexTransformerWithAdaptiveAttention(config, **kwargs)
    
    elif model_type == "ensemble":
        return ComplexTransformerEnsemble(config, **kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in complex transformer model."""
    total_params = 0
    trainable_params = 0
    complex_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        
        if torch.is_complex(param):
            complex_params += param_count
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'complex_parameters': complex_params,
        'real_parameters': total_params - complex_params
    }


def get_model_complexity(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """Analyze model computational complexity."""
    def count_ops(module, input_tensor, output_tensor):
        """Count operations for a module."""
        if isinstance(module, ComplexLinear):
            # Complex linear: 4 real operations per complex multiplication
            return 4 * module.in_features * module.out_features * input_tensor[0].shape[0]
        elif isinstance(module, MultiHeadQuantumAttention):
            # Approximate attention complexity
            seq_len = input_tensor[0].shape[1]
            embed_dim = input_tensor[0].shape[2]
            return 4 * seq_len * seq_len * embed_dim  # Simplified
        return 0
    
    # This is a simplified complexity analysis
    # In practice, you'd want to use tools like ptflops or torchinfo
    
    param_stats = count_parameters(model)
    
    # Estimate FLOPs (simplified)
    total_flops = 0
    for module in model.modules():
        if isinstance(module, (ComplexLinear, MultiHeadQuantumAttention)):
            # This is a very rough estimate
            total_flops += 1000000  # Placeholder
    
    return {
        'parameters': param_stats,
        'estimated_flops': total_flops,
        'model_size_mb': param_stats['total_parameters'] * 4 / (1024 * 1024)  # Assuming float32
    }