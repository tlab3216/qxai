"""
Quantum-Inspired State Attribution (QISA) for Q-XAI framework.
Implements Wirtinger calculus-based attribution for complex-valued neural networks.
Provides faithful explanations for non-holomorphic functions as described in the paper.
"""

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings

from utils.complex_math import WirtingerGradient
from config.model_config import QISAConfig
from models.complex_transformer import ComplexTransformerClassifier


class QISAAttributor:
    """
    Main QISA attribution class implementing Wirtinger calculus for complex-valued models.
    
    Computes faithful attribution maps using Equations 6-8 from the paper:
    - ∂L/∂z = 1/2 * (∂L/∂x - i∂L/∂y)  (Equation 6)
    - ∂L/∂z* = 1/2 * (∂L/∂x + i∂L/∂y)  (Equation 7)
    - Attribution = ||∂L/∂z||² + ||∂L/∂z*||²  (Equation 8)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QISAConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or next(model.parameters()).device
        
        # Ensure model is in evaluation mode for attribution
        self.model.eval()
        
        # Cache for intermediate computations
        self._attribution_cache = {}
        self._gradient_cache = {}
        
    def attribute(
        self,
        inputs: torch.Tensor,
        target: Union[int, torch.Tensor],
        layer_name: Optional[str] = None,
        abs_attribution: bool = True,
        normalize: bool = True,
        smooth: bool = None
    ) -> torch.Tensor:
        """
        Compute QISA attribution for given inputs and target.
        
        Args:
            inputs: Complex input tensor (batch_size, seq_len, features)
            target: Target class index or tensor
            layer_name: Specific layer to compute attribution for (default: input layer)
            abs_attribution: Whether to return absolute attribution values
            normalize: Whether to normalize attribution values
            smooth: Whether to apply smoothing (uses config default if None)
            
        Returns:
            Attribution tensor of same shape as inputs
        """
        if not torch.is_complex(inputs):
            raise ValueError("QISA requires complex-valued inputs")
        
        inputs = inputs.to(self.device)
        inputs.requires_grad_(True)
        
        # Handle target
        if isinstance(target, int):
            target = torch.tensor([target], device=self.device)
        elif isinstance(target, torch.Tensor):
            target = target.to(self.device)
        
        # Forward pass
        outputs = self._forward_with_hooks(inputs, layer_name)
        
        # Compute loss for target class
        if outputs.dim() == 3:  # (batch, seq, classes)
            # Pool outputs for classification
            pooled_outputs = torch.mean(outputs, dim=1)
        else:
            pooled_outputs = outputs
            
        # Use cross-entropy loss for attribution
        loss = F.cross_entropy(pooled_outputs, target)
        
        # Compute Wirtinger gradients
        attribution = self._compute_wirtinger_attribution(
            loss, inputs, layer_name
        )
        
        # Post-processing
        if abs_attribution:
            attribution = torch.abs(attribution)
            
        if normalize or self.config.normalize_attributions:
            attribution = self._normalize_attribution(attribution)
            
        if smooth is None:
            smooth = self.config.smooth_attributions
        if smooth:
            attribution = self._smooth_attribution(attribution)
        
        return attribution.detach()
    
    def _forward_with_hooks(
        self,
        inputs: torch.Tensor,
        layer_name: Optional[str] = None
    ) -> torch.Tensor:
        """Forward pass with hooks for intermediate layer attribution."""
        if layer_name is None:
            # Standard forward pass for input attribution
            if hasattr(self.model, 'forward'):
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    return outputs[0]  # Return logits
                return outputs
            else:
                raise ValueError("Model must have a forward method")
        else:
            # Hook-based forward pass for specific layer
            return self._forward_with_layer_hook(inputs, layer_name)
    
    def _forward_with_layer_hook(
        self,
        inputs: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """Forward pass with hook on specific layer."""
        activation = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activation[name] = output
            return hook
        
        # Register hook
        target_module = self._get_module_by_name(layer_name)
        handle = target_module.register_forward_hook(hook_fn(layer_name))
        
        try:
            # Forward pass
            self.model(inputs)
            
            # Get activation from target layer
            if layer_name in activation:
                return activation[layer_name]
            else:
                raise ValueError(f"Layer {layer_name} not found in activations")
        finally:
            handle.remove()
    
    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get module by name from model."""
        module_dict = dict(self.model.named_modules())
        if name in module_dict:
            return module_dict[name]
        else:
            raise ValueError(f"Module {name} not found in model")
    
    def _compute_wirtinger_attribution(
        self,
        loss: torch.Tensor,
        inputs: torch.Tensor,
        layer_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute attribution using Wirtinger calculus.
        Implements Equations 6-8 from the paper.
        """
        # Compute Wirtinger derivatives
        grad_z, grad_z_conj = WirtingerGradient.compute_wirtinger_derivatives(
            loss, inputs, 
            create_graph=self.config.create_graph,
            retain_graph=True
        )
        
        # Compute attribution magnitude (Equation 8)
        attribution = WirtingerGradient.compute_attribution_magnitude(
            grad_z, grad_z_conj, method=self.config.attribution_method
        )
        
        # Cache gradients for analysis
        self._gradient_cache = {
            'grad_z': grad_z.detach(),
            'grad_z_conj': grad_z_conj.detach(),
            'attribution': attribution.detach()
        }
        
        return attribution
    
    def _normalize_attribution(self, attribution: torch.Tensor) -> torch.Tensor:
        """Normalize attribution values."""
        # Flatten for normalization
        batch_size = attribution.shape[0]
        flattened = attribution.view(batch_size, -1)
        
        # Min-max normalization per sample
        min_vals = torch.min(flattened, dim=1, keepdim=True)[0]
        max_vals = torch.max(flattened, dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        
        normalized_flat = (flattened - min_vals) / range_vals
        return normalized_flat.view_as(attribution)
    
    def _smooth_attribution(self, attribution: torch.Tensor) -> torch.Tensor:
        """Apply smoothing to attribution maps."""
        if attribution.dim() < 3:
            return attribution
        
        kernel_size = self.config.smooth_kernel_size
        
        # Create smoothing kernel
        if kernel_size > 1:
            # Use 1D convolution for sequence smoothing
            with torch.no_grad():
                # Reshape for convolution: (batch * features, 1, seq_len)
                batch_size, seq_len, features = attribution.shape
                reshaped = attribution.permute(0, 2, 1).contiguous()
                reshaped = reshaped.view(batch_size * features, 1, seq_len)
                
                # Create Gaussian kernel
                kernel = self._create_smoothing_kernel(kernel_size, attribution.device)
                
                # Apply convolution with padding
                padding = kernel_size // 2
                smoothed = F.conv1d(
                    reshaped, kernel, padding=padding, groups=1
                )
                
                # Reshape back
                smoothed = smoothed.view(batch_size, features, seq_len)
                smoothed = smoothed.permute(0, 2, 1).contiguous()
                
                return smoothed
        
        return attribution
    
    def _create_smoothing_kernel(self, kernel_size: int, device: torch.device) -> torch.Tensor:
        """Create Gaussian smoothing kernel."""
        if kernel_size == 1:
            return torch.tensor([1.0], device=device).view(1, 1, 1)
        
        # Create 1D Gaussian kernel
        sigma = kernel_size / 3.0
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        x = x - kernel_size // 2
        
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, kernel_size)
    
    def get_gradient_components(self) -> Dict[str, torch.Tensor]:
        """Get the cached Wirtinger gradient components for analysis."""
        return self._gradient_cache.copy()
    
    def compare_attribution_methods(
        self,
        inputs: torch.Tensor,
        target: Union[int, torch.Tensor],
        methods: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compare different attribution computation methods.
        
        Args:
            inputs: Input tensor
            target: Target class
            methods: List of methods to compare
            
        Returns:
            Dictionary mapping method names to attribution tensors
        """
        if methods is None:
            methods = ["squared_magnitude", "magnitude", "real_part", "euclidean_norm"]
        
        results = {}
        
        # Store original method
        original_method = self.config.attribution_method
        
        for method in methods:
            if method in ["squared_magnitude", "magnitude", "real_part", "euclidean_norm"]:
                self.config.attribution_method = method
                attribution = self.attribute(inputs, target, normalize=False, smooth=False)
                results[method] = attribution
            else:
                warnings.warn(f"Unknown attribution method: {method}")
        
        # Restore original method
        self.config.attribution_method = original_method
        
        return results


class IntegratedQISA(QISAAttributor):
    """
    Integrated QISA that accumulates gradients along paths.
    Extends basic QISA with path integration for smoother attributions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QISAConfig,
        baseline: str = "zero",
        steps: int = 50,
        device: Optional[torch.device] = None
    ):
        super().__init__(model, config, device)
        self.baseline = baseline
        self.steps = steps
    
    def attribute(
        self,
        inputs: torch.Tensor,
        target: Union[int, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute integrated QISA attribution.
        
        Integrates gradients along a path from baseline to input.
        """
        # Create baseline
        baseline_inputs = self._create_baseline(inputs)
        
        # Create path from baseline to input
        alphas = torch.linspace(0, 1, self.steps, device=self.device)
        
        # Accumulate gradients along path
        integrated_gradients = torch.zeros_like(inputs)
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated_inputs = baseline_inputs + alpha * (inputs - baseline_inputs)
            interpolated_inputs.requires_grad_(True)
            
            # Forward pass
            outputs = self._forward_with_hooks(interpolated_inputs)
            
            # Handle target
            if isinstance(target, int):
                target_tensor = torch.tensor([target], device=self.device)
            else:
                target_tensor = target.to(self.device)
            
            # Compute loss
            if outputs.dim() == 3:
                pooled_outputs = torch.mean(outputs, dim=1)
            else:
                pooled_outputs = outputs
            
            loss = F.cross_entropy(pooled_outputs, target_tensor)
            
            # Compute Wirtinger gradients
            grad_z, grad_z_conj = WirtingerGradient.compute_wirtinger_derivatives(
                loss, interpolated_inputs, create_graph=False
            )
            
            # Compute attribution for this step
            step_attribution = WirtingerGradient.compute_attribution_magnitude(
                grad_z, grad_z_conj, method=self.config.attribution_method
            )
            
            # Accumulate
            integrated_gradients += step_attribution / self.steps
        
        # Multiply by (input - baseline) for final integrated attribution
        final_attribution = integrated_gradients * (inputs - baseline_inputs).abs()
        
        # Post-processing
        if self.config.normalize_attributions:
            final_attribution = self._normalize_attribution(final_attribution)
        
        if self.config.smooth_attributions:
            final_attribution = self._smooth_attribution(final_attribution)
        
        return final_attribution.detach()
    
    def _create_baseline(self, inputs: torch.Tensor) -> torch.Tensor:
        """Create baseline for integration."""
        if self.baseline == "zero":
            return torch.zeros_like(inputs)
        elif self.baseline == "mean":
            # Use mean of input as baseline
            mean_real = torch.mean(inputs.real)
            mean_imag = torch.mean(inputs.imag)
            return torch.complex(
                torch.full_like(inputs.real, mean_real),
                torch.full_like(inputs.imag, mean_imag)
            )
        elif self.baseline == "random":
            # Random complex baseline
            return torch.randn_like(inputs) * 0.1
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline}")


class LayerwiseQISA(QISAAttributor):
    """
    Layer-wise QISA that computes attributions for multiple layers.
    Useful for understanding information flow through the network.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QISAConfig,
        target_layers: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__(model, config, device)
        
        if target_layers is None:
            # Auto-detect complex layers
            self.target_layers = self._find_complex_layers()
        else:
            self.target_layers = target_layers
    
    def _find_complex_layers(self) -> List[str]:
        """Automatically find layers with complex parameters."""
        complex_layers = []
        
        for name, module in self.model.named_modules():
            # Check if module has complex parameters
            has_complex = any(
                torch.is_complex(param) for param in module.parameters()
            )
            
            if has_complex and len(list(module.children())) == 0:  # Leaf module
                complex_layers.append(name)
        
        return complex_layers
    
    def attribute_all_layers(
        self,
        inputs: torch.Tensor,
        target: Union[int, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attributions for all target layers.
        
        Returns:
            Dictionary mapping layer names to attribution tensors
        """
        layer_attributions = {}
        
        for layer_name in self.target_layers:
            try:
                attribution = self.attribute(
                    inputs, target, layer_name=layer_name
                )
                layer_attributions[layer_name] = attribution
            except Exception as e:
                warnings.warn(f"Failed to compute attribution for layer {layer_name}: {e}")
                continue
        
        return layer_attributions
    
    def get_attribution_flow(
        self,
        inputs: torch.Tensor,
        target: Union[int, torch.Tensor],
        aggregation: str = "mean"
    ) -> Dict[str, float]:
        """
        Analyze attribution flow through layers.
        
        Args:
            inputs: Input tensor
            target: Target class
            aggregation: How to aggregate attribution per layer ('mean', 'sum', 'max')
            
        Returns:
            Dictionary mapping layer names to aggregated attribution values
        """
        layer_attributions = self.attribute_all_layers(inputs, target)
        attribution_flow = {}
        
        for layer_name, attribution in layer_attributions.items():
            if aggregation == "mean":
                flow_value = torch.mean(attribution).item()
            elif aggregation == "sum":
                flow_value = torch.sum(attribution).item()
            elif aggregation == "max":
                flow_value = torch.max(attribution).item()
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            attribution_flow[layer_name] = flow_value
        
        return attribution_flow


class QISAVisualizer:
    """
    Visualizer for QISA attribution results.
    Creates publication-quality visualizations of attribution maps.
    """
    
    def __init__(self, attributor: QISAAttributor):
        self.attributor = attributor
    
    def visualize_attribution(
        self,
        inputs: torch.Tensor,
        attribution: torch.Tensor,
        target: Optional[int] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> 'matplotlib.figure.Figure':
        """
        Create visualization of QISA attribution.
        
        Args:
            inputs: Original input tensor
            attribution: Attribution tensor from QISA
            target: Target class (for title)
            save_path: Path to save figure
            title: Custom title
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from utils.visualization import QISAVisualizer as VizUtils
        except ImportError:
            raise ImportError("Matplotlib required for visualization")
        
        # Use the visualization utilities we created earlier
        viz_utils = VizUtils()
        
        # Convert tensors for visualization
        if torch.is_complex(inputs):
            inputs_for_viz = inputs
        else:
            # Convert real inputs to complex for consistent visualization
            inputs_for_viz = torch.complex(inputs, torch.zeros_like(inputs))
        
        # Create the plot
        fig = viz_utils.plot_attribution_map(
            inputs_for_viz[0],  # First batch element
            attribution[0],     # First batch element
            title=title or f"QISA Attribution (Target: {target})"
        )
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_methods_visualization(
        self,
        inputs: torch.Tensor,
        target: Union[int, torch.Tensor],
        methods: List[str] = None,
        save_path: Optional[str] = None
    ) -> 'matplotlib.figure.Figure':
        """
        Visualize comparison of different attribution methods.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for visualization")
        
        # Get attributions for different methods
        method_attributions = self.attributor.compare_attribution_methods(
            inputs, target, methods
        )
        
        # Create subplot for each method
        n_methods = len(method_attributions)
        fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_name, attribution) in enumerate(method_attributions.items()):
            ax = axes[idx]
            
            # Plot attribution as heatmap
            attr_data = attribution[0].cpu().numpy()  # First batch element
            if attr_data.ndim == 2:
                im = ax.imshow(attr_data.T, aspect='auto', origin='lower', cmap='hot')
                plt.colorbar(im, ax=ax)
            
            ax.set_title(f'{method_name}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class QISAMetrics:
    """
    Metrics for evaluating QISA attribution quality.
    Implements faithfulness and stability measures.
    """
    
    def __init__(self, attributor: QISAAttributor):
        self.attributor = attributor
    
    def compute_faithfulness(
        self,
        inputs: torch.Tensor,
        attribution: torch.Tensor,
        target: Union[int, torch.Tensor],
        steps: int = 20
    ) -> Dict[str, float]:
        """
        Compute faithfulness metrics using deletion/insertion curves.
        
        Args:
            inputs: Input tensor
            attribution: Attribution tensor
            target: Target class
            steps: Number of steps for deletion/insertion
            
        Returns:
            Dictionary with faithfulness metrics
        """
        # Get model predictions for original input
        with torch.no_grad():
            original_outputs = self.attributor.model(inputs)
            if isinstance(original_outputs, tuple):
                original_outputs = original_outputs[0]
            
            if original_outputs.dim() == 3:
                original_outputs = torch.mean(original_outputs, dim=1)
            
            original_pred = torch.softmax(original_outputs, dim=-1)
            if isinstance(target, int):
                original_score = original_pred[0, target].item()
            else:
                original_score = original_pred[0, target[0]].item()
        
        # Deletion curve
        deletion_scores = self._compute_deletion_curve(
            inputs, attribution, target, steps
        )
        
        # Insertion curve
        insertion_scores = self._compute_insertion_curve(
            inputs, attribution, target, steps
        )
        
        # Compute AUC scores
        x_vals = np.linspace(0, 1, steps)
        deletion_auc = np.trapz(deletion_scores, x_vals)
        insertion_auc = np.trapz(insertion_scores, x_vals)
        
        return {
            'deletion_auc': deletion_auc,
            'insertion_auc': insertion_auc,
            'faithfulness_correlation': np.corrcoef(deletion_scores, 1 - x_vals)[0, 1],
            'original_score': original_score
        }
    
    def _compute_deletion_curve(
        self,
        inputs: torch.Tensor,
        attribution: torch.Tensor,
        target: Union[int, torch.Tensor],
        steps: int
    ) -> np.ndarray:
        """Compute deletion curve by removing most important features."""
        scores = []
        
        # Get flattened attribution for sorting
        batch_size, seq_len, features = inputs.shape
        attr_flat = attribution.view(batch_size, -1)
        
        # Sort by attribution magnitude (descending)
        _, sorted_indices = torch.sort(attr_flat, dim=1, descending=True)
        
        for step in range(steps):
            # Fraction of features to remove
            fraction_to_remove = step / (steps - 1)
            num_to_remove = int(fraction_to_remove * attr_flat.shape[1])
            
            # Create masked input
            masked_input = inputs.clone()
            
            if num_to_remove > 0:
                # Get indices to remove for each batch element
                remove_indices = sorted_indices[:, :num_to_remove]
                
                # Create mask
                mask = torch.ones_like(attr_flat)
                for b in range(batch_size):
                    mask[b, remove_indices[b]] = 0
                
                # Apply mask (set removed features to zero)
                mask_reshaped = mask.view(batch_size, seq_len, features)
                masked_input = masked_input * mask_reshaped
            
            # Get prediction for masked input
            with torch.no_grad():
                outputs = self.attributor.model(masked_input)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if outputs.dim() == 3:
                    outputs = torch.mean(outputs, dim=1)
                
                pred = torch.softmax(outputs, dim=-1)
                if isinstance(target, int):
                    score = pred[0, target].item()
                else:
                    score = pred[0, target[0]].item()
                
                scores.append(score)
        
        return np.array(scores)
    
    def _compute_insertion_curve(
        self,
        inputs: torch.Tensor,
        attribution: torch.Tensor,
        target: Union[int, torch.Tensor],
        steps: int
    ) -> np.ndarray:
        """Compute insertion curve by gradually adding most important features."""
        scores = []
        
        # Start with zero baseline
        baseline = torch.zeros_like(inputs)
        
        # Get flattened attribution for sorting
        batch_size, seq_len, features = inputs.shape
        attr_flat = attribution.view(batch_size, -1)
        
        # Sort by attribution magnitude (descending)
        _, sorted_indices = torch.sort(attr_flat, dim=1, descending=True)
        
        for step in range(steps):
            # Fraction of features to include
            fraction_to_include = step / (steps - 1)
            num_to_include = int(fraction_to_include * attr_flat.shape[1])
            
            # Start with baseline
            modified_input = baseline.clone()
            
            if num_to_include > 0:
                # Get indices to include for each batch element
                include_indices = sorted_indices[:, :num_to_include]
                
                # Create mask
                mask = torch.zeros_like(attr_flat)
                for b in range(batch_size):
                    mask[b, include_indices[b]] = 1
                
                # Apply mask (keep only selected features from original input)
                mask_reshaped = mask.view(batch_size, seq_len, features)
                modified_input = baseline + mask_reshaped * (inputs - baseline)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.attributor.model(modified_input)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if outputs.dim() == 3:
                    outputs = torch.mean(outputs, dim=1)
                
                pred = torch.softmax(outputs, dim=-1)
                if isinstance(target, int):
                    score = pred[0, target].item()
                else:
                    score = pred[0, target[0]].item()
                
                scores.append(score)
        
        return np.array(scores)
    
    def compute_stability(
        self,
        inputs: torch.Tensor,
        target: Union[int, torch.Tensor],
        noise_level: float = 0.1,
        num_samples: int = 10
    ) -> float:
        """
        Compute stability of attribution under input perturbations.
        
        Args:
            inputs: Input tensor
            target: Target class
            noise_level: Standard deviation of Gaussian noise
            num_samples: Number of noisy samples to test
            
        Returns:
            Average correlation between original and perturbed attributions
        """
        # Get original attribution
        original_attr = self.attributor.attribute(inputs, target)
        
        correlations = []
        
        for _ in range(num_samples):
            # Add complex noise
            noise = torch.randn_like(inputs) * noise_level
            noisy_inputs = inputs + noise
            
            # Get attribution for noisy input
            noisy_attr = self.attributor.attribute(noisy_inputs, target)
            
            # Compute correlation
            orig_flat = original_attr.flatten().cpu().numpy()
            noisy_flat = noisy_attr.flatten().cpu().numpy()
            
            correlation = np.corrcoef(orig_flat, noisy_flat)[0, 1]
            if not np.isnan(correlation):
                correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.0


def create_qisa_attributor(
    model: nn.Module,
    config: QISAConfig,
    method: str = "standard",
    **kwargs
) -> QISAAttributor:
    """
    Factory function to create different types of QISA attributors.
    
    Args:
        model: Complex-valued model
        config: QISA configuration
        method: Type of attributor ('standard', 'integrated', 'layerwise')
        **kwargs: Additional arguments for specific attributor types
        
    Returns:
        QISA attributor instance
    """
    if method == "standard":
        return QISAAttributor(model, config, **kwargs)
    elif method == "integrated":
        return IntegratedQISA(model, config, **kwargs)
    elif method == "layerwise":
        return LayerwiseQISA(model, config, **kwargs)
    else:
        raise ValueError(f"Unknown QISA method: {method}")


# Utility functions for QISA analysis

def analyze_attribution_patterns(
    attribution: torch.Tensor,
    inputs: torch.Tensor,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Analyze patterns in QISA attribution maps.
    
    Args:
        attribution: Attribution tensor
        inputs: Original input tensor
        threshold: Threshold for significant attribution
        
    Returns:
        Dictionary with analysis results
    """
    # Convert to numpy for analysis
    attr_np = attribution.cpu().numpy()
    
    # Compute statistics
    analysis = {
        'mean_attribution': np.mean(attr_np),
        'std_attribution': np.std(attr_np),
        'max_attribution': np.max(attr_np),
        'min_attribution': np.min(attr_np),
        'sparsity': np.mean(attr_np < threshold),
        'attribution_range': np.max(attr_np) - np.min(attr_np)
    }
    
    # Temporal patterns (if applicable)
    if attr_np.ndim >= 2:
        analysis['temporal_variance'] = np.var(np.mean(attr_np, axis=-1))
        analysis['feature_variance'] = np.var(np.mean(attr_np, axis=-2))
        
        # Find peaks in attribution
        if attr_np.shape[0] == 1:  # Single sample
            temporal_profile = np.mean(attr_np[0], axis=-1)
            peaks = []
            for i in range(1, len(temporal_profile) - 1):
                if (temporal_profile[i] > temporal_profile[i-1] and 
                    temporal_profile[i] > temporal_profile[i+1] and
                    temporal_profile[i] > threshold):
                    peaks.append(i)
            analysis['attribution_peaks'] = peaks
    
    return analysis


def batch_qisa_attribution(
    attributor: QISAAttributor,
    inputs_list: List[torch.Tensor],
    targets_list: List[Union[int, torch.Tensor]],
    batch_size: int = 8,
    progress_callback: Optional[Callable] = None
) -> List[torch.Tensor]:
    """
    Compute QISA attributions for a batch of inputs efficiently.
    
    Args:
        attributor: QISA attributor instance
        inputs_list: List of input tensors
        targets_list: List of target classes
        batch_size: Processing batch size
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of attribution tensors
    """
    attributions = []
    
    for i in range(0, len(inputs_list), batch_size):
        batch_inputs = inputs_list[i:i+batch_size]
        batch_targets = targets_list[i:i+batch_size]
        
        batch_attributions = []
        
        for j, (inputs, target) in enumerate(zip(batch_inputs, batch_targets)):
            attribution = attributor.attribute(inputs.unsqueeze(0), target)
            batch_attributions.append(attribution.squeeze(0))
            
            # Progress callback
            if progress_callback:
                progress_callback(i + j + 1, len(inputs_list))
        
        attributions.extend(batch_attributions)
    
    return attributions


def compare_qisa_with_baselines(
    model: nn.Module,
    inputs: torch.Tensor,
    target: Union[int, torch.Tensor],
    config: QISAConfig,
    baseline_methods: List[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Compare QISA with baseline attribution methods.
    
    Args:
        model: Complex-valued model
        inputs: Input tensor
        target: Target class
        config: QISA configuration
        baseline_methods: List of baseline methods to compare with
        
    Returns:
        Dictionary mapping method names to attribution tensors
    """
    if baseline_methods is None:
        baseline_methods = ['gradient', 'integrated_gradients', 'grad_cam']
    
    results = {}
    
    # QISA attribution
    qisa_attributor = QISAAttributor(model, config)
    results['qisa'] = qisa_attributor.attribute(inputs, target)
    
    # Baseline methods (simplified implementations)
    for method in baseline_methods:
        try:
            if method == 'gradient':
                results[method] = _compute_gradient_attribution(model, inputs, target)
            elif method == 'integrated_gradients':
                results[method] = _compute_integrated_gradients(model, inputs, target)
            elif method == 'grad_cam':
                results[method] = _compute_grad_cam(model, inputs, target)
            else:
                print(f"Warning: Unknown baseline method {method}")
        except Exception as e:
            print(f"Error computing {method}: {e}")
            continue
    
    return results


def _compute_gradient_attribution(
    model: nn.Module,
    inputs: torch.Tensor,
    target: Union[int, torch.Tensor]
) -> torch.Tensor:
    """Compute simple gradient attribution (for real inputs only)."""
    inputs.requires_grad_(True)
    
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    if outputs.dim() == 3:
        outputs = torch.mean(outputs, dim=1)
    
    if isinstance(target, int):
        target_tensor = torch.tensor([target], device=inputs.device)
    else:
        target_tensor = target
    
    loss = F.cross_entropy(outputs, target_tensor)
    
    gradients = torch.autograd.grad(
        outputs=loss,
        inputs=inputs,
        create_graph=False,
        retain_graph=False
    )[0]
    
    # For complex inputs, take magnitude
    if torch.is_complex(gradients):
        gradients = torch.abs(gradients)
    
    return gradients.detach()


def _compute_integrated_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    target: Union[int, torch.Tensor],
    steps: int = 50
) -> torch.Tensor:
    """Compute integrated gradients (simplified version)."""
    baseline = torch.zeros_like(inputs)
    
    # Scale inputs and compute gradients
    alphas = torch.linspace(0, 1, steps, device=inputs.device)
    integrated_grads = torch.zeros_like(inputs)
    
    for alpha in alphas:
        scaled_inputs = baseline + alpha * (inputs - baseline)
        scaled_inputs.requires_grad_(True)
        
        outputs = model(scaled_inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        if outputs.dim() == 3:
            outputs = torch.mean(outputs, dim=1)
        
        if isinstance(target, int):
            target_tensor = torch.tensor([target], device=inputs.device)
        else:
            target_tensor = target
        
        loss = F.cross_entropy(outputs, target_tensor)
        
        gradients = torch.autograd.grad(
            outputs=loss,
            inputs=scaled_inputs,
            create_graph=False,
            retain_graph=False
        )[0]
        
        if torch.is_complex(gradients):
            gradients = torch.abs(gradients)
        
        integrated_grads += gradients / steps
    
    # Multiply by (input - baseline)
    if torch.is_complex(inputs):
        path_diff = torch.abs(inputs - baseline)
    else:
        path_diff = inputs - baseline
    
    return (integrated_grads * path_diff).detach()


def _compute_grad_cam(
    model: nn.Module,
    inputs: torch.Tensor,
    target: Union[int, torch.Tensor]
) -> torch.Tensor:
    """Compute simplified Grad-CAM (works for convolutional-like layers)."""
    # This is a simplified version - real Grad-CAM needs specific layer access
    inputs.requires_grad_(True)
    
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    if outputs.dim() == 3:
        outputs = torch.mean(outputs, dim=1)
    
    if isinstance(target, int):
        target_tensor = torch.tensor([target], device=inputs.device)
    else:
        target_tensor = target
    
    # Get the target class score
    target_score = outputs[0, target_tensor[0]]
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=target_score,
        inputs=inputs,
        create_graph=False,
        retain_graph=False
    )[0]
    
    # For complex inputs, take magnitude
    if torch.is_complex(gradients):
        gradients = torch.abs(gradients)
    
    # Apply ReLU to get positive contributions
    grad_cam = F.relu(gradients)
    
    return grad_cam.detach()


class QISABenchmark:
    """
    Benchmark suite for evaluating QISA performance and quality.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QISAConfig,
        test_inputs: List[torch.Tensor],
        test_targets: List[Union[int, torch.Tensor]]
    ):
        self.model = model
        self.config = config
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.attributor = QISAAttributor(model, config)
        self.metrics_calculator = QISAMetrics(self.attributor)
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive QISA benchmark.
        
        Returns:
            Dictionary with benchmark results
        """
        results = {
            'faithfulness_scores': [],
            'stability_scores': [],
            'computation_times': [],
            'attribution_statistics': [],
            'method_comparisons': {}
        }
        
        print("Running QISA benchmark...")
        
        for i, (inputs, target) in enumerate(zip(self.test_inputs, self.test_targets)):
            print(f"Processing sample {i+1}/{len(self.test_inputs)}")
            
            # Time the attribution computation
            import time
            start_time = time.time()
            attribution = self.attributor.attribute(inputs.unsqueeze(0), target)
            computation_time = time.time() - start_time
            
            results['computation_times'].append(computation_time)
            
            # Compute faithfulness
            faithfulness = self.metrics_calculator.compute_faithfulness(
                inputs.unsqueeze(0), attribution, target
            )
            results['faithfulness_scores'].append(faithfulness)
            
            # Compute stability
            stability = self.metrics_calculator.compute_stability(
                inputs.unsqueeze(0), target
            )
            results['stability_scores'].append(stability)
            
            # Attribution statistics
            stats = analyze_attribution_patterns(attribution, inputs.unsqueeze(0))
            results['attribution_statistics'].append(stats)
        
        # Aggregate results
        results['summary'] = self._aggregate_results(results)
        
        return results
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate benchmark results into summary statistics."""
        summary = {}
        
        # Faithfulness metrics
        if results['faithfulness_scores']:
            deletion_aucs = [f['deletion_auc'] for f in results['faithfulness_scores']]
            insertion_aucs = [f['insertion_auc'] for f in results['faithfulness_scores']]
            
            summary['mean_deletion_auc'] = np.mean(deletion_aucs)
            summary['std_deletion_auc'] = np.std(deletion_aucs)
            summary['mean_insertion_auc'] = np.mean(insertion_aucs)
            summary['std_insertion_auc'] = np.std(insertion_aucs)
        
        # Stability
        if results['stability_scores']:
            summary['mean_stability'] = np.mean(results['stability_scores'])
            summary['std_stability'] = np.std(results['stability_scores'])
        
        # Computation time
        if results['computation_times']:
            summary['mean_computation_time'] = np.mean(results['computation_times'])
            summary['std_computation_time'] = np.std(results['computation_times'])
        
        # Attribution statistics
        if results['attribution_statistics']:
            sparsity_values = [s['sparsity'] for s in results['attribution_statistics']]
            summary['mean_sparsity'] = np.mean(sparsity_values)
            summary['std_sparsity'] = np.std(sparsity_values)
        
        return summary
    
    def generate_report(self, results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("QISA Benchmark Report")
        report.append("=" * 50)
        report.append("")
        
        summary = results.get('summary', {})
        
        report.append("Faithfulness Metrics:")
        report.append(f"  Deletion AUC: {summary.get('mean_deletion_auc', 0):.4f} ± {summary.get('std_deletion_auc', 0):.4f}")
        report.append(f"  Insertion AUC: {summary.get('mean_insertion_auc', 0):.4f} ± {summary.get('std_insertion_auc', 0):.4f}")
        report.append("")
        
        report.append("Stability Metrics:")
        report.append(f"  Mean Stability: {summary.get('mean_stability', 0):.4f} ± {summary.get('std_stability', 0):.4f}")
        report.append("")
        
        report.append("Performance Metrics:")
        report.append(f"  Mean Computation Time: {summary.get('mean_computation_time', 0):.4f} ± {summary.get('std_computation_time', 0):.4f} seconds")
        report.append("")
        
        report.append("Attribution Quality:")
        report.append(f"  Mean Sparsity: {summary.get('mean_sparsity', 0):.4f} ± {summary.get('std_sparsity', 0):.4f}")
        report.append("")
        
        # Configuration details
        report.append("Configuration:")
        report.append(f"  Attribution Method: {self.config.attribution_method}")
        report.append(f"  Wirtinger Alpha: {self.config.wirtinger_alpha}")
        report.append(f"  Normalize Attributions: {self.config.normalize_attributions}")
        report.append(f"  Smooth Attributions: {self.config.smooth_attributions}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


class QISAExplainer:
    """
    High-level explainer interface for QISA.
    Provides easy-to-use methods for explaining model predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QISAConfig,
        class_names: Optional[List[str]] = None
    ):
        self.model = model
        self.config = config
        self.class_names = class_names
        self.attributor = QISAAttributor(model, config)
        self.visualizer = QISAVisualizer(self.attributor)
    
    def explain(
        self,
        inputs: torch.Tensor,
        target: Optional[Union[int, torch.Tensor]] = None,
        top_k: int = 1,
        visualize: bool = False,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for model prediction.
        
        Args:
            inputs: Input tensor to explain
            target: Target class (if None, uses predicted class)
            top_k: Number of top predictions to explain
            visualize: Whether to generate visualizations
            save_path: Path to save visualizations
            
        Returns:
            Dictionary with explanation results
        """
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(inputs.unsqueeze(0))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            if outputs.dim() == 3:
                outputs = torch.mean(outputs, dim=1)
            
            probabilities = torch.softmax(outputs, dim=-1)
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
        
        # If no target specified, use top prediction
        if target is None:
            target = top_indices[0, 0].item()
        
        # Compute QISA attribution
        attribution = self.attributor.attribute(inputs.unsqueeze(0), target)
        
        # Analyze attribution patterns
        attribution_analysis = analyze_attribution_patterns(attribution, inputs.unsqueeze(0))
        
        # Create explanation dictionary
        explanation = {
            'predicted_class': top_indices[0, 0].item(),
            'predicted_probability': top_probs[0, 0].item(),
            'top_k_predictions': [
                {
                    'class_idx': top_indices[0, i].item(),
                    'class_name': self.class_names[top_indices[0, i].item()] if self.class_names else None,
                    'probability': top_probs[0, i].item()
                }
                for i in range(top_k)
            ],
            'target_class': target,
            'target_class_name': self.class_names[target] if self.class_names else None,
            'attribution': attribution,
            'attribution_analysis': attribution_analysis
        }
        
        # Add visualization if requested
        if visualize:
            try:
                fig = self.visualizer.visualize_attribution(
                    inputs.unsqueeze(0), attribution, target, save_path
                )
                explanation['visualization'] = fig
            except Exception as e:
                print(f"Visualization failed: {e}")
        
        return explanation
    
    def explain_differences(
        self,
        inputs1: torch.Tensor,
        inputs2: torch.Tensor,
        target: Optional[Union[int, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Explain differences between two inputs using QISA.
        
        Args:
            inputs1: First input tensor
            inputs2: Second input tensor
            target: Target class for attribution
            
        Returns:
            Dictionary with difference analysis
        """
        # Get attributions for both inputs
        attr1 = self.attributor.attribute(inputs1.unsqueeze(0), target)
        attr2 = self.attributor.attribute(inputs2.unsqueeze(0), target)
        
        # Compute attribution difference
        attr_diff = attr2 - attr1
        
        # Analyze the differences
        analysis = {
            'attribution_1': attr1,
            'attribution_2': attr2,
            'attribution_difference': attr_diff,
            'difference_magnitude': torch.norm(attr_diff).item(),
            'correlation': torch.corrcoef(torch.stack([
                attr1.flatten(), attr2.flatten()
            ]))[0, 1].item()
        }
        
        return analysis
    
    def batch_explain(
        self,
        inputs_list: List[torch.Tensor],
        targets_list: Optional[List[Union[int, torch.Tensor]]] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of inputs.
        
        Args:
            inputs_list: List of input tensors
            targets_list: List of target classes (optional)
            progress_callback: Optional progress callback
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i, inputs in enumerate(inputs_list):
            target = targets_list[i] if targets_list else None
            
            explanation = self.explain(inputs, target, visualize=False)
            explanations.append(explanation)
            
            if progress_callback:
                progress_callback(i + 1, len(inputs_list))
        
        return explanations


# Main interface functions

def explain_with_qisa(
    model: nn.Module,
    inputs: torch.Tensor,
    target: Optional[Union[int, torch.Tensor]] = None,
    config: Optional[QISAConfig] = None,
    method: str = "standard",
    **kwargs
) -> torch.Tensor:
    """
    Main interface function for QISA explanation.
    
    Args:
        model: Complex-valued model to explain
        inputs: Input tensor
        target: Target class
        config: QISA configuration (uses default if None)
        method: Attribution method ('standard', 'integrated', 'layerwise')
        **kwargs: Additional arguments
        
    Returns:
        Attribution tensor
    """
    if config is None:
        config = QISAConfig()
    
    attributor = create_qisa_attributor(model, config, method, **kwargs)
    return attributor.attribute(inputs, target)


def benchmark_qisa(
    model: nn.Module,
    test_data: List[Tuple[torch.Tensor, Union[int, torch.Tensor]]],
    config: Optional[QISAConfig] = None,
    save_report: bool = True,
    report_path: str = "qisa_benchmark_report.txt"
) -> Dict[str, Any]:
    """
    Run QISA benchmark on test data.
    
    Args:
        model: Model to benchmark
        test_data: List of (input, target) tuples
        config: QISA configuration
        save_report: Whether to save benchmark report
        report_path: Path to save report
        
    Returns:
        Benchmark results dictionary
    """
    if config is None:
        config = QISAConfig()
    
    inputs_list = [data[0] for data in test_data]
    targets_list = [data[1] for data in test_data]
    
    benchmark = QISABenchmark(model, config, inputs_list, targets_list)
    results = benchmark.run_benchmark()
    
    if save_report:
        benchmark.generate_report(results, report_path)
        print(f"Benchmark report saved to {report_path}")
    
    return results