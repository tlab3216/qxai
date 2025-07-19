#!/usr/bin/env python3
"""
Q-XAI Model Evaluation Script

Comprehensive evaluation script for trained Q-XAI models.
Provides detailed analysis of classification performance, interpretability,
uncertainty quantification, and conformal prediction capabilities.

Usage:
    python evaluate_model.py --model_path checkpoints/best_model.pt --dataset dcase2019
    python evaluate_model.py --model_path checkpoints/best_model.pt --dataset esc50 --comprehensive
    python evaluate_model.py --model_path checkpoints/best_model.pt --robustness_test
"""

import argparse
import logging
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Q-XAI Framework imports
from models.complex_transformer import ComplexTransformerClassifier, create_complex_transformer
from interpretability.qisa import (
    QISAAttributor, create_qisa_attributor, QISAMetrics, QISAVisualizer,
    explain_with_qisa, benchmark_qisa
)
from interpretability.auq import (
    AUQEstimator, estimate_uncertainty_with_auq, AUQCalibrator,
    analyze_uncertainty_contributions, benchmark_auq
)
from interpretability.qicp import (
    QICPPredictor, create_qicp_predictor, run_qicp_evaluation,
    qicp_robustness_test, QICPVisualizer
)
from data.datasets import (
    TUT2016Dataset, DCASE2019Dataset, ESC50Dataset,
    CochlSceneDataset, DCASE2025Dataset, create_dataloader
)
from data.preprocessing import create_preprocessing_pipeline
from training.losses import ComplexCrossEntropyLoss
from utils.metrics import ClassificationMetrics, UncertaintyMetrics, InterpretabilityMetrics
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curves, plot_calibration_curve,
    create_comprehensive_report_plots
)
from config.model_config import ComplexTransformerConfig, QISAConfig, AUQConfig, QICPConfig
from config.data_config import (
    DatasetConfig, AudioConfig, SpectrogramConfig,
    TUT2016Config, DCASE2019Config, ESC50Config, CochlSceneConfig, DCASE2025Config
)


class QXAIModelEvaluator:
    """
    Comprehensive evaluator for trained Q-XAI models.
    
    Provides detailed analysis across all Q-XAI components:
    - Classification performance
    - QISA interpretability analysis
    - AUQ uncertainty quantification
    - QICP conformal prediction
    """
    
    def __init__(
        self,
        model_path: str,
        dataset_name: str,
        data_dir: str,
        device: Optional[torch.device] = None,
        output_dir: str = "evaluation_results"
    ):
        self.model_path = Path(model_path)
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model = None
        self.dataset_config = None
        self.test_loader = None
        self.qisa_attributor = None
        self.auq_estimator = None
        self.qicp_predictor = None
        
        # Results storage
        self.results = {
            'classification': {},
            'interpretability': {},
            'uncertainty': {},
            'conformal_prediction': {},
            'robustness': {}
        }
        
        self.logger.info(f"Initialized Q-XAI evaluator for {dataset_name}")
        self.logger.info(f"Model path: {model_path}")
        self.logger.info(f"Output directory: {output_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"evaluation_{self.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model_and_data(self):
        """Load trained model and setup data loaders."""
        self.logger.info("Loading model and setting up data")
        
        # Load checkpoint
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model configuration if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            if isinstance(config, dict):
                # Handle different config formats
                model_config = config.get('model', config)
            else:
                model_config = config.model if hasattr(config, 'model') else config
        else:
            # Default configuration if not found in checkpoint
            self.logger.warning("Model config not found in checkpoint, using defaults")
            model_config = self._get_default_model_config()
        
        # Setup dataset
        self._setup_dataset()
        
        # Create model
        self.model = create_complex_transformer(
            config=model_config,
            model_type="standard",
            num_classes=len(self.dataset_config.class_names)
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup Q-XAI components
        self._setup_qxai_components()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded - Parameters: {total_params:,}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def _get_default_model_config(self) -> ComplexTransformerConfig:
        """Get default model configuration."""
        return ComplexTransformerConfig(
            input_dim=64,
            embed_dim=256,
            num_layers=6,
            num_heads=8,
            ff_dim=1024,
            num_classes=10,  # Will be updated based on dataset
            max_seq_length=1024,
            dropout=0.1,
            activation='complex_gelu'
        )
    
    def _setup_dataset(self):
        """Setup dataset and data loader."""
        # Audio and spectrogram configs
        audio_config = AudioConfig(
            sample_rate=16000,
            mono=True,
            normalize=True,
            norm_type='peak'
        )
        
        spectrogram_config = SpectrogramConfig(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=64,
            f_min=0,
            f_max=8000,
            to_db=True,
            normalize_spec=True
        )
        
        # Dataset-specific configuration
        if self.dataset_name == 'tut2016':
            self.dataset_config = TUT2016Config(
                root_path=self.data_dir,
                class_names=[
                    'beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
                    'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
                    'office', 'park', 'residential_area', 'train', 'tram'
                ]
            )
            dataset_class = TUT2016Dataset
            
        elif self.dataset_name == 'dcase2019':
            self.dataset_config = DCASE2019Config(
                root_path=self.data_dir,
                class_names=[
                    'airport', 'bus', 'metro', 'metro_station', 'park',
                    'public_square', 'shopping_mall', 'street_pedestrian',
                    'street_traffic', 'tram'
                ]
            )
            dataset_class = DCASE2019Dataset
            
        elif self.dataset_name == 'esc50':
            self.dataset_config = ESC50Config(
                root_path=self.data_dir,
                class_names=[f'class_{i}' for i in range(50)]
            )
            dataset_class = ESC50Dataset
            
        elif self.dataset_name == 'cochlscene':
            self.dataset_config = CochlSceneConfig(
                root_path=self.data_dir,
                class_names=[
                    'airport', 'bus', 'metro', 'metro_station', 'park',
                    'public_square', 'shopping_mall', 'street_pedestrian',
                    'street_traffic', 'tram', 'beach', 'forest', 'train'
                ]
            )
            dataset_class = CochlSceneDataset
            
        elif self.dataset_name == 'dcase2025':
            self.dataset_config = DCASE2025Config(
                root_path=self.data_dir,
                class_names=[
                    'airport', 'bus', 'metro', 'metro_station', 'park',
                    'public_square', 'shopping_mall', 'street_pedestrian',
                    'street_traffic', 'tram'
                ]
            )
            dataset_class = DCASE2025Dataset
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Create test dataset
        test_dataset = dataset_class(
            config=self.dataset_config,
            audio_config=audio_config,
            spectrogram_config=spectrogram_config,
            split='test'
        )
        
        # Create data loader
        self.test_loader = create_dataloader(
            test_dataset, batch_size=32, shuffle=False, 
            num_workers=4, pin_memory=True
        )
    
    def _setup_qxai_components(self):
        """Setup Q-XAI interpretability components."""
        # QISA Attributor
        qisa_config = QISAConfig()
        self.qisa_attributor = create_qisa_attributor(
            self.model, qisa_config, method="standard"
        )
        
        # AUQ Estimator
        auq_config = AUQConfig()
        self.auq_estimator = AUQEstimator(
            self.model, auq_config, device=self.device
        )
        
        # QICP Predictor
        qicp_config = QICPConfig()
        self.qicp_predictor = create_qicp_predictor(
            qicp_config, score_type="born_rule"
        )
    
    def evaluate_classification(self) -> Dict[str, Any]:
        """Comprehensive classification performance evaluation."""
        self.logger.info("Evaluating classification performance")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_logits = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(self.device)
                    targets = batch['label'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    raise ValueError("Unsupported batch format")
                
                # Forward pass
                outputs = self.model(inputs, return_complex_amplitudes=True)
                
                if isinstance(outputs, tuple):
                    logits, complex_amplitudes, _ = outputs
                else:
                    logits = outputs
                    complex_amplitudes = None
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        all_logits = np.array(all_logits)
        
        # Compute comprehensive metrics
        results = {
            # Basic metrics
            'accuracy': accuracy_score(all_targets, all_predictions),
            'macro_f1': f1_score(all_targets, all_predictions, average='macro'),
            'weighted_f1': f1_score(all_targets, all_predictions, average='weighted'),
            'macro_precision': precision_score(all_targets, all_predictions, average='macro', zero_division=0),
            'macro_recall': recall_score(all_targets, all_predictions, average='macro', zero_division=0),
            
            # Per-class metrics
            'per_class_f1': f1_score(all_targets, all_predictions, average=None).tolist(),
            'per_class_precision': precision_score(all_targets, all_predictions, average=None, zero_division=0).tolist(),
            'per_class_recall': recall_score(all_targets, all_predictions, average=None, zero_division=0).tolist(),
            
            # Confusion matrix
            'confusion_matrix': confusion_matrix(all_targets, all_predictions).tolist(),
            
            # Classification report
            'classification_report': classification_report(
                all_targets, all_predictions, 
                target_names=self.dataset_config.class_names,
                output_dict=True
            )
        }
        
        # Compute calibration metrics
        calibration_metrics = UncertaintyMetrics.compute_calibration_metrics(
            torch.tensor(all_probabilities),
            torch.tensor(all_targets)
        )
        results.update(calibration_metrics)
        
        # Log results
        self.logger.info(f"Classification Results:")
        self.logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        self.logger.info(f"  Macro F1: {results['macro_f1']:.4f}")
        self.logger.info(f"  ECE: {results['ece']:.4f}")
        self.logger.info(f"  Brier Score: {results['brier_score']:.4f}")
        
        return results
    
    def evaluate_interpretability(self, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate QISA interpretability quality."""
        self.logger.info(f"Evaluating interpretability on {num_samples} samples")
        
        # Run QISA benchmark
        test_data = []
        sample_count = 0
        
        for batch in self.test_loader:
            if sample_count >= num_samples:
                break
                
            if isinstance(batch, dict):
                inputs = batch['spectrogram']
                targets = batch['label']
            else:
                inputs, targets = batch[0], batch[1]
            
            batch_size = min(inputs.size(0), num_samples - sample_count)
            
            for i in range(batch_size):
                test_data.append((inputs[i], targets[i]))
                sample_count += 1
                
                if sample_count >= num_samples:
                    break
        
        # Run QISA benchmark
        qisa_results = benchmark_qisa(
            self.model, test_data, save_report=True,
            report_path=str(self.output_dir / "qisa_benchmark_report.txt")
        )
        
        # Compute additional interpretability metrics
        sample_inputs = [data[0] for data in test_data[:10]]
        sample_targets = [data[1] for data in test_data[:10]]
        
        # Compute attribution quality metrics
        attribution_metrics = {
            'faithfulness_scores': [],
            'stability_scores': [],
            'sparsity_scores': []
        }
        
        for i, (inputs, target) in enumerate(zip(sample_inputs, sample_targets)):
            try:
                # QISA attribution
                attribution = explain_with_qisa(
                    self.model, inputs.unsqueeze(0), target.item()
                )
                
                # Compute faithfulness
                qisa_metrics = QISAMetrics(self.qisa_attributor)
                faithfulness = qisa_metrics.compute_faithfulness(
                    inputs.unsqueeze(0), attribution, target.item()
                )
                attribution_metrics['faithfulness_scores'].append(
                    faithfulness['deletion_auc']
                )
                
                # Compute stability
                stability = qisa_metrics.compute_stability(
                    inputs.unsqueeze(0), target.item()
                )
                attribution_metrics['stability_scores'].append(stability)
                
                # Compute sparsity
                sparsity = InterpretabilityMetrics.compute_sparsity(attribution)
                attribution_metrics['sparsity_scores'].append(sparsity)
                
            except Exception as e:
                self.logger.warning(f"Failed to compute attribution metrics for sample {i}: {e}")
                continue
        
        # Aggregate results
        results = {
            'benchmark_results': qisa_results,
            'average_faithfulness': np.mean(attribution_metrics['faithfulness_scores']) if attribution_metrics['faithfulness_scores'] else 0,
            'average_stability': np.mean(attribution_metrics['stability_scores']) if attribution_metrics['stability_scores'] else 0,
            'average_sparsity': np.mean(attribution_metrics['sparsity_scores']) if attribution_metrics['sparsity_scores'] else 0,
            'num_samples_analyzed': len(attribution_metrics['faithfulness_scores'])
        }
        
        self.logger.info(f"Interpretability Results:")
        self.logger.info(f"  Average Faithfulness: {results['average_faithfulness']:.4f}")
        self.logger.info(f"  Average Stability: {results['average_stability']:.4f}")
        self.logger.info(f"  Average Sparsity: {results['average_sparsity']:.4f}")
        
        return results
    
    def evaluate_uncertainty(self, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate AUQ uncertainty quantification."""
        self.logger.info(f"Evaluating uncertainty quantification on {num_samples} samples")
        
        # Prepare test data
        test_data = []
        sample_count = 0
        
        for batch in self.test_loader:
            if sample_count >= num_samples:
                break
                
            if isinstance(batch, dict):
                inputs = batch['spectrogram']
                targets = batch['label']
            else:
                inputs, targets = batch[0], batch[1]
            
            batch_size = min(inputs.size(0), num_samples - sample_count)
            
            for i in range(batch_size):
                test_data.append((inputs[i], targets[i]))
                sample_count += 1
                
                if sample_count >= num_samples:
                    break
        
        # Run AUQ benchmark
        auq_results = benchmark_auq(
            self.model, test_data, save_report=True,
            report_path=str(self.output_dir / "auq_benchmark_report.txt")
        )
        
        # Analyze uncertainty components for sample inputs
        sample_uncertainties = []
        
        for i, (inputs, target) in enumerate(test_data[:20]):
            try:
                uncertainty_components = estimate_uncertainty_with_auq(
                    self.model, inputs.unsqueeze(0)
                )
                
                sample_uncertainties.append({
                    'epistemic': torch.mean(uncertainty_components.epistemic).item(),
                    'aleatoric': torch.mean(uncertainty_components.aleatoric).item(),
                    'covariance': torch.mean(uncertainty_components.covariance).item(),
                    'total': torch.mean(uncertainty_components.total).item()
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to compute uncertainty for sample {i}: {e}")
                continue
        
        # Analyze uncertainty contributions
        if sample_uncertainties:
            uncertainty_analysis = analyze_uncertainty_contributions(
                uncertainty_components, self.dataset_config.class_names
            )
        else:
            uncertainty_analysis = {}
        
        # Aggregate results
        results = {
            'benchmark_results': auq_results,
            'uncertainty_analysis': uncertainty_analysis,
            'num_samples_analyzed': len(sample_uncertainties)
        }
        
        if sample_uncertainties:
            results.update({
                'mean_epistemic': np.mean([u['epistemic'] for u in sample_uncertainties]),
                'mean_aleatoric': np.mean([u['aleatoric'] for u in sample_uncertainties]),
                'mean_covariance': np.mean([u['covariance'] for u in sample_uncertainties]),
                'mean_total': np.mean([u['total'] for u in sample_uncertainties])
            })
        
        self.logger.info(f"Uncertainty Results:")
        if 'mean_total' in results:
            self.logger.info(f"  Mean Total Uncertainty: {results['mean_total']:.4f}")
            self.logger.info(f"  Mean Epistemic: {results['mean_epistemic']:.4f}")
            self.logger.info(f"  Mean Aleatoric: {results['mean_aleatoric']:.4f}")
            self.logger.info(f"  Mean Covariance: {results['mean_covariance']:.4f}")
        
        return results
    
    def evaluate_conformal_prediction(self) -> Dict[str, Any]:
        """Evaluate QICP conformal prediction."""
        self.logger.info("Evaluating conformal prediction")
        
        # Split test data for calibration and evaluation
        all_batches = list(self.test_loader)
        split_idx = len(all_batches) // 2
        
        cal_batches = all_batches[:split_idx]
        test_batches = all_batches[split_idx:]
        
        # Create temporary data loaders
        from torch.utils.data import TensorDataset
        
        # Extract calibration data
        cal_inputs, cal_labels = [], []
        for batch in cal_batches:
            if isinstance(batch, dict):
                cal_inputs.append(batch['spectrogram'])
                cal_labels.append(batch['label'])
            else:
                cal_inputs.append(batch[0])
                cal_labels.append(batch[1])
        
        cal_inputs = torch.cat(cal_inputs)
        cal_labels = torch.cat(cal_labels)
        cal_dataset = TensorDataset(cal_inputs, cal_labels)
        cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=False)
        
        # Extract test data
        test_inputs, test_labels = [], []
        for batch in test_batches:
            if isinstance(batch, dict):
                test_inputs.append(batch['spectrogram'])
                test_labels.append(batch['label'])
            else:
                test_inputs.append(batch[0])
                test_labels.append(batch[1])
        
        test_inputs = torch.cat(test_inputs)
        test_labels = torch.cat(test_labels)
        test_dataset = TensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Run QICP evaluation
        qicp_results = run_qicp_evaluation(
            self.model, cal_loader, test_loader, device=self.device, return_detailed=True
        )
        
        # Test robustness
        robustness_results = qicp_robustness_test(
            self.qicp_predictor, self.model, test_loader, device=self.device
        )
        
        results = {
            'qicp_evaluation': qicp_results,
            'robustness_analysis': robustness_results
        }
        
        # Log results
        main_results = qicp_results.get('test_results', {})
        self.logger.info(f"Conformal Prediction Results:")
        self.logger.info(f"  Coverage: {main_results.get('coverage', 0):.4f}")
        self.logger.info(f"  Average Set Size: {main_results.get('average_set_size', 0):.2f}")
        self.logger.info(f"  Coverage Gap: {main_results.get('coverage_gap', 0):.4f}")
        
        return results
    
    def test_robustness(self, noise_levels: List[float] = None) -> Dict[str, Any]:
        """Test model robustness under various conditions."""
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        self.logger.info(f"Testing robustness at noise levels: {noise_levels}")
        
        robustness_results = {}
        
        for noise_level in noise_levels:
            self.logger.info(f"Testing at noise level: {noise_level}")
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in self.test_loader:
                    if isinstance(batch, dict):
                        inputs = batch['spectrogram'].to(self.device)
                        targets = batch['label'].to(self.device)
                    else:
                        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    
                    # Add noise
                    if noise_level > 0:
                        noise = torch.randn_like(inputs) * noise_level
                        noisy_inputs = inputs + noise
                    else:
                        noisy_inputs = inputs
                    
                    # Forward pass
                    outputs = self.model(noisy_inputs, return_complex_amplitudes=False)
                    
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)
            
            accuracy = correct / total if total > 0 else 0.0
            robustness_results[f'noise_{noise_level}'] = {
                'accuracy': accuracy,
                'samples_tested': total
            }
        
        # Compute robustness metrics
        clean_accuracy = robustness_results.get('noise_0.0', {}).get('accuracy', 0.0)
        
        summary = {
            'clean_accuracy': clean_accuracy,
            'noise_degradation': {},
            'robustness_score': 0.0
        }
        
        total_degradation = 0.0
        for noise_level in noise_levels:
            if noise_level > 0:
                noisy_accuracy = robustness_results[f'noise_{noise_level}']['accuracy']
                degradation = clean_accuracy - noisy_accuracy
                summary['noise_degradation'][f'noise_{noise_level}'] = {
                    'accuracy': noisy_accuracy,
                    'degradation': degradation,
                    'relative_degradation': degradation / clean_accuracy if clean_accuracy > 0 else 0
                }
                total_degradation += degradation
        
        # Robustness score (higher is better)
        if len(noise_levels) > 1:
            summary['robustness_score'] = 1.0 - (total_degradation / ((len(noise_levels) - 1) * clean_accuracy))
        
        robustness_results['summary'] = summary
        
        self.logger.info(f"Robustness Results:")
        self.logger.info(f"  Clean Accuracy: {clean_accuracy:.4f}")
        self.logger.info(f"  Robustness Score: {summary['robustness_score']:.4f}")
        
        return robustness_results
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        self.logger.info("Generating visualizations")
        
        # Classification visualizations
        if 'classification' in self.results:
            cls_results = self.results['classification']
            
            # Confusion matrix
            if 'confusion_matrix' in cls_results:
                plt.figure(figsize=(12, 10))
                sns.heatmap(
                    cls_results['confusion_matrix'],
                    annot=True, fmt='d',
                    xticklabels=self.dataset_config.class_names,
                    yticklabels=self.dataset_config.class_names,
                    cmap='Blues'
                )
                plt.title(f'Confusion Matrix - {self.dataset_name.upper()}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Per-class performance
            if 'per_class_f1' in cls_results:
                plt.figure(figsize=(14, 8))
                class_names = self.dataset_config.class_names
                x_pos = np.arange(len(class_names))
                
                plt.bar(x_pos, cls_results['per_class_f1'], alpha=0.7)
                plt.xlabel('Class')
                plt.ylabel('F1 Score')
                plt.title(f'Per-Class F1 Scores - {self.dataset_name.upper()}')
                plt.xticks(x_pos, class_names, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.output_dir / 'per_class_f1.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Robustness visualization
        if 'robustness' in self.results:
            rob_results = self.results['robustness']
            
            # Extract noise levels and accuracies
            noise_levels = []
            accuracies = []
            
        # Robustness visualization
        if 'robustness' in self.results:
            rob_results = self.results['robustness']
            
            # Extract noise levels and accuracies
            noise_levels = []
            accuracies = []
            
            for key, value in rob_results.items():
                if key.startswith('noise_') and isinstance(value, dict):
                    noise_level = float(key.split('_')[1])
                    noise_levels.append(noise_level)
                    accuracies.append(value['accuracy'])
            
            if noise_levels:
                # Sort by noise level
                sorted_data = sorted(zip(noise_levels, accuracies))
                noise_levels, accuracies = zip(*sorted_data)
                
                plt.figure(figsize=(10, 6))
                plt.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=6)
                plt.xlabel('Noise Level')
                plt.ylabel('Accuracy')
                plt.title(f'Robustness Analysis - {self.dataset_name.upper()}')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.0)
                plt.tight_layout()
                plt.savefig(self.output_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Uncertainty visualization
        if 'uncertainty' in self.results:
            unc_results = self.results['uncertainty']
            
            if all(key in unc_results for key in ['mean_epistemic', 'mean_aleatoric', 'mean_covariance']):
                # Uncertainty decomposition pie chart
                uncertainties = [
                    unc_results['mean_epistemic'],
                    unc_results['mean_aleatoric'],
                    unc_results['mean_covariance']
                ]
                labels = ['Epistemic', 'Aleatoric', 'Covariance (Novel)']
                colors = ['#ff9999', '#66b3ff', '#99ff99']
                
                plt.figure(figsize=(8, 8))
                plt.pie(uncertainties, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title(f'Uncertainty Decomposition - {self.dataset_name.upper()}')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'uncertainty_decomposition.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        self.logger.info(f"Visualizations saved to {self.output_dir}")
    
    def save_results(self):
        """Save comprehensive evaluation results."""
        # Save detailed results
        results_file = self.output_dir / f"evaluation_results_{self.dataset_name}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report()
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _generate_summary_report(self):
        """Generate human-readable summary report."""
        report_file = self.output_dir / f"evaluation_summary_{self.dataset_name}.txt"
        
        with open(report_file, 'w') as f:
            f.write("Q-XAI Model Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {self.dataset_name.upper()}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Classification Results
            if 'classification' in self.results:
                cls_results = self.results['classification']
                f.write("CLASSIFICATION PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {cls_results.get('accuracy', 0):.4f}\n")
                f.write(f"Macro F1: {cls_results.get('macro_f1', 0):.4f}\n")
                f.write(f"Weighted F1: {cls_results.get('weighted_f1', 0):.4f}\n")
                f.write(f"Macro Precision: {cls_results.get('macro_precision', 0):.4f}\n")
                f.write(f"Macro Recall: {cls_results.get('macro_recall', 0):.4f}\n")
                f.write(f"ECE (Calibration): {cls_results.get('ece', 0):.4f}\n")
                f.write(f"Brier Score: {cls_results.get('brier_score', 0):.4f}\n\n")
            
            # Interpretability Results
            if 'interpretability' in self.results:
                int_results = self.results['interpretability']
                f.write("INTERPRETABILITY ANALYSIS (QISA)\n")
                f.write("-" * 35 + "\n")
                f.write(f"Average Faithfulness: {int_results.get('average_faithfulness', 0):.4f}\n")
                f.write(f"Average Stability: {int_results.get('average_stability', 0):.4f}\n")
                f.write(f"Average Sparsity: {int_results.get('average_sparsity', 0):.4f}\n")
                f.write(f"Samples Analyzed: {int_results.get('num_samples_analyzed', 0)}\n\n")
            
            # Uncertainty Results
            if 'uncertainty' in self.results:
                unc_results = self.results['uncertainty']
                f.write("UNCERTAINTY QUANTIFICATION (AUQ)\n")
                f.write("-" * 36 + "\n")
                if 'mean_total' in unc_results:
                    f.write(f"Mean Total Uncertainty: {unc_results['mean_total']:.4f}\n")
                    f.write(f"Mean Epistemic Uncertainty: {unc_results['mean_epistemic']:.4f}\n")
                    f.write(f"Mean Aleatoric Uncertainty: {unc_results['mean_aleatoric']:.4f}\n")
                    f.write(f"Mean Covariance Uncertainty: {unc_results['mean_covariance']:.4f} (Novel)\n")
                
                f.write(f"Samples Analyzed: {unc_results.get('num_samples_analyzed', 0)}\n\n")
            
            # Conformal Prediction Results
            if 'conformal_prediction' in self.results:
                cp_results = self.results['conformal_prediction']
                f.write("CONFORMAL PREDICTION (QICP)\n")
                f.write("-" * 30 + "\n")
                
                main_results = cp_results.get('qicp_evaluation', {}).get('test_results', {})
                f.write(f"Coverage: {main_results.get('coverage', 0):.4f}\n")
                f.write(f"Target Coverage: {main_results.get('target_coverage', 0.9):.4f}\n")
                f.write(f"Coverage Gap: {main_results.get('coverage_gap', 0):.4f}\n")
                f.write(f"Average Set Size: {main_results.get('average_set_size', 0):.2f}\n\n")
            
            # Robustness Results
            if 'robustness' in self.results:
                rob_results = self.results['robustness']
                f.write("ROBUSTNESS ANALYSIS\n")
                f.write("-" * 20 + "\n")
                
                summary = rob_results.get('summary', {})
                f.write(f"Clean Accuracy: {summary.get('clean_accuracy', 0):.4f}\n")
                f.write(f"Robustness Score: {summary.get('robustness_score', 0):.4f}\n")
                
                # Noise degradation
                degradation = summary.get('noise_degradation', {})
                if degradation:
                    f.write("\nNoise Level Performance:\n")
                    for noise_key, noise_data in degradation.items():
                        noise_level = noise_key.split('_')[1]
                        f.write(f"  Noise {noise_level}: {noise_data['accuracy']:.4f} "
                               f"(degradation: {noise_data['degradation']:.4f})\n")
                f.write("\n")
            
            f.write("=" * 50 + "\n")
            f.write("Evaluation completed successfully!\n")
    
    def run_comprehensive_evaluation(
        self,
        include_interpretability: bool = True,
        include_uncertainty: bool = True,
        include_conformal: bool = True,
        include_robustness: bool = False,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation of all Q-XAI components."""
        self.logger.info("Starting comprehensive Q-XAI evaluation")
        
        # Load model and data
        self.load_model_and_data()
        
        # Classification evaluation (always included)
        self.results['classification'] = self.evaluate_classification()
        
        # Interpretability evaluation
        if include_interpretability:
            self.results['interpretability'] = self.evaluate_interpretability(num_samples)
        
        # Uncertainty evaluation
        if include_uncertainty:
            self.results['uncertainty'] = self.evaluate_uncertainty(num_samples)
        
        # Conformal prediction evaluation
        if include_conformal:
            self.results['conformal_prediction'] = self.evaluate_conformal_prediction()
        
        # Robustness evaluation
        if include_robustness:
            self.results['robustness'] = self.test_robustness()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save results
        self.save_results()
        
        self.logger.info("Comprehensive evaluation completed")
        return self.results


def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description='Q-XAI Model Evaluation')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['tut2016', 'dcase2019', 'esc50', 'cochlscene', 'dcase2025'],
                       help='Dataset used for training')
    parser.add_argument('--data_dir', type=str, default='data/',
                       help='Directory containing the dataset')
    
    # Evaluation options
    parser.add_argument('--output_dir', type=str, default='evaluation_results/',
                       help='Directory to save evaluation results')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive evaluation of all components')
    parser.add_argument('--classification_only', action='store_true',
                       help='Run only classification evaluation')
    parser.add_argument('--interpretability', action='store_true',
                       help='Include QISA interpretability evaluation')
    parser.add_argument('--uncertainty', action='store_true',
                       help='Include AUQ uncertainty evaluation')
    parser.add_argument('--conformal', action='store_true',
                       help='Include QICP conformal prediction evaluation')
    parser.add_argument('--robustness_test', action='store_true',
                       help='Include robustness testing')
    
    # Sample sizes
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for interpretability/uncertainty analysis')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Q-XAI Model Evaluation")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print("-" * 50)
    
    # Create evaluator
    evaluator = QXAIModelEvaluator(
        model_path=args.model_path,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        device=device,
        output_dir=args.output_dir
    )
    
    try:
        if args.comprehensive:
            # Run comprehensive evaluation
            results = evaluator.run_comprehensive_evaluation(
                include_interpretability=True,
                include_uncertainty=True,
                include_conformal=True,
                include_robustness=args.robustness_test,
                num_samples=args.num_samples
            )
            
        elif args.classification_only:
            # Run only classification evaluation
            evaluator.load_model_and_data()
            evaluator.results['classification'] = evaluator.evaluate_classification()
            evaluator.generate_visualizations()
            evaluator.save_results()
            results = evaluator.results
            
        else:
            # Run selective evaluation based on flags
            evaluator.load_model_and_data()
            
            # Always include classification
            evaluator.results['classification'] = evaluator.evaluate_classification()
            
            if args.interpretability:
                evaluator.results['interpretability'] = evaluator.evaluate_interpretability(args.num_samples)
            
            if args.uncertainty:
                evaluator.results['uncertainty'] = evaluator.evaluate_uncertainty(args.num_samples)
            
            if args.conformal:
                evaluator.results['conformal_prediction'] = evaluator.evaluate_conformal_prediction()
            
            if args.robustness_test:
                evaluator.results['robustness'] = evaluator.test_robustness()
            
            evaluator.generate_visualizations()
            evaluator.save_results()
            results = evaluator.results
        
        # Print summary
        print("\nEvaluation Summary:")
        print("=" * 30)
        
        if 'classification' in results:
            cls_results = results['classification']
            print(f"Classification Accuracy: {cls_results.get('accuracy', 0):.4f}")
            print(f"Macro F1 Score: {cls_results.get('macro_f1', 0):.4f}")
            print(f"ECE (Calibration): {cls_results.get('ece', 0):.4f}")
        
        if 'interpretability' in results:
            int_results = results['interpretability']
            print(f"Average Faithfulness: {int_results.get('average_faithfulness', 0):.4f}")
            print(f"Average Sparsity: {int_results.get('average_sparsity', 0):.4f}")
        
        if 'uncertainty' in results:
            unc_results = results['uncertainty']
            if 'mean_total' in unc_results:
                print(f"Mean Total Uncertainty: {unc_results['mean_total']:.4f}")
        
        if 'conformal_prediction' in results:
            cp_results = results['conformal_prediction']
            main_results = cp_results.get('qicp_evaluation', {}).get('test_results', {})
            print(f"QICP Coverage: {main_results.get('coverage', 0):.4f}")
            print(f"QICP Set Size: {main_results.get('average_set_size', 0):.2f}")
        
        if 'robustness' in results:
            rob_results = results['robustness']
            summary = rob_results.get('summary', {})
            print(f"Robustness Score: {summary.get('robustness_score', 0):.4f}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        raise
    
    return results


def evaluate_multiple_models(
    model_paths: List[str],
    dataset_name: str,
    data_dir: str = "data/",
    output_dir: str = "comparative_evaluation/"
) -> Dict[str, Any]:
    """
    Evaluate multiple models for comparison.
    
    Args:
        model_paths: List of paths to model checkpoints
        dataset_name: Dataset name
        data_dir: Data directory
        output_dir: Output directory for results
        
    Returns:
        Comparative evaluation results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for i, model_path in enumerate(model_paths):
        model_name = f"model_{i}_{Path(model_path).stem}"
        print(f"\nEvaluating {model_name}")
        print("-" * 40)
        
        try:
            evaluator = QXAIModelEvaluator(
                model_path=model_path,
                dataset_name=dataset_name,
                data_dir=data_dir,
                output_dir=str(output_path / model_name)
            )
            
            results = evaluator.run_comprehensive_evaluation(
                include_interpretability=True,
                include_uncertainty=True,
                include_conformal=True,
                include_robustness=False,
                num_samples=50  # Reduced for faster comparison
            )
            
            all_results[model_name] = results
            
        except Exception as e:
            print(f"Failed to evaluate {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Generate comparative report
    _generate_comparative_report(all_results, output_path)
    
    return all_results


def _generate_comparative_report(results: Dict[str, Any], output_path: Path):
    """Generate comparative analysis report."""
    report_file = output_path / "comparative_analysis.txt"
    
    with open(report_file, 'w') as f:
        f.write("Q-XAI Models Comparative Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        # Create comparison table
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':<10} {'F1':<8} {'ECE':<8} {'Coverage':<10}\n")
        f.write("-" * 56 + "\n")
        
        for model_name, model_results in results.items():
            if 'error' not in model_results:
                cls_results = model_results.get('classification', {})
                cp_results = model_results.get('conformal_prediction', {})
                
                accuracy = cls_results.get('accuracy', 0)
                f1 = cls_results.get('macro_f1', 0)
                ece = cls_results.get('ece', 0)
                
                main_cp = cp_results.get('qicp_evaluation', {}).get('test_results', {})
                coverage = main_cp.get('coverage', 0)
                
                f.write(f"{model_name:<20} {accuracy:<10.4f} {f1:<8.4f} {ece:<8.4f} {coverage:<10.4f}\n")
            else:
                f.write(f"{model_name:<20} {'ERROR':<10} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10}\n")
    
    print(f"Comparative analysis saved to {report_file}")


if __name__ == '__main__':
    main()