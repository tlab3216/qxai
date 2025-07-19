"""
Training pipeline for Q-XAI framework.
Implements complete training loop with complex-valued backpropagation, validation, 
checkpointing, and integration with QISA, AUQ, and QICP components.
"""

import os
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Q-XAI Framework imports
from models.complex_transformer import ComplexTransformerClassifier, create_complex_transformer
from interpretability.qisa import QISAAttributor, create_qisa_attributor
from interpretability.auq import AUQEstimator, UncertaintyComponents
from interpretability.qicp import QICPPredictor, create_qicp_predictor
from config.model_config import ComplexTransformerConfig, TrainingConfig, QXAIConfig
from utils.complex_math import WirtingerGradient
from utils.metrics import ClassificationMetrics, UncertaintyMetrics, InterpretabilityMetrics
from data.datasets import BaseASCDataset


@dataclass
class TrainingState:
    """Encapsulates training state for checkpointing."""
    epoch: int
    step: int
    best_val_acc: float
    best_val_loss: float
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    learning_rate: float
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
    config: Dict[str, Any]


class EarlyStopping:
    """Early stopping implementation with patience and delta threshold."""
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        monitor: str = "val_accuracy",
        mode: str = "max",
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        
    def __call__(self, current_value: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        improved = False
        
        if self.mode == 'max':
            if current_value > self.best_value + self.min_delta:
                improved = True
        else:
            if current_value < self.best_value - self.min_delta:
                improved = True
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.stopped_epoch = True
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class QXAITrainer:
    """
    Main trainer class for Q-XAI framework.
    Handles training, validation, interpretability analysis, and uncertainty quantification.
    """
    
    def __init__(
        self,
        config: QXAIConfig,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        experiment_name: str = "q_xai_experiment"
    ):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = experiment_name
        
        # Create model if not provided
        if model is None:
            self.model = create_complex_transformer(
                config.model, 
                model_type="standard",
                num_classes=config.model.num_classes
            )
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None  # For mixed precision
        
        # Q-XAI components
        self.qisa_attributor = None
        self.auq_estimator = None
        self.qicp_predictor = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
        # Early stopping
        self.early_stopping = None
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.patience,
                min_delta=config.training.min_delta,
                monitor=config.training.monitor_metric,
                mode=config.training.monitor_mode
            )
        
        # Logging
        self.writer = None
        self.log_dir = None
        self.checkpoint_dir = None
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        self.interpretability_metrics = []
        
        self._setup_logging()
        self._setup_training_components()
        self._setup_qxai_components()
    
    def _setup_logging(self):
        """Setup logging and checkpoint directories."""
        # Create experiment directory
        exp_dir = Path(f"experiments/{self.experiment_name}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard logging
        self.log_dir = exp_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Setup checkpoint directory
        self.checkpoint_dir = exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Save config
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def _setup_training_components(self):
        """Setup optimizer, scheduler, and loss function."""
        # Optimizer
        if self.config.training.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.beta1, self.config.training.beta2),
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.beta1, self.config.training.beta2),
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        # Scheduler
        if self.config.training.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.cosine_eta_min
            )
        elif self.config.training.scheduler.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.epochs // 3,
                gamma=0.1
            )
        elif self.config.training.scheduler.lower() == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        elif self.config.training.scheduler.lower() == "none":
            self.scheduler = None
        
        # Add warmup if specified
        if self.config.training.warmup_epochs > 0 and self.scheduler:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=self.config.training.warmup_start_lr / self.config.training.learning_rate,
                end_factor=1.0,
                total_iters=self.config.training.warmup_epochs
            )
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, self.scheduler],
                milestones=[self.config.training.warmup_epochs]
            )
        
        # Loss function
        if self.config.training.loss_function.lower() == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.training.label_smoothing
            )
        elif self.config.training.loss_function.lower() == "focal_loss":
            from training.losses import FocalLoss
            self.criterion = FocalLoss(
                alpha=1.0,
                gamma=2.0,
                label_smoothing=self.config.training.label_smoothing
            )
        else:
            raise ValueError(f"Unknown loss function: {self.config.training.loss_function}")
        
        # Mixed precision scaler
        if self.config.training.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _setup_qxai_components(self):
        """Setup Q-XAI interpretability and uncertainty components."""
        # QISA Attributor
        self.qisa_attributor = create_qisa_attributor(
            self.model,
            self.config.qisa,
            method="standard"
        )
        
        # AUQ Estimator
        self.auq_estimator = AUQEstimator(
            self.model,
            self.config.auq,
            device=self.device
        )
        
        # QICP Predictor (will be calibrated later)
        self.qicp_predictor = create_qicp_predictor(
            self.config.qicp,
            score_type="born_rule"
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        batch_times = []
        
        # Progress tracking
        num_batches = len(train_loader)
        log_interval = max(1, num_batches // 10)  # Log 10 times per epoch
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Move data to device
            if isinstance(batch, dict):
                inputs = batch['spectrogram'].to(self.device)
                targets = batch['label'].to(self.device)
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            else:
                raise ValueError("Unsupported batch format")
            
            batch_size = inputs.size(0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.config.training.use_amp and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs, return_complex_amplitudes=False)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.clip_grad_norm
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(inputs, return_complex_amplitudes=False)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.clip_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            
            # Track batch time
            batch_times.append(time.time() - batch_start_time)
            
            # Global step increment
            self.global_step += 1
            
            # Logging
            if batch_idx % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_batch_time = np.mean(batch_times[-10:])  # Last 10 batches
                
                print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches} '
                      f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}, '
                      f'Time: {avg_batch_time:.3f}s/batch')
                
                # Tensorboard logging
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
                self.writer.add_scalar('Train/BatchTime', avg_batch_time, self.global_step)
        
        # Epoch metrics
        epoch_loss = total_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'avg_batch_time': np.mean(batch_times)
        }
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int,
        run_interpretability: bool = False
    ) -> Dict[str, float]:
        """Validate for one epoch with optional interpretability analysis."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        # Interpretability metrics
        interpretability_results = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move data to device
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(self.device)
                    targets = batch['label'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                
                batch_size = inputs.size(0)
                
                # Forward pass
                if self.config.training.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs, return_complex_amplitudes=True)
                else:
                    outputs = self.model(inputs, return_complex_amplitudes=True)
                
                if isinstance(outputs, tuple):
                    logits, complex_amplitudes, _ = outputs
                else:
                    logits = outputs
                    complex_amplitudes = None
                
                # Compute loss
                loss = self.criterion(logits, targets)
                
                # Accumulate metrics
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                correct_predictions += (predicted == targets).sum().item()
                
                # Store for comprehensive metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Run interpretability analysis on a subset
                if run_interpretability and batch_idx < 5:  # Only first few batches
                    try:
                        # QISA attribution for first sample in batch
                        sample_input = inputs[0:1]
                        sample_target = targets[0:1]
                        
                        attribution = self.qisa_attributor.attribute(
                            sample_input, sample_target[0].item()
                        )
                        
                        # Store attribution statistics
                        if 'qisa_sparsity' not in interpretability_results:
                            interpretability_results['qisa_sparsity'] = []
                        
                        sparsity = InterpretabilityMetrics.compute_sparsity(attribution)
                        interpretability_results['qisa_sparsity'].append(sparsity)
                        
                    except Exception as e:
                        print(f"Interpretability analysis failed: {e}")
        
        # Epoch metrics
        epoch_loss = total_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        
        # Comprehensive classification metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # F1 scores
        macro_f1 = ClassificationMetrics.compute_f1_scores(
            torch.tensor(all_predictions),
            torch.tensor(all_targets),
            self.config.model.num_classes,
            average='macro'
        )
        
        # Uncertainty calibration
        ece = UncertaintyMetrics.compute_ece(
            torch.tensor(all_probabilities),
            torch.tensor(all_predictions),
            torch.tensor(all_targets)
        )
        
        validation_metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'macro_f1': macro_f1,
            'ece': ece
        }
        
        # Add interpretability metrics
        if interpretability_results:
            validation_metrics['avg_qisa_sparsity'] = np.mean(
                interpretability_results['qisa_sparsity']
            )
        
        return validation_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if None)
            
        Returns:
            Training history and final metrics
        """
        if num_epochs is None:
            num_epochs = self.config.training.epochs
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        
        training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_macro_f1': [],
            'val_ece': [],
            'learning_rates': []
        }
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            run_interpretability = (epoch % 10 == 0)  # Every 10 epochs
            val_metrics = self.validate_epoch(val_loader, epoch, run_interpretability)
            
            # Learning rate step
            if self.scheduler:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_accuracy'].append(train_metrics['accuracy'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            training_history['val_macro_f1'].append(val_metrics['macro_f1'])
            training_history['val_ece'].append(val_metrics['ece'])
            training_history['learning_rates'].append(current_lr)
            
            # Tensorboard logging
            self.writer.add_scalar('Train/EpochLoss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/EpochAccuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/MacroF1', val_metrics['macro_f1'], epoch)
            self.writer.add_scalar('Val/ECE', val_metrics['ece'], epoch)
            
            # Update best metrics
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, is_best=True)
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping check
            if self.early_stopping:
                monitor_value = val_metrics.get(
                    self.config.training.monitor_metric.replace('val_', ''),
                    val_metrics['accuracy']
                )
                
                if self.early_stopping(monitor_value, self.model):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.1f}s')
            print(f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.4f}')
            print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
            print(f'Val F1: {val_metrics["macro_f1"]:.4f}, Val ECE: {val_metrics["ece"]:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            print('-' * 60)
        
        total_time = time.time() - start_time
        print(f'Training completed in {total_time:.1f}s')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        
        # Final checkpoint
        self.save_checkpoint(epoch, is_best=False, final=True)
        
        return {
            'history': training_history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'total_training_time': total_time,
            'final_epoch': epoch + 1
        }
    
    def calibrate_qicp(
        self,
        calibration_loader: DataLoader
    ) -> Dict[str, float]:
        """Calibrate QICP predictor after training."""
        print("Calibrating QICP predictor...")
        
        calibration_stats = self.qicp_predictor.calibrate(
            self.model, calibration_loader, self.device
        )
        
        print(f"QICP calibration complete:")
        print(f"  Threshold: {calibration_stats['threshold']:.4f}")
        print(f"  Calibration samples: {calibration_stats['num_calibration_samples']}")
        
        return calibration_stats
    
    def evaluate_comprehensive(
        self,
        test_loader: DataLoader,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation including interpretability and uncertainty.
        
        Args:
            test_loader: Test data loader
            save_results: Whether to save detailed results
            
        Returns:
            Comprehensive evaluation results
        """
        print("Running comprehensive evaluation...")
        
        self.model.eval()
        evaluation_results = {
            'classification': {},
            'interpretability': {},
            'uncertainty': {},
            'conformal_prediction': {}
        }
        
        # Classification evaluation
        with torch.no_grad():
            all_predictions = []
            all_targets = []
            all_probabilities = []
            all_complex_amplitudes = []
            
            for batch in test_loader:
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(self.device)
                    targets = batch['label'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                
                outputs = self.model(inputs, return_complex_amplitudes=True)
                
                if isinstance(outputs, tuple):
                    logits, complex_amplitudes, _ = outputs
                else:
                    logits = outputs
                    complex_amplitudes = logits
                
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                if complex_amplitudes is not None:
                    all_complex_amplitudes.extend(complex_amplitudes.cpu().numpy())
        
        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Classification metrics
        accuracy = ClassificationMetrics.compute_accuracy(
            torch.tensor(all_predictions), torch.tensor(all_targets)
        )
        
        macro_f1 = ClassificationMetrics.compute_f1_scores(
            torch.tensor(all_predictions), torch.tensor(all_targets),
            self.config.model.num_classes, average='macro'
        )
        
        confusion_matrix = ClassificationMetrics.compute_confusion_matrix(
            torch.tensor(all_predictions), torch.tensor(all_targets),
            self.config.model.num_classes
        )
        
        evaluation_results['classification'] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'confusion_matrix': confusion_matrix.tolist()
        }
        
        # Uncertainty evaluation
        ece = UncertaintyMetrics.compute_ece(
            torch.tensor(all_probabilities),
            torch.tensor(all_predictions),
            torch.tensor(all_targets)
        )
        
        brier_score = UncertaintyMetrics.compute_brier_score(
            torch.tensor(all_probabilities),
            torch.tensor(all_targets)
        )
        
        evaluation_results['uncertainty'] = {
            'ece': ece,
            'brier_score': brier_score
        }
        
        # QICP evaluation if calibrated
        if self.qicp_predictor.is_calibrated:
            qicp_result = self.qicp_predictor.predict(
                self.model, test_loader, self.device, return_scores=True
            )
            
            evaluation_results['conformal_prediction'] = {
                'coverage': qicp_result.coverage,
                'average_set_size': qicp_result.average_set_size,
                'target_coverage': self.config.qicp.confidence_level,
                'coverage_gap': abs(qicp_result.coverage - self.config.qicp.confidence_level)
            }
        
        # Sample interpretability analysis
        print("Running interpretability analysis on sample data...")
        sample_batch = next(iter(test_loader))
        
        if isinstance(sample_batch, dict):
            sample_inputs = sample_batch['spectrogram'][:5].to(self.device)
            sample_targets = sample_batch['label'][:5].to(self.device)
        else:
            sample_inputs = sample_batch[0][:5].to(self.device)
            sample_targets = sample_batch[1][:5].to(self.device)
        
        # QISA attribution for samples
        sample_attributions = []
        for i in range(len(sample_inputs)):
            attribution = self.qisa_attributor.attribute(
                sample_inputs[i:i+1], sample_targets[i].item()
            )
            sample_attributions.append(attribution.cpu().numpy())
        
        # Compute average sparsity
        avg_sparsity = np.mean([
            InterpretabilityMetrics.compute_sparsity(torch.tensor(attr))
            for attr in sample_attributions
        ])
        
        evaluation_results['interpretability'] = {
            'average_attribution_sparsity': avg_sparsity,
            'num_samples_analyzed': len(sample_attributions)
        }
        
        # Save results if requested
        if save_results:
            results_path = self.checkpoint_dir / "evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            print(f"Evaluation results saved to {results_path}")
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"  Classification Accuracy: {accuracy:.4f}")
        print(f"  Macro F1 Score: {macro_f1:.4f}")
        print(f"  Expected Calibration Error: {ece:.4f}")
        print(f"  Brier Score: {brier_score:.4f}")
        if self.qicp_predictor.is_calibrated:
            print(f"  QICP Coverage: {qicp_result.coverage:.4f} (target: {self.config.qicp.confidence_level:.4f})")
            print(f"  Average Set Size: {qicp_result.average_set_size:.2f}")
        print(f"  Average Attribution Sparsity: {avg_sparsity:.4f}")
        
        return evaluation_results
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        final: bool = False
    ):
        """Save model checkpoint."""
        checkpoint_data = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'qicp_calibrated': self.qicp_predictor.is_calibrated if self.qicp_predictor else False
        }
        
        # Add QICP calibration data if available
        if self.qicp_predictor and self.qicp_predictor.is_calibrated:
            checkpoint_data['qicp_threshold'] = self.qicp_predictor.threshold
            checkpoint_data['qicp_calibration_scores'] = self.qicp_predictor.calibration_scores.tolist()
        
        # Save different types of checkpoints
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Best model checkpoint saved to {checkpoint_path}")
        
        if final:
            checkpoint_path = self.checkpoint_dir / "final_model.pt"
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Final model checkpoint saved to {checkpoint_path}")
        
        # Regular epoch checkpoint
        if not is_best and not final:
            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(checkpoint_data, checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if load_scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load QICP calibration if available
        if checkpoint.get('qicp_calibrated', False) and self.qicp_predictor:
            self.qicp_predictor.threshold = checkpoint.get('qicp_threshold')
            if 'qicp_calibration_scores' in checkpoint:
                self.qicp_predictor.calibration_scores = np.array(
                    checkpoint['qicp_calibration_scores']
                )
            self.qicp_predictor.is_calibrated = True
        
        print(f"Checkpoint loaded successfully:")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best validation accuracy: {self.best_val_acc:.4f}")
        
        return checkpoint
    
    def run_interpretability_analysis(
        self,
        data_loader: DataLoader,
        num_samples: int = 50,
        save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive interpretability analysis.
        
        Args:
            data_loader: Data loader for analysis
            num_samples: Number of samples to analyze
            save_visualizations: Whether to save attribution visualizations
            
        Returns:
            Interpretability analysis results
        """
        print(f"Running interpretability analysis on {num_samples} samples...")
        
        self.model.eval()
        attribution_metrics = {
            'sparsity_scores': [],
            'attribution_magnitudes': [],
            'sample_analyses': []
        }
        
        samples_processed = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if samples_processed >= num_samples:
                    break
                
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(self.device)
                    targets = batch['label'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                
                batch_size = min(inputs.size(0), num_samples - samples_processed)
                
                for i in range(batch_size):
                    sample_input = inputs[i:i+1]
                    sample_target = targets[i].item()
                    
                    try:
                        # QISA attribution
                        attribution = self.qisa_attributor.attribute(
                            sample_input, sample_target
                        )
                        
                        # Compute metrics
                        sparsity = InterpretabilityMetrics.compute_sparsity(attribution)
                        magnitude = torch.mean(torch.abs(attribution)).item()
                        
                        attribution_metrics['sparsity_scores'].append(sparsity)
                        attribution_metrics['attribution_magnitudes'].append(magnitude)
                        
                        # Store sample analysis
                        sample_analysis = {
                            'sample_idx': samples_processed,
                            'true_label': sample_target,
                            'sparsity': sparsity,
                            'magnitude': magnitude
                        }
                        
                        # Get model prediction for this sample
                        outputs = self.model(sample_input, return_complex_amplitudes=True)
                        if isinstance(outputs, tuple):
                            logits, complex_amplitudes, _ = outputs
                        else:
                            logits = outputs
                        
                        probabilities = torch.softmax(logits, dim=1)
                        predicted_class = torch.argmax(logits, dim=1).item()
                        confidence = torch.max(probabilities, dim=1)[0].item()
                        
                        sample_analysis.update({
                            'predicted_label': predicted_class,
                            'confidence': confidence,
                            'correct_prediction': predicted_class == sample_target
                        })
                        
                        attribution_metrics['sample_analyses'].append(sample_analysis)
                        
                        # Save visualization if requested
                        if save_visualizations and samples_processed < 10:
                            try:
                                from interpretability.qisa import QISAVisualizer
                                
                                visualizer = QISAVisualizer(self.qisa_attributor)
                                save_path = self.checkpoint_dir / f"attribution_sample_{samples_processed}.png"
                                
                                fig = visualizer.visualize_attribution(
                                    sample_input, attribution, target=sample_target,
                                    save_path=str(save_path)
                                )
                                
                                if fig:
                                    import matplotlib.pyplot as plt
                                    plt.close(fig)
                                
                            except Exception as e:
                                print(f"Failed to save visualization for sample {samples_processed}: {e}")
                        
                        samples_processed += 1
                        
                    except Exception as e:
                        print(f"Failed to analyze sample {samples_processed}: {e}")
                        continue
        
        # Compute summary statistics
        summary_stats = {
            'mean_sparsity': np.mean(attribution_metrics['sparsity_scores']),
            'std_sparsity': np.std(attribution_metrics['sparsity_scores']),
            'mean_magnitude': np.mean(attribution_metrics['attribution_magnitudes']),
            'std_magnitude': np.std(attribution_metrics['attribution_magnitudes']),
            'samples_analyzed': samples_processed
        }
        
        # Compute correlation between attribution properties and prediction confidence
        confidences = [analysis['confidence'] for analysis in attribution_metrics['sample_analyses']]
        sparsities = attribution_metrics['sparsity_scores']
        
        if len(confidences) > 1:
            correlation_coef = np.corrcoef(confidences, sparsities)[0, 1]
            summary_stats['confidence_sparsity_correlation'] = correlation_coef
        
        attribution_metrics['summary'] = summary_stats
        
        # Save detailed results
        results_path = self.checkpoint_dir / "interpretability_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(attribution_metrics, f, indent=2, default=str)
        
        print(f"Interpretability analysis complete:")
        print(f"  Samples analyzed: {samples_processed}")
        print(f"  Mean sparsity: {summary_stats['mean_sparsity']:.4f}")
        print(f"  Mean magnitude: {summary_stats['mean_magnitude']:.4f}")
        if 'confidence_sparsity_correlation' in summary_stats:
            print(f"  Confidence-Sparsity correlation: {summary_stats['confidence_sparsity_correlation']:.4f}")
        
        return attribution_metrics
    
    def run_uncertainty_analysis(
        self,
        data_loader: DataLoader,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Run comprehensive uncertainty analysis using AUQ.
        
        Args:
            data_loader: Data loader for analysis
            num_samples: Number of samples to analyze
            
        Returns:
            Uncertainty analysis results
        """
        print(f"Running uncertainty analysis on {num_samples} samples...")
        
        self.model.eval()
        uncertainty_results = {
            'epistemic_uncertainties': [],
            'aleatoric_uncertainties': [],
            'covariance_uncertainties': [],
            'total_uncertainties': [],
            'prediction_confidences': [],
            'correct_predictions': []
        }
        
        samples_processed = 0
        
        for batch_idx, batch in enumerate(data_loader):
            if samples_processed >= num_samples:
                break
            
            if isinstance(batch, dict):
                inputs = batch['spectrogram'].to(self.device)
                targets = batch['label'].to(self.device)
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            
            batch_size = min(inputs.size(0), num_samples - samples_processed)
            
            for i in range(batch_size):
                sample_input = inputs[i:i+1]
                sample_target = targets[i].item()
                
                try:
                    # AUQ uncertainty estimation
                    uncertainty_components = self.auq_estimator.estimate_uncertainty(
                        sample_input, num_samples=self.config.auq.num_mc_samples
                    )
                    
                    # Extract uncertainty values for this sample
                    epistemic = torch.mean(uncertainty_components.epistemic[0]).item()
                    aleatoric = torch.mean(uncertainty_components.aleatoric[0]).item()
                    covariance = torch.mean(uncertainty_components.covariance[0]).item()
                    total = torch.mean(uncertainty_components.total[0]).item()
                    
                    uncertainty_results['epistemic_uncertainties'].append(epistemic)
                    uncertainty_results['aleatoric_uncertainties'].append(aleatoric)
                    uncertainty_results['covariance_uncertainties'].append(covariance)
                    uncertainty_results['total_uncertainties'].append(total)
                    
                    # Get prediction confidence
                    with torch.no_grad():
                        outputs = self.model(sample_input, return_complex_amplitudes=True)
                        if isinstance(outputs, tuple):
                            logits, _, _ = outputs
                        else:
                            logits = outputs
                        
                        probabilities = torch.softmax(logits, dim=1)
                        predicted_class = torch.argmax(logits, dim=1).item()
                        confidence = torch.max(probabilities, dim=1)[0].item()
                        
                        uncertainty_results['prediction_confidences'].append(confidence)
                        uncertainty_results['correct_predictions'].append(
                            predicted_class == sample_target
                        )
                    
                    samples_processed += 1
                    
                except Exception as e:
                    print(f"Failed to analyze uncertainty for sample {samples_processed}: {e}")
                    continue
        
        # Compute summary statistics and correlations
        summary_stats = {
            'mean_epistemic': np.mean(uncertainty_results['epistemic_uncertainties']),
            'mean_aleatoric': np.mean(uncertainty_results['aleatoric_uncertainties']),
            'mean_covariance': np.mean(uncertainty_results['covariance_uncertainties']),
            'mean_total': np.mean(uncertainty_results['total_uncertainties']),
            'std_epistemic': np.std(uncertainty_results['epistemic_uncertainties']),
            'std_aleatoric': np.std(uncertainty_results['aleatoric_uncertainties']),
            'std_covariance': np.std(uncertainty_results['covariance_uncertainties']),
            'std_total': np.std(uncertainty_results['total_uncertainties']),
            'samples_analyzed': samples_processed
        }
        
        # Correlation analysis
        if len(uncertainty_results['prediction_confidences']) > 1:
            # Uncertainty vs confidence correlation (should be negative)
            total_uncertainties = uncertainty_results['total_uncertainties']
            confidences = uncertainty_results['prediction_confidences']
            
            uncertainty_confidence_corr = np.corrcoef(total_uncertainties, confidences)[0, 1]
            summary_stats['uncertainty_confidence_correlation'] = uncertainty_confidence_corr
            
            # Uncertainty vs correctness correlation
            correctness = [float(correct) for correct in uncertainty_results['correct_predictions']]
            uncertainty_correctness_corr = np.corrcoef(total_uncertainties, correctness)[0, 1]
            summary_stats['uncertainty_correctness_correlation'] = uncertainty_correctness_corr
        
        # Relative importance of uncertainty components
        total_uncertainty_sum = (
            summary_stats['mean_epistemic'] + 
            summary_stats['mean_aleatoric'] + 
            summary_stats['mean_covariance']
        )
        
        if total_uncertainty_sum > 0:
            summary_stats['epistemic_fraction'] = summary_stats['mean_epistemic'] / total_uncertainty_sum
            summary_stats['aleatoric_fraction'] = summary_stats['mean_aleatoric'] / total_uncertainty_sum
            summary_stats['covariance_fraction'] = summary_stats['mean_covariance'] / total_uncertainty_sum
        
        uncertainty_results['summary'] = summary_stats
        
        # Save detailed results
        results_path = self.checkpoint_dir / "uncertainty_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(uncertainty_results, f, indent=2, default=str)
        
        print(f"Uncertainty analysis complete:")
        print(f"  Samples analyzed: {samples_processed}")
        print(f"  Mean epistemic uncertainty: {summary_stats['mean_epistemic']:.4f}")
        print(f"  Mean aleatoric uncertainty: {summary_stats['mean_aleatoric']:.4f}")
        print(f"  Mean covariance uncertainty: {summary_stats['mean_covariance']:.4f}")
        print(f"  Mean total uncertainty: {summary_stats['mean_total']:.4f}")
        if 'uncertainty_confidence_correlation' in summary_stats:
            print(f"  Uncertainty-Confidence correlation: {summary_stats['uncertainty_confidence_correlation']:.4f}")
        
        return uncertainty_results
    
    def test_robustness(
        self,
        test_loader: DataLoader,
        noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3],
        num_samples_per_level: int = 100
    ) -> Dict[str, Any]:
        """
        Test model robustness under different noise levels.
        
        Args:
            test_loader: Test data loader
            noise_levels: List of noise standard deviations
            num_samples_per_level: Number of samples to test per noise level
            
        Returns:
            Robustness test results
        """
        print("Testing model robustness under noise...")
        
        self.model.eval()
        robustness_results = {}
        
        for noise_level in noise_levels:
            print(f"Testing noise level: {noise_level}")
            
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, batch in enumerate(test_loader):
                if total_samples >= num_samples_per_level:
                    break
                
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(self.device)
                    targets = batch['label'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                
                # Add noise
                if noise_level > 0:
                    noise = torch.randn_like(inputs) * noise_level
                    noisy_inputs = inputs + noise
                else:
                    noisy_inputs = inputs
                
                # Limit batch size to remaining samples needed
                remaining_samples = num_samples_per_level - total_samples
                batch_size = min(inputs.size(0), remaining_samples)
                
                with torch.no_grad():
                    outputs = self.model(noisy_inputs[:batch_size], return_complex_amplitudes=False)
                    _, predicted = torch.max(outputs, 1)
                    
                    correct_predictions += (predicted == targets[:batch_size]).sum().item()
                    total_samples += batch_size
            
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            robustness_results[f'noise_{noise_level}'] = {
                'accuracy': accuracy,
                'samples_tested': total_samples
            }
        
        # Compute robustness metrics
        clean_accuracy = robustness_results.get('noise_0.0', {}).get('accuracy', 0.0)
        robustness_summary = {
            'clean_accuracy': clean_accuracy,
            'robustness_scores': {}
        }
        
        for noise_level in noise_levels:
            if noise_level > 0:
                noisy_accuracy = robustness_results[f'noise_{noise_level}']['accuracy']
                robustness_drop = clean_accuracy - noisy_accuracy
                robustness_summary['robustness_scores'][f'noise_{noise_level}'] = {
                    'accuracy': noisy_accuracy,
                    'accuracy_drop': robustness_drop,
                    'relative_drop': robustness_drop / clean_accuracy if clean_accuracy > 0 else 0.0
                }
        
        robustness_results['summary'] = robustness_summary
        
        # Save results
        results_path = self.checkpoint_dir / "robustness_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(robustness_results, f, indent=2, default=str)
        
        print("Robustness analysis complete:")
        print(f"  Clean accuracy: {clean_accuracy:.4f}")
        for noise_level in noise_levels:
            if noise_level > 0:
                score = robustness_summary['robustness_scores'][f'noise_{noise_level}']
                print(f"  Noise {noise_level}: {score['accuracy']:.4f} (drop: {score['accuracy_drop']:.4f})")
        
        return robustness_results
    
    def close(self):
        """Clean up resources."""
        if self.writer:
            self.writer.close()


class DistributedQXAITrainer(QXAITrainer):
    """
    Distributed training implementation for Q-XAI framework.
    Supports multi-GPU training with proper synchronization.
    """
    
    def __init__(
        self,
        config: QXAIConfig,
        local_rank: int = 0,
        world_size: int = 1,
        **kwargs
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        
        # Initialize distributed training
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
        
        super().__init__(config, device=torch.device(f'cuda:{local_rank}'), **kwargs)
        
        # Wrap model for distributed training
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank])
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False):
        """Save checkpoint only from rank 0."""
        if self.local_rank == 0:
            # Extract model state dict from DDP wrapper if needed
            model_state_dict = (
                self.model.module.state_dict() 
                if hasattr(self.model, 'module') 
                else self.model.state_dict()
            )
            
            checkpoint_data = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_acc': self.best_val_acc,
                'best_val_loss': self.best_val_loss,
                'config': asdict(self.config)
            }
            
            if is_best:
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint_data, checkpoint_path)
            
            if final:
                checkpoint_path = self.checkpoint_dir / "final_model.pt"
                torch.save(checkpoint_data, checkpoint_path)


# Utility functions

def create_trainer(
    config: QXAIConfig,
    model: Optional[nn.Module] = None,
    distributed: bool = False,
    local_rank: int = 0,
    world_size: int = 1,
    **kwargs
) -> QXAITrainer:
    """
    Factory function to create appropriate trainer.
    
    Args:
        config: Q-XAI configuration
        model: Optional pre-created model
        distributed: Whether to use distributed training
        local_rank: Local rank for distributed training
        world_size: World size for distributed training
        **kwargs: Additional trainer arguments
        
    Returns:
        Trainer instance
    """
    if distributed and world_size > 1:
        return DistributedQXAITrainer(
            config, 
            model=model,
            local_rank=local_rank,
            world_size=world_size,
            **kwargs
        )
    else:
        return QXAITrainer(config, model=model, **kwargs)


def run_full_training_pipeline(
    config: QXAIConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    calibration_loader: Optional[DataLoader] = None,
    experiment_name: str = "q_xai_full_pipeline",
    save_detailed_analysis: bool = True
) -> Dict[str, Any]:
    """
    Run complete Q-XAI training and evaluation pipeline.
    
    Args:
        config: Q-XAI configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        calibration_loader: Optional calibration data for QICP
        experiment_name: Name for the experiment
        save_detailed_analysis: Whether to run detailed analysis
        
    Returns:
        Complete pipeline results
    """
    print(f"Starting Q-XAI full training pipeline: {experiment_name}")
    
    # Create trainer
    trainer = QXAITrainer(config, experiment_name=experiment_name)
    
    # Training phase
    print("Phase 1: Model Training")
    training_results = trainer.train(train_loader, val_loader)
    
    # QICP Calibration phase
    if calibration_loader is None:
        print("Using validation loader for QICP calibration")
        calibration_loader = val_loader
    
    print("Phase 2: QICP Calibration")
    calibration_stats = trainer.calibrate_qicp(calibration_loader)
    
    # Comprehensive evaluation phase
    print("Phase 3: Comprehensive Evaluation")
    evaluation_results = trainer.evaluate_comprehensive(test_loader, save_results=True)
    
    # Additional analyses if requested
    detailed_analyses = {}
    if save_detailed_analysis:
        print("Phase 4: Detailed Analysis")
        
        # Interpretability analysis
        interpretability_results = trainer.run_interpretability_analysis(
            test_loader, num_samples=50, save_visualizations=True
        )
        detailed_analyses['interpretability'] = interpretability_results
        
        # Uncertainty analysis
        uncertainty_results = trainer.run_uncertainty_analysis(
            test_loader, num_samples=100
        )
        detailed_analyses['uncertainty'] = uncertainty_results
        
        # Robustness analysis
        robustness_results = trainer.test_robustness(
            test_loader, noise_levels=[0.0, 0.1, 0.2, 0.3], num_samples_per_level=100
        )
        detailed_analyses['robustness'] = robustness_results
    
    # Compile final results
    pipeline_results = {
        'training': training_results,
        'calibration': calibration_stats,
        'evaluation': evaluation_results,
        'detailed_analyses': detailed_analyses,
        'experiment_name': experiment_name,
        'config': asdict(config)
    }
    
    # Save complete results
    results_path = trainer.checkpoint_dir / "complete_pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(pipeline_results, f, indent=2, default=str)
    
    print(f"Complete Q-XAI pipeline finished!")
    print(f"Results saved to: {trainer.checkpoint_dir}")
    print(f"Best validation accuracy: {training_results['best_val_acc']:.4f}")
    print(f"Final test accuracy: {evaluation_results['classification']['accuracy']:.4f}")
    
    # Cleanup
    trainer.close()
    
    return pipeline_results