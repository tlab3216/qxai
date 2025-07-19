#!/usr/bin/env python3
"""
Q-XAI: Interpretable Complex-Valued Transformers for Acoustic Scene Classification
Main Training and Evaluation Script

This script implements the complete Q-XAI framework training pipeline including:
1. Complex-valued transformer training
2. QISA attribution computation
3. AUQ uncertainty quantification  
4. QICP conformal prediction calibration

Usage:
    python train_q_xai.py --dataset dcase2019 --epochs 100
    python train_q_xai.py --dataset esc50 --epochs 100 --batch_size 32
    python train_q_xai.py --dataset tut2016 --use_full_pipeline --use_wandb
"""

import argparse
import logging
import os
import random
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
import wandb

# Import Q-XAI components from existing structure
from models.complex_transformer import ComplexTransformerClassifier, create_complex_transformer
from interpretability.qisa import QISAAttributor, create_qisa_attributor
from interpretability.auq import AUQEstimator, estimate_uncertainty_with_auq
from interpretability.qicp import QICPPredictor, create_qicp_predictor
from data.datasets import (
    TUT2016Dataset, DCASE2019Dataset, ESC50Dataset, 
    CochlSceneDataset, DCASE2025Dataset, create_dataloader
)
from data.preprocessing import create_preprocessing_pipeline
from training.trainer import QXAITrainer, create_trainer, run_full_training_pipeline
from training.losses import create_loss_function, ComplexCrossEntropyLoss
from utils.metrics import ClassificationMetrics, UncertaintyMetrics, InterpretabilityMetrics
from utils.visualization import plot_attribution_maps, plot_uncertainty_decomposition
from config.model_config import ComplexTransformerConfig, QXAIConfig, TrainingConfig
from config.data_config import (
    DatasetConfig, AudioConfig, SpectrogramConfig, AugmentationConfig,
    TUT2016Config, DCASE2019Config, ESC50Config, CochlSceneConfig, DCASE2025Config
)


def get_dataset_and_config(dataset_name: str, data_dir: str) -> Tuple[Any, DatasetConfig, AudioConfig, SpectrogramConfig]:
    """Get dataset class and configuration based on dataset name."""
    
    # Audio configuration (common across datasets)
    audio_config = AudioConfig(
        sample_rate=16000,
        mono=True,
        normalize=True,
        norm_type='peak'
    )
    
    # Spectrogram configuration (common across datasets) 
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
    
    if dataset_name == 'tut2016':
        dataset_config = TUT2016Config(
            root_path=data_dir,
            class_names=[
                'beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
                'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
                'office', 'park', 'residential_area', 'train', 'tram'
            ],
            fold_based_split=True
        )
        dataset_class = TUT2016Dataset
        
    elif dataset_name == 'dcase2019':
        dataset_config = DCASE2019Config(
            root_path=data_dir,
            class_names=[
                'airport', 'bus', 'metro', 'metro_station', 'park',
                'public_square', 'shopping_mall', 'street_pedestrian',
                'street_traffic', 'tram'
            ],
            use_official_split=True,
            device_mismatch=True
        )
        dataset_class = DCASE2019Dataset
        
    elif dataset_name == 'esc50':
        dataset_config = ESC50Config(
            root_path=data_dir,
            class_names=[f'class_{i}' for i in range(50)],  # ESC-50 has 50 classes
            fold_based_eval=True
        )
        dataset_class = ESC50Dataset
        
    elif dataset_name == 'cochlscene':
        dataset_config = CochlSceneConfig(
            root_path=data_dir,
            class_names=[
                'airport', 'bus', 'metro', 'metro_station', 'park',
                'public_square', 'shopping_mall', 'street_pedestrian',
                'street_traffic', 'tram', 'beach', 'forest', 'train'
            ],
            quality_filter=True
        )
        dataset_class = CochlSceneDataset
        
    elif dataset_name == 'dcase2025':
        dataset_config = DCASE2025Config(
            root_path=data_dir,
            class_names=[
                'airport', 'bus', 'metro', 'metro_station', 'park',
                'public_square', 'shopping_mall', 'street_pedestrian',
                'street_traffic', 'tram'
            ],
            development_mode=True,
            low_complexity=True
        )
        dataset_class = DCASE2025Dataset
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_class, dataset_config, audio_config, spectrogram_config


def create_data_loaders(
    dataset_class: Any,
    dataset_config: DatasetConfig,
    audio_config: AudioConfig,
    spectrogram_config: SpectrogramConfig,
    batch_size: int,
    num_workers: int,
    augmentation_config: Optional[AugmentationConfig] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Create train, validation, test, and calibration data loaders."""
    
    # Create datasets for each split
    train_dataset = dataset_class(
        config=dataset_config,
        audio_config=audio_config,
        spectrogram_config=spectrogram_config,
        augmentation_config=augmentation_config,
        split='train'
    )
    
    val_dataset = dataset_class(
        config=dataset_config,
        audio_config=audio_config,
        spectrogram_config=spectrogram_config,
        split='val'
    )
    
    test_dataset = dataset_class(
        config=dataset_config,
        audio_config=audio_config,
        spectrogram_config=spectrogram_config,
        split='test'
    )
    
    # Create calibration dataset from validation set
    cal_dataset = dataset_class(
        config=dataset_config,
        audio_config=audio_config,
        spectrogram_config=spectrogram_config,
        split='val'  # Use validation data for calibration
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = create_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = create_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    cal_loader = create_dataloader(
        cal_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, cal_loader


def create_model_config(args: argparse.Namespace, num_classes: int) -> ComplexTransformerConfig:
    """Create model configuration from arguments."""
    return ComplexTransformerConfig(
        input_dim=args.n_mels,
        embed_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.hidden_dim * 4,
        num_classes=num_classes,
        max_seq_length=args.max_length,
        dropout=args.dropout,
        activation='complex_gelu',
        pos_encoding_type='complex_sinusoidal',
        pos_encoding_dropout=0.1,
        path_dropout=0.0
    )


def create_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Create training configuration from arguments."""
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer='adamw',
        scheduler='cosine',
        warmup_epochs=10,
        clip_grad_norm=1.0,
        use_amp=True,
        early_stopping=True,
        patience=15,
        min_delta=0.001,
        monitor_metric='val_accuracy',
        monitor_mode='max',
        loss_function='cross_entropy',
        label_smoothing=0.1
    )


def run_qxai_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run complete Q-XAI experiment."""
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/qxai_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Setup reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Random seed: {args.seed}")
    
    # Setup data
    logger.info(f"Setting up {args.dataset} dataset")
    dataset_class, dataset_config, audio_config, spectrogram_config = get_dataset_and_config(
        args.dataset, args.data_dir
    )
    
    # Create augmentation config for training
    augmentation_config = AugmentationConfig(
        use_specaugment=True,
        freq_mask_param=8,
        time_mask_param=25,
        num_freq_masks=2,
        num_time_masks=2,
        use_audio_augment=True,
        noise_prob=0.2,
        volume_prob=0.2
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader, cal_loader = create_data_loaders(
        dataset_class=dataset_class,
        dataset_config=dataset_config,
        audio_config=audio_config,
        spectrogram_config=spectrogram_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_config=augmentation_config
    )
    
    num_classes = len(dataset_config.class_names)
    logger.info(f"Dataset loaded - Classes: {num_classes}")
    
    # Create configurations
    model_config = create_model_config(args, num_classes)
    training_config = create_training_config(args)
    
    # Create Q-XAI configuration
    qxai_config = QXAIConfig(
        model=model_config,
        training=training_config
    )
    
    # Initialize model
    logger.info("Creating Q-XAI model")
    model = create_complex_transformer(
        config=model_config,
        model_type="standard",
        num_classes=num_classes
    ).to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Create trainer
    experiment_name = f"qxai_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    if args.use_full_pipeline:
        # Use the full training pipeline from trainer.py
        logger.info("Using full Q-XAI training pipeline")
        
        results = run_full_training_pipeline(
            config=qxai_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            calibration_loader=cal_loader,
            experiment_name=experiment_name,
            save_detailed_analysis=True
        )
        
    else:
        # Use simplified training approach
        logger.info("Using simplified training approach")
        
        trainer = create_trainer(
            config=qxai_config,
            model=model,
            experiment_name=experiment_name
        )
        
        # Training phase
        logger.info("Starting training phase")
        training_results = trainer.train(train_loader, val_loader)
        
        # Calibration phase  
        logger.info("Starting QICP calibration")
        calibration_stats = trainer.calibrate_qicp(cal_loader)
        
        # Evaluation phase
        logger.info("Starting comprehensive evaluation")
        evaluation_results = trainer.evaluate_comprehensive(test_loader)
        
        results = {
            'training': training_results,
            'calibration': calibration_stats,
            'evaluation': evaluation_results,
            'experiment_name': experiment_name
        }
        
        trainer.close()
    
    # Save final results
    results_file = Path(args.checkpoint_dir) / f"{experiment_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Experiment completed! Results saved to {results_file}")
    
    # Print summary
    if 'evaluation' in results:
        eval_results = results['evaluation']
        if 'classification' in eval_results:
            acc = eval_results['classification'].get('accuracy', 0)
            f1 = eval_results['classification'].get('macro_f1', 0)
            logger.info(f"Final Results - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        if 'uncertainty' in eval_results:
            ece = eval_results['uncertainty'].get('ece', 0)
            logger.info(f"Uncertainty - ECE: {ece:.4f}")
        
        if 'conformal_prediction' in eval_results:
            coverage = eval_results['conformal_prediction'].get('coverage', 0)
            set_size = eval_results['conformal_prediction'].get('average_set_size', 0)
            logger.info(f"QICP - Coverage: {coverage:.4f}, Avg Set Size: {set_size:.2f}")
    
    return results


class QXAIExperimentSimple:
    """Simplified Q-XAI experiment class for basic training and evaluation."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model = None
        self.qisa_attributor = None
        self.auq_estimator = None
        self.qicp_predictor = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.cal_loader = None
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
    def setup_data(self):
        """Setup data loaders."""
        self.logger.info(f"Setting up {self.args.dataset} dataset")
        
        dataset_class, dataset_config, audio_config, spectrogram_config = get_dataset_and_config(
            self.args.dataset, self.args.data_dir
        )
        
        # Create augmentation config
        augmentation_config = AugmentationConfig(
            use_specaugment=True,
            freq_mask_param=8,
            time_mask_param=25,
            num_freq_masks=2,
            num_time_masks=2
        )
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader, self.cal_loader = create_data_loaders(
            dataset_class=dataset_class,
            dataset_config=dataset_config,
            audio_config=audio_config,
            spectrogram_config=spectrogram_config,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            augmentation_config=augmentation_config
        )
        
        self.num_classes = len(dataset_config.class_names)
        self.logger.info(f"Data loaded - Classes: {self.num_classes}")
    
    def setup_model(self):
        """Setup model and training components."""
        self.logger.info("Setting up model")
        
        # Create model config
        model_config = create_model_config(self.args, self.num_classes)
        
        # Create model
        self.model = create_complex_transformer(
            config=model_config,
            model_type="standard",
            num_classes=self.num_classes
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=0
        )
        
        # Setup loss function
        self.criterion = ComplexCrossEntropyLoss()
        
        # Setup Q-XAI components
        self.qisa_attributor = create_qisa_attributor(
            self.model,
            method="standard"
        )
        
        self.auq_estimator = AUQEstimator(
            self.model,
            num_samples=self.args.auq_samples
        )
        
        self.qicp_predictor = create_qicp_predictor(
            score_type="born_rule"
        )
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model setup complete - Parameters: {total_params:,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle batch format
            if isinstance(batch, dict):
                inputs = batch['spectrogram'].to(self.device)
                targets = batch['label'].to(self.device)
            else:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            
            batch_size = inputs.size(0)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get complex outputs and convert to logits
            outputs = self.model(inputs, return_complex_amplitudes=True)
            if isinstance(outputs, tuple):
                logits, complex_amplitudes, _ = outputs
            else:
                logits = outputs
            
            # Compute loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': correct / total_samples
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle batch format
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(self.device)
                    targets = batch['label'].to(self.device)
                else:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                
                batch_size = inputs.size(0)
                
                # Forward pass
                outputs = self.model(inputs, return_complex_amplitudes=True)
                if isinstance(outputs, tuple):
                    logits, complex_amplitudes, _ = outputs
                else:
                    logits = outputs
                
                # Compute loss
                loss = self.criterion(logits, targets)
                
                # Track metrics
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                _, predicted = torch.max(logits, 1)
                correct += (predicted == targets).sum().item()
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': correct / total_samples
        }
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.args.epochs} epochs")
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Scheduler step
            self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch+1}/{self.args.epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.logger.info(f"Training completed. Best val accuracy: {best_val_acc:.4f}")
        return {'best_val_accuracy': best_val_acc}
    
    def evaluate(self) -> Dict[str, Any]:
        """Comprehensive evaluation."""
        self.logger.info("Starting evaluation")
        
        # Load best model
        self.load_checkpoint()
        
        # Calibrate QICP
        self.qicp_predictor.calibrate(self.model, self.cal_loader, self.device)
        
        # Test evaluation
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    inputs = batch['spectrogram'].to(self.device)
                    targets = batch['label'].to(self.device)
                else:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                
                outputs = self.model(inputs, return_complex_amplitudes=True)
                if isinstance(outputs, tuple):
                    logits, complex_amplitudes, _ = outputs
                else:
                    logits = outputs
                
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        # QICP evaluation
        qicp_result = self.qicp_predictor.predict(self.model, self.test_loader, self.device)
        
        results = {
            'classification': {
                'accuracy': accuracy,
                'f1_score': f1
            },
            'conformal_prediction': {
                'coverage': qicp_result.coverage,
                'average_set_size': qicp_result.average_set_size
            }
        }
        
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"Test F1: {f1:.4f}")
        self.logger.info(f"QICP Coverage: {qicp_result.coverage:.4f}")
        
        return results
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
    
    def load_checkpoint(self):
        """Load best checkpoint."""
        checkpoint_path = Path(self.args.checkpoint_dir) / 'best_model.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded best model checkpoint")
    
    def run(self) -> Dict[str, Any]:
        """Run complete experiment."""
        self.setup_data()
        self.setup_model()
        
        training_results = self.train()
        evaluation_results = self.evaluate()
        
        return {
            'training': training_results,
            'evaluation': evaluation_results
        }


def main():
    """Main function to run Q-XAI experiments."""
    parser = argparse.ArgumentParser(description='Q-XAI: Interpretable Complex-Valued Transformers')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='dcase2019',
                       choices=['tut2016', 'dcase2019', 'esc50', 'cochlscene', 'dcase2025'],
                       help='Dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='data/',
                       help='Directory containing the dataset')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension of the transformer')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                       help='Weight decay for optimizer')
    
    # Q-XAI specific arguments
    parser.add_argument('--auq_samples', type=int, default=50,
                       help='Number of Monte Carlo samples for AUQ')
    parser.add_argument('--conformal_alpha', type=float, default=0.1,
                       help='Miscoverage rate for conformal prediction')
    
    # Data processing arguments
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Audio sample rate')
    parser.add_argument('--n_mels', type=int, default=64,
                       help='Number of mel-frequency bins')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Experiment arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                       help='Directory to save model checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb_project', type=str, default='q-xai',
                       help='W&B project name')
    parser.add_argument('--use_full_pipeline', action='store_true',
                       help='Use full Q-XAI training pipeline with detailed analysis')
    parser.add_argument('--simple_mode', action='store_true',
                       help='Use simplified training mode (faster, less comprehensive)')
    
    args = parser.parse_args()
    
    # Setup experiment tracking
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"qxai_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    try:
        if args.simple_mode:
            # Use simplified experiment class
            print("Running Q-XAI in simple mode...")
            experiment = QXAIExperimentSimple(args)
            results = experiment.run()
            
        else:
            # Use full experiment pipeline
            print("Running full Q-XAI experiment...")
            results = run_qxai_experiment(args)
        
        print(f"\nQ-XAI Experiment completed successfully!")
        print(f"Results saved to: {args.checkpoint_dir}")
        
        # Print summary
        if 'evaluation' in results:
            eval_results = results['evaluation']
            if 'classification' in eval_results:
                acc = eval_results['classification'].get('accuracy', 0)
                f1 = eval_results['classification'].get('f1_score', 0)
                print(f"Final Results - Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            if 'uncertainty' in eval_results:
                ece = eval_results['uncertainty'].get('ece', 0)
                brier = eval_results['uncertainty'].get('brier_score', 0)
                print(f"Uncertainty - ECE: {ece:.4f}, Brier: {brier:.4f}")
            
            if 'conformal_prediction' in eval_results:
                coverage = eval_results['conformal_prediction'].get('coverage', 0)
                set_size = eval_results['conformal_prediction'].get('average_set_size', 0)
                target_coverage = eval_results['conformal_prediction'].get('target_coverage', 0.9)
                print(f"QICP - Coverage: {coverage:.4f} (target: {target_coverage:.4f}), Avg Set Size: {set_size:.2f}")
            
            if 'interpretability' in eval_results:
                sparsity = eval_results['interpretability'].get('average_attribution_sparsity', 0)
                print(f"Interpretability - Avg Attribution Sparsity: {sparsity:.4f}")
        
        # Log final results to wandb
        if args.use_wandb and 'evaluation' in results:
            eval_results = results['evaluation']
            final_metrics = {}
            
            if 'classification' in eval_results:
                final_metrics.update({
                    'final_accuracy': eval_results['classification'].get('accuracy', 0),
                    'final_macro_f1': eval_results['classification'].get('f1_score', 0)
                })
            
            if 'uncertainty' in eval_results:
                final_metrics.update({
                    'final_ece': eval_results['uncertainty'].get('ece', 0),
                    'final_brier_score': eval_results['uncertainty'].get('brier_score', 0)
                })
            
            if 'conformal_prediction' in eval_results:
                final_metrics.update({
                    'final_qicp_coverage': eval_results['conformal_prediction'].get('coverage', 0),
                    'final_qicp_set_size': eval_results['conformal_prediction'].get('average_set_size', 0)
                })
            
            if 'interpretability' in eval_results:
                final_metrics.update({
                    'final_attribution_sparsity': eval_results['interpretability'].get('average_attribution_sparsity', 0)
                })
            
            wandb.log(final_metrics)
        
        return results
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise
    
    finally:
        if args.use_wandb:
            wandb.finish()


def run_benchmark_experiments():
    """Run benchmark experiments on all datasets for paper reproduction."""
    datasets = ['tut2016', 'dcase2019', 'esc50', 'cochlscene', 'dcase2025']
    
    # Default arguments for benchmark
    base_args = {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'hidden_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1,
        'auq_samples': 50,
        'conformal_alpha': 0.1,
        'seed': 42,
        'use_full_pipeline': True,
        'data_dir': 'data/',
        'checkpoint_dir': 'benchmark_results/',
        'sample_rate': 16000,
        'n_mels': 64,
        'max_length': 1024,
        'num_workers': 4
    }
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Running benchmark on {dataset.upper()}")
        print(f"{'='*60}")
        
        # Create args namespace
        args = argparse.Namespace(**base_args)
        args.dataset = dataset
        args.checkpoint_dir = f"benchmark_results/{dataset}/"
        
        try:
            # Run experiment
            results = run_qxai_experiment(args)
            all_results[dataset] = results
            
            # Print dataset summary
            if 'evaluation' in results:
                eval_results = results['evaluation']
                print(f"\n{dataset.upper()} Results:")
                if 'classification' in eval_results:
                    acc = eval_results['classification'].get('accuracy', 0)
                    f1 = eval_results['classification'].get('f1_score', 0)
                    print(f"  Accuracy: {acc:.4f}")
                    print(f"  F1-Score: {f1:.4f}")
                
                if 'uncertainty' in eval_results:
                    ece = eval_results['uncertainty'].get('ece', 0)
                    print(f"  ECE: {ece:.4f}")
                
                if 'conformal_prediction' in eval_results:
                    coverage = eval_results['conformal_prediction'].get('coverage', 0)
                    set_size = eval_results['conformal_prediction'].get('average_set_size', 0)
                    print(f"  QICP Coverage: {coverage:.4f}")
                    print(f"  QICP Set Size: {set_size:.2f}")
            
        except Exception as e:
            print(f"Failed to run benchmark on {dataset}: {e}")
            all_results[dataset] = {'error': str(e)}
    
    # Save comprehensive benchmark results
    benchmark_file = Path("benchmark_results/comprehensive_benchmark.json")
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(benchmark_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {benchmark_file}")
    
    # Print summary table
    print("\nSummary Table:")
    print(f"{'Dataset':<12} {'Accuracy':<10} {'F1':<8} {'ECE':<8} {'Coverage':<10} {'Set Size':<10}")
    print("-" * 66)
    
    for dataset, results in all_results.items():
        if 'error' not in results and 'evaluation' in results:
            eval_res = results['evaluation']
            acc = eval_res.get('classification', {}).get('accuracy', 0)
            f1 = eval_res.get('classification', {}).get('f1_score', 0) 
            ece = eval_res.get('uncertainty', {}).get('ece', 0)
            coverage = eval_res.get('conformal_prediction', {}).get('coverage', 0)
            set_size = eval_res.get('conformal_prediction', {}).get('average_set_size', 0)
            
            print(f"{dataset:<12} {acc:<10.4f} {f1:<8.4f} {ece:<8.4f} {coverage:<10.4f} {set_size:<10.2f}")
        else:
            print(f"{dataset:<12} {'ERROR':<10} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10}")
    
    return all_results


if __name__ == '__main__':
    # Check if running benchmark mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        run_benchmark_experiments()
    else:
        main()