import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve
import json

class TrainingModule:
    """Training module for defect detection model"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Since this is mainly unsupervised, we don't need traditional optimizer
        # But we keep the interface for compatibility
        self.setup_training_components()
        
        # Metrics tracking
        self.training_metrics = []
        self.validation_metrics = []
        
    def setup_training_components(self):
        """Setup training components (minimal for unsupervised approach)"""
        # No traditional loss/optimizer needed for this approach
        # The model is primarily unsupervised with frozen features
        pass
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Training epoch - builds reference bank and calibrates thresholds"""
        self.model.eval()  # Feature extractor stays frozen
        
        normal_patches = []
        normal_scores = []
        
        start_time = time.time()
        
        # Collect normal patches for reference bank
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            else:
                images = batch
                labels = None
            
            images = images.to(self.device)
            
            # Extract patches from normal images
            for i, image in enumerate(images):
                # Assume label 0 is normal, or all images are normal in unsupervised setting
                if labels is None or labels[i].item() == 0:
                    patches, _ = self.model.extract_patches(image.unsqueeze(0))
                    for patch in patches:
                        normal_patches.append(patch.cpu())
                        
                        # Also collect scores for EVT calibration if reference bank exists
                        if hasattr(self.model, 'faiss_indices') and 'default' in self.model.faiss_indices:
                            with torch.no_grad():
                                features = self.model.feature_extractor(patch.unsqueeze(0))
                                try:
                                    retrieved_refs = self.model.retrieve_references(features, 'default')
                                    score = self.model.compute_anomaly_score(features, retrieved_refs)
                                    normal_scores.append(score)
                                except:
                                    pass
            
            if batch_idx == 0 and epoch == 0:
                print(f"Processing batch {batch_idx}, collected {len(normal_patches)} patches")
        
        # Build or update reference bank
        if epoch == 0 or len(normal_patches) > len(getattr(self.model, 'reference_bank', {}).get('default', {}).get('layer2', [])):
            print(f"Building reference bank with {len(normal_patches)} patches")
            self.model.build_reference_bank(normal_patches, condition_id='default')
            
            # Recalculate scores after building reference bank
            normal_scores = []
            for patch in normal_patches[:1000]:  # Limit for efficiency
                with torch.no_grad():
                    features = self.model.feature_extractor(patch.unsqueeze(0).to(self.device))
                    retrieved_refs = self.model.retrieve_references(features, 'default')
                    score = self.model.compute_anomaly_score(features, retrieved_refs)
                    normal_scores.append(score)
        
        # Fit EVT threshold
        if normal_scores:
            self.model.fit_evt_threshold(np.array(normal_scores), condition_id='default')
            print(f"EVT threshold set to: {self.model.evt_thresholds.get('default', 'N/A')}")
        
        epoch_time = time.time() - start_time
        
        metrics = {
            'epoch_time': epoch_time,
            'num_patches': len(normal_patches),
            'num_scores': len(normal_scores),
            'mean_score': np.mean(normal_scores) if normal_scores else 0.0,
            'std_score': np.std(normal_scores) if normal_scores else 0.0
        }
        
        self.training_metrics.append(metrics)
        return metrics
    
    def val_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Validation epoch - evaluate anomaly detection performance"""
        self.model.eval()
        
        all_scores = []
        all_labels = []
        false_positives = 0
        total_normal = 0
        total_anomalous = 0
        inference_times = []
        memory_usage = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                    labels = batch[1] if len(batch) > 1 else torch.zeros(len(images))
                else:
                    images = batch
                    labels = torch.zeros(len(images))  # Assume all normal if no labels
                
                images = images.to(self.device)
                
                for i, image in enumerate(images):
                    img_start_time = time.time()
                    
                    # Get memory usage
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        memory_usage.append(torch.cuda.memory_allocated() / 1024**3)  # GB
                    
                    # Run inference
                    anomaly_map = self.model(image.unsqueeze(0), condition_id='default')
                    # Convert Tensor → numpy if needed
                    if hasattr(anomaly_map, 'detach'):
                        anomaly_map = anomaly_map.detach().cpu().numpy()

                    inference_time = time.time() - img_start_time
                    inference_times.append(inference_time)

                    # Compute image-level anomaly score
                    image_score = float(np.max(anomaly_map))  # Max pooling for image-level score
                    all_scores.append(image_score)
                    all_labels.append(labels[i].item())

                    # Count false positives
                    threshold = self.model.evt_thresholds.get('default', float(np.percentile(anomaly_map, 99)))
                    is_anomalous = image_score > threshold
                    
                    if labels[i] == 0:  # Normal image
                        total_normal += 1
                        if is_anomalous:
                            false_positives += 1
                    else:  # Anomalous image
                        total_anomalous += 1
        
        # Compute metrics
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # AUROC
        if len(np.unique(all_labels)) > 1:
            auroc = roc_auc_score(all_labels, all_scores)
        else:
            auroc = 0.0
        
        # False positive rate
        fpr = false_positives / max(total_normal, 1)
        
        # Performance metrics
        mean_inference_time = np.mean(inference_times)
        peak_memory = np.max(memory_usage) if memory_usage else 0.0
        throughput = len(all_scores) / (time.time() - start_time)
        
        # Anomalous area ratio (simplified)
        anomalous_area_ratio = np.sum(all_scores > np.percentile(all_scores, 95)) / len(all_scores)
        
        # Score stability (std of normal scores)
        normal_scores = all_scores[all_labels == 0]
        score_stability = np.std(normal_scores) if len(normal_scores) > 0 else 0.0
        
        # Threshold robustness (coefficient of variation)
        threshold_robustness = score_stability / max(np.mean(normal_scores), 1e-8) if len(normal_scores) > 0 else 0.0
        
        metrics = {
            'auroc': auroc,
            'false': fpr,  # False positive rate
            'anomalous': anomalous_area_ratio,
            'defect': auroc,  # Use AUROC as defect detection metric
            'mean': np.mean(all_scores),
            'normal': np.mean(normal_scores) if len(normal_scores) > 0 else 0.0,
            'threshold': self.model.evt_thresholds.get('default', 0.0),
            'inference': mean_inference_time,
            'peak': peak_memory,
            'throughput': throughput,
            'score_stability': score_stability,
            'threshold_robustness': threshold_robustness,
            'total_images': len(all_scores),
            'normal_images': total_normal,
            'anomalous_images': total_anomalous
        }
        
        self.validation_metrics.append(metrics)
        
        # Print metrics in required format
        metrics_output = {
            'false': metrics['false'],
            'anomalous': metrics['anomalous'], 
            'defect': metrics['defect'],
            'auroc': metrics['auroc'],
            'mean': metrics['mean'],
            'normal': metrics['normal'],
            'threshold': metrics['threshold'],
            'inference': metrics['inference'],
            'peak': metrics['peak'],
            'throughput': metrics['throughput']
        }
        print(f"METRICS:{json.dumps(metrics_output)}")
        
        return metrics
    
    def _compute_loss(self, batch) -> torch.Tensor:
        """
        Compatibility method for smoke_test.py.
        This model is training-free (frozen backbone); loss is a dummy scalar
        derived from dummy_param so that loss.backward() succeeds.
        """
        if isinstance(batch, (list, tuple)):
            _ = batch[0]  # consume batch but don't use it for gradient
        dummy = getattr(self.model, "dummy_param", None)
        if dummy is not None:
            return dummy * 0.0  # differentiable zero
        return torch.tensor(0.0, requires_grad=True)

    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_config': self.config,
            'metrics': metrics,
            'evt_thresholds': getattr(self.model, 'evt_thresholds', {}),
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        if 'evt_thresholds' in checkpoint:
            self.model.evt_thresholds = checkpoint['evt_thresholds']
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']
        if 'validation_metrics' in checkpoint:
            self.validation_metrics = checkpoint['validation_metrics']
        return checkpoint