import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class GraphContrastiveLoss(nn.Module):
    """图对比学习损失函数"""
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        计算图对比损失
        Args:
            features: 图表示 [num_graphs, hidden_dim]
            batch: batch索引 [num_graphs]
        """
        device = features.device
        batch_size = len(torch.unique(batch))
        
        # 计算相似度矩阵
        sim = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        sim = sim / self.temperature
        
        # 创建正样本mask
        pos_mask = (batch.unsqueeze(0) == batch.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  # 移除自身
        
        # 计算对比损失
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (log_prob * pos_mask).sum(1) / pos_mask.sum(1)
        
        return -mean_log_prob.mean()

class CurriculumScheduler:
    """课程学习调度器"""
    
    def __init__(self, 
                 initial_complexity: float = 0.3,
                 final_complexity: float = 1.0,
                 epochs: int = 100,
                 warmup_epochs: int = 10):
        self.initial_complexity = initial_complexity
        self.final_complexity = final_complexity
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        
    def get_complexity(self, epoch: int) -> float:
        """获取当前epoch的任务复杂度"""
        if epoch < self.warmup_epochs:
            return self.initial_complexity
            
        progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
        return min(self.initial_complexity + (self.final_complexity - self.initial_complexity) * progress,
                  self.final_complexity)
                  
    def should_increase_complexity(self, metrics: Dict[str, float], threshold: float = 0.8) -> bool:
        """基于性能指标决定是否增加复杂度"""
        return metrics.get('success_rate', 0) > threshold

class HierarchicalTrainer:
    """分层训练管理器"""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 curriculum: CurriculumScheduler,
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.curriculum = curriculum
        self.device = device
        self.contrastive_loss = GraphContrastiveLoss()
        
    def compute_hierarchical_loss(self, 
                                outputs: Dict[str, torch.Tensor],
                                targets: Dict[str, torch.Tensor],
                                graph_features: torch.Tensor,
                                batch_indices: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算分层损失"""
        losses = {}
        
        # 1. 任务特定损失
        task_loss = F.mse_loss(outputs['predictions'], targets['values'])
        losses['task'] = task_loss
        
        # 2. 图对比损失
        contrastive_loss = self.contrastive_loss(graph_features, batch_indices)
        losses['contrastive'] = contrastive_loss
        
        # 3. 结构正则化损失（促进层次结构的学习）
        if 'ldba_attention' in outputs and 'seq_attention' in outputs:
            structure_loss = self.compute_structure_loss(
                outputs['ldba_attention'],
                outputs['seq_attention']
            )
            losses['structure'] = structure_loss
            
        # 总损失
        total_loss = sum(losses.values())
        
        return total_loss, {k: v.item() for k, v in losses.items()}
        
    @staticmethod
    def compute_structure_loss(ldba_attn: torch.Tensor, seq_attn: torch.Tensor) -> torch.Tensor:
        """计算结构正则化损失"""
        # 促进注意力分布的多样性
        diversity_loss = -(ldba_attn.entropy().mean() + seq_attn.entropy().mean())
        
        # 促进层次间的信息流动
        correlation = F.cosine_similarity(
            ldba_attn.view(-1, ldba_attn.size(-1)),
            seq_attn.view(-1, seq_attn.size(-1))
        ).mean()
        correlation_loss = torch.abs(correlation)  # 降低层次间的冗余
        
        return diversity_loss + 0.1 * correlation_loss
        
    def train_epoch(self, 
                   train_loader: Any,
                   epoch: int,
                   **kwargs) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        metrics = {}
        
        # 获取当前的任务复杂度
        complexity = self.curriculum.get_complexity(epoch)
        
        for batch in train_loader:
            # 根据复杂度采样任务
            batch = self.sample_by_complexity(batch, complexity)
            
            # 前向传播
            outputs = self.model(batch)
            total_loss, losses = self.compute_hierarchical_loss(
                outputs,
                batch['targets'],
                outputs['graph_features'],
                batch['batch_indices']
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            epoch_losses.append(total_loss.item())
            
            # 更新指标
            for k, v in losses.items():
                metrics[k] = metrics.get(k, 0) + v
                
        # 计算平均损失和指标
        metrics = {k: v / len(train_loader) for k, v in metrics.items()}
        metrics['loss'] = sum(epoch_losses) / len(epoch_losses)
        
        # 更新学习率
        self.scheduler.step(metrics['loss'])
        
        return metrics
        
    @staticmethod
    def sample_by_complexity(batch: Dict[str, Any], complexity: float) -> Dict[str, Any]:
        """根据复杂度采样任务"""
        batch_size = len(batch['batch_indices'])
        num_samples = int(batch_size * complexity)
        indices = torch.randperm(batch_size)[:num_samples]
        
        return {k: v[indices] if torch.is_tensor(v) else v 
               for k, v in batch.items()}