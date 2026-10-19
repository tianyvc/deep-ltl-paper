"""
层次化价值函数：结合自动机状态追踪与短期环境奖励

核心思想：
- 短期价值：估计即时环境回报（V_short）
- 长期价值：估计距离满足公式的进度（V_long）
- 动态融合：根据自动机阶段调整权重
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AutomatonState:
    """自动机状态的增强表示"""
    state_id: int
    is_accepting: bool
    in_scc: bool
    distance_to_accepting: int  # 到最近接受状态的距离
    progress_ratio: float  # 任务完成进度 [0, 1]


class AutomatonStateTracker:
    """
    实时追踪智能体在 LDBA 中的位置
    """
    def __init__(self, ldba, device='cpu'):
        self.ldba = ldba
        self.device = device
        self.current_states = None  # [batch_size]
        
        # 预计算：每个状态到接受状态的最短距离
        self.state_to_accepting_distance = self._compute_distances()
    
    def reset(self, batch_size: int):
        """重置为初始状态"""
        self.current_states = torch.full(
            (batch_size,), 
            self.ldba.initial_state, 
            dtype=torch.long, 
            device=self.device
        )
    
    def step(self, propositions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据环境命题更新自动机状态
        
        Args:
            propositions: [batch_size, n_props] 布尔张量
        
        Returns:
            next_states: [batch_size] 新状态 ID
            is_accepting: [batch_size] 是否在接受状态
        """
        batch_size = propositions.shape[0]
        next_states = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        is_accepting = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for i in range(batch_size):
            current_q = self.current_states[i].item()
            props_set = self._tensor_to_prop_set(propositions[i])
            
            # 使用 LDBA 的 get_next_state 方法
            try:
                next_q, accepting = self.ldba.get_next_state(current_q, props_set)
                next_states[i] = next_q
                is_accepting[i] = accepting
            except (ValueError, KeyError):
                # 如果转移无效，保持当前状态
                next_states[i] = current_q
                is_accepting[i] = False
        
        self.current_states = next_states
        return next_states, is_accepting
    
    def get_state_features(self) -> torch.Tensor:
        """
        提取当前自动机状态的特征
        
        Returns:
            features: [batch_size, 4] 
                      [is_accepting, in_scc, dist_norm, progress]
        """
        batch_size = len(self.current_states)
        features = torch.zeros(batch_size, 4, device=self.device)
        
        # 确保 SCC 已计算
        if not self.ldba.state_to_scc:
            self.ldba.compute_sccs()
        
        max_dist = max(self.state_to_accepting_distance.values())
        if max_dist == 0:
            max_dist = 1  # 避免除零
        
        for i, q in enumerate(self.current_states):
            q_val = q.item()
            scc_info = self.ldba.state_to_scc.get(q_val)
            
            features[i, 0] = float(scc_info.accepting if scc_info else False)
            features[i, 1] = float(len(scc_info.states) > 1 if scc_info else False)
            features[i, 2] = self.state_to_accepting_distance[q_val] / max_dist
            features[i, 3] = 1.0 - features[i, 2]  # 进度 = 1 - 归一化距离
        
        return features
    
    def _compute_distances(self) -> Dict[int, int]:
        """BFS 计算每个状态到接受状态的最短距离"""
        from collections import deque
        
        distances = {q: float('inf') for q in self.ldba.states}
        queue = deque()
        
        # 确保 SCC 已计算
        if not self.ldba.state_to_scc:
            self.ldba.compute_sccs()
        
        # 从所有接受状态开始反向搜索
        for q in self.ldba.states:
            scc_info = self.ldba.state_to_scc.get(q)
            if scc_info and scc_info.accepting:
                distances[q] = 0
                queue.append(q)
        
        # 如果没有接受状态，返回默认距离
        if not queue:
            return {q: 1 for q in self.ldba.states}
        
        while queue:
            q = queue.popleft()
            # 反向遍历（找谁能到达 q）
            if q in self.ldba.state_to_incoming_transitions:
                for trans in self.ldba.state_to_incoming_transitions[q]:
                    prev_q = trans.source
                    if distances[prev_q] > distances[q] + 1:
                        distances[prev_q] = distances[q] + 1
                        queue.append(prev_q)
        
        # 处理不可达状态
        max_dist = max([d for d in distances.values() if d != float('inf')], default=1)
        for q in distances:
            if distances[q] == float('inf'):
                distances[q] = max_dist + 1
        
        return distances
    
    def _tensor_to_prop_set(self, prop_tensor: torch.Tensor) -> set:
        """将布尔张量转为命题集合"""
        props = []
        for i, val in enumerate(prop_tensor):
            if val > 0.5:
                props.append(self.ldba.propositions[i])
        return set(props)


class HierarchicalValueNetwork(nn.Module):
    """
    双时间尺度价值网络
    
    V = α * V_long(q, z_φ) + (1-α) * V_short(s)
    
    其中 α 根据自动机阶段动态调整
    """
    def __init__(
        self, 
        env_state_dim: int,
        automaton_state_dim: int = 4,  # [accepting, scc, dist, progress]
        formula_emb_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # 短期价值网络（基于环境状态）
        self.short_term_net = nn.Sequential(
            nn.Linear(env_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 长期价值网络（基于自动机 + 公式）
        self.long_term_net = nn.Sequential(
            nn.Linear(automaton_state_dim + formula_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 权重调节网络（学习 α）
        self.weight_net = nn.Sequential(
            nn.Linear(automaton_state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # α ∈ [0, 1]
        )
    
    def forward(
        self, 
        env_state: torch.Tensor,
        automaton_features: torch.Tensor,
        formula_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            env_state: [batch, env_dim]
            automaton_features: [batch, 4]
            formula_embedding: [batch, emb_dim]
        
        Returns:
            dict with keys: v_total, v_short, v_long, alpha
        """
        # 计算短期价值
        v_short = self.short_term_net(env_state)  # [batch, 1]
        
        # 计算长期价值
        long_input = torch.cat([automaton_features, formula_embedding], dim=-1)
        v_long = self.long_term_net(long_input)  # [batch, 1]
        
        # 动态权重
        alpha = self.weight_net(automaton_features)  # [batch, 1]
        
        # 融合
        v_total = alpha * v_long + (1 - alpha) * v_short
        
        return {
            'v_total': v_total,
            'v_short': v_short,
            'v_long': v_long,
            'alpha': alpha
        }


class MultiTimescaleLoss(nn.Module):
    """
    多时间尺度联合损失
    
    L = λ_short * L_short + λ_mid * L_mid + λ_long * L_long
    """
    def __init__(
        self, 
        lambda_short: float = 1.0,
        lambda_mid: float = 0.5,
        lambda_long: float = 0.3
    ):
        super().__init__()
        self.lambda_short = lambda_short
        self.lambda_mid = lambda_mid
        self.lambda_long = lambda_long
    
    def forward(
        self,
        v_pred: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            v_pred: [batch, seq_len] 预测价值
            rewards: [batch, seq_len] 实际奖励
            dones: [batch, seq_len] 终止标记
            gamma: 折扣因子
        
        Returns:
            dict with total_loss, loss_short, loss_mid, loss_long
        """
        batch_size, seq_len = rewards.shape
        
        # 短期损失：1-step TD
        returns_1step = rewards[:, :-1] + gamma * v_pred[:, 1:] * (1 - dones[:, :-1])
        loss_short = nn.functional.mse_loss(v_pred[:, :-1], returns_1step.detach())
        
        # 中期损失：5-step return
        n_mid = min(5, seq_len)
        returns_mid = self._compute_nstep_return(rewards, v_pred, dones, n_mid, gamma)
        loss_mid = nn.functional.mse_loss(v_pred[:, :-n_mid], returns_mid.detach())
        
        # 长期损失：完整 episode return
        returns_long = self._compute_mc_return(rewards, dones, gamma)
        loss_long = nn.functional.mse_loss(v_pred[:, 0], returns_long.detach())
        
        # 总损失
        total_loss = (
            self.lambda_short * loss_short +
            self.lambda_mid * loss_mid +
            self.lambda_long * loss_long
        )
        
        return {
            'total_loss': total_loss,
            'loss_short': loss_short,
            'loss_mid': loss_mid,
            'loss_long': loss_long
        }
    
    def _compute_nstep_return(self, rewards, values, dones, n, gamma):
        """计算 n-step return"""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros(batch_size, seq_len - n, device=rewards.device)
        
        for i in range(seq_len - n):
            ret = 0
            for j in range(n):
                ret += (gamma ** j) * rewards[:, i + j]
                if dones[:, i + j].any():
                    break
            ret += (gamma ** n) * values[:, i + n] * (1 - dones[:, i + n - 1])
            returns[:, i] = ret
        
        return returns
    
    def _compute_mc_return(self, rewards, dones, gamma):
        """计算 Monte Carlo return"""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros(batch_size, device=rewards.device)
        
        for i in range(batch_size):
            ret = 0
            for t in range(seq_len):
                ret += (gamma ** t) * rewards[i, t]
                if dones[i, t]:
                    break
            returns[i] = ret
        
        return returns


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    batch_size = 16
    env_dim = 32
    formula_dim = 64
    seq_len = 100
    
    # 初始化模块
    value_net = HierarchicalValueNetwork(env_dim, formula_emb_dim=formula_dim)
    loss_fn = MultiTimescaleLoss()
    
    # 模拟输入
    env_states = torch.randn(batch_size, seq_len, env_dim)
    automaton_features = torch.rand(batch_size, seq_len, 4)
    formula_emb = torch.randn(batch_size, formula_dim)
    rewards = torch.randn(batch_size, seq_len)
    dones = torch.zeros(batch_size, seq_len)
    
    # 前向传播
    v_preds = []
    for t in range(seq_len):
        out = value_net(
            env_states[:, t],
            automaton_features[:, t],
            formula_emb
        )
        v_preds.append(out['v_total'])
    
    v_preds = torch.cat(v_preds, dim=1)
    
    # 计算损失
    losses = loss_fn(v_preds, rewards, dones)
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    print(f"Short-term: {losses['loss_short'].item():.4f}")
    print(f"Mid-term: {losses['loss_mid'].item():.4f}")
    print(f"Long-term: {losses['loss_long'].item():.4f}")
