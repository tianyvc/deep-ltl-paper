"""
Options Framework：基于自动机状态的层次化强化学习

核心思想：
- 每个自动机状态 q 对应一个 Option（技能）
- Meta-Controller 选择目标状态
- Option 策略执行直到到达目标或终止
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class OptionStatus(Enum):
    """Option 执行状态"""
    RUNNING = 0
    TERMINATED = 1
    SUCCEEDED = 2
    FAILED = 3


@dataclass
class OptionTransition:
    """Option 执行的转移"""
    state: torch.Tensor
    automaton_state: int
    action: int
    reward: float
    next_state: torch.Tensor
    next_automaton_state: int
    intrinsic_reward: float
    done: bool
    option_status: OptionStatus


class AutomatonOption(nn.Module):
    """
    单个 Option：对应一个自动机目标状态
    
    - Initiation: 从哪些状态可以开始
    - Policy: 如何行动
    - Termination: 何时停止
    """
    def __init__(
        self,
        option_id: int,
        target_automaton_state: int,
        ldba,
        policy_network: nn.Module,
        max_steps: int = 50,
        intrinsic_reward_scale: float = 1.0
    ):
        super().__init__()
        self.option_id = option_id
        self.target_q = target_automaton_state
        self.ldba = ldba
        self.policy = policy_network
        self.max_steps = max_steps
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # 步数计数器
        self.steps_executed = 0
    
    def can_initiate(self, current_q: int) -> bool:
        """判断是否可以从当前自动机状态启动该 Option"""
        # 检查是否存在路径：current_q → target_q
        return self._is_reachable(current_q, self.target_q)
    
    def select_action(
        self, 
        state: torch.Tensor, 
        current_q: int,
        formula_emb: torch.Tensor
    ) -> int:
        """Option 策略选择动作"""
        # 构造输入：[state, current_q_onehot, target_q_onehot, formula_emb]
        q_current_onehot = self._onehot(current_q, self.ldba.num_states)
        q_target_onehot = self._onehot(self.target_q, self.ldba.num_states)
        
        policy_input = torch.cat([
            state,
            q_current_onehot,
            q_target_onehot,
            formula_emb
        ], dim=-1)
        
        action_logits = self.policy(policy_input)
        action = torch.argmax(action_logits, dim=-1)
        
        return action.item()
    
    def compute_intrinsic_reward(
        self, 
        prev_q: int, 
        next_q: int
    ) -> float:
        """
        计算内在奖励（引导 Option 向目标前进）
        
        奖励结构：
        - 到达目标：+1.0
        - 接近目标：+0.1 * (距离减少量)
        - 远离目标：-0.1 * (距离增加量)
        """
        if next_q == self.target_q:
            return 1.0 * self.intrinsic_reward_scale
        
        # 计算距离变化
        dist_prev = self._distance_to_target(prev_q)
        dist_next = self._distance_to_target(next_q)
        dist_delta = dist_prev - dist_next
        
        return 0.1 * dist_delta * self.intrinsic_reward_scale
    
    def check_termination(
        self, 
        current_q: int, 
        steps: int
    ) -> OptionStatus:
        """检查终止条件"""
        # 成功：到达目标状态
        if current_q == self.target_q:
            return OptionStatus.SUCCEEDED
        
        # 失败：超过最大步数
        if steps >= self.max_steps:
            return OptionStatus.FAILED
        
        # 继续执行
        return OptionStatus.RUNNING
    
    def _is_reachable(self, from_q: int, to_q: int) -> bool:
        """BFS 检查可达性"""
        if from_q == to_q:
            return True
        
        from collections import deque
        visited = set()
        queue = deque([from_q])
        
        while queue:
            q = queue.popleft()
            if q == to_q:
                return True
            
            visited.add(q)
            for trans in self.ldba.state_to_transitions.get(q, []):
                if trans.target not in visited:
                    queue.append(trans.target)
        
        return False
    
    def _distance_to_target(self, q: int) -> int:
        """到目标状态的最短距离（BFS）"""
        if q == self.target_q:
            return 0
        
        from collections import deque
        visited = {q: 0}
        queue = deque([q])
        
        while queue:
            current = queue.popleft()
            for trans in self.ldba.state_to_transitions.get(current, []):
                next_q = trans.target
                if next_q not in visited:
                    visited[next_q] = visited[current] + 1
                    if next_q == self.target_q:
                        return visited[next_q]
                    queue.append(next_q)
        
        return float('inf')
    
    def _onehot(self, idx: int, num_classes: int) -> torch.Tensor:
        """One-hot 编码"""
        vec = torch.zeros(num_classes)
        vec[idx] = 1.0
        return vec


class MetaController(nn.Module):
    """
    高层元控制器：选择要执行的 Option
    
    Q(q, o) = 期望从状态 q 执行 option o 的价值
    """
    def __init__(
        self,
        num_automaton_states: int,
        num_options: int,
        formula_emb_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.num_automaton_states = num_automaton_states
        self.num_options = num_options
        
        # Meta-Q 网络
        self.q_network = nn.Sequential(
            nn.Linear(num_automaton_states + num_options + formula_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def select_option(
        self,
        current_q: int,
        available_options: List[int],
        formula_emb: torch.Tensor,
        epsilon: float = 0.1
    ) -> int:
        """
        选择 Option（ε-greedy）
        
        Args:
            current_q: 当前自动机状态
            available_options: 可用 Option ID 列表
            formula_emb: 公式嵌入
            epsilon: 探索率
        
        Returns:
            selected_option_id
        """
        if torch.rand(1).item() < epsilon:
            # 探索：随机选择
            return available_options[torch.randint(len(available_options), (1,)).item()]
        
        # 利用：选择最大 Q 值的 Option
        q_current_onehot = self._onehot(current_q, self.num_automaton_states)
        
        q_values = []
        for option_id in available_options:
            option_onehot = self._onehot(option_id, self.num_options)
            q_input = torch.cat([q_current_onehot, option_onehot, formula_emb], dim=-1)
            q_val = self.q_network(q_input)
            q_values.append(q_val.item())
        
        best_idx = torch.argmax(torch.tensor(q_values)).item()
        return available_options[best_idx]
    
    def compute_q_value(
        self,
        q: int,
        option_id: int,
        formula_emb: torch.Tensor
    ) -> torch.Tensor:
        """计算 Q(q, o)"""
        q_onehot = self._onehot(q, self.num_automaton_states)
        option_onehot = self._onehot(option_id, self.num_options)
        q_input = torch.cat([q_onehot, option_onehot, formula_emb], dim=-1)
        return self.q_network(q_input)
    
    def _onehot(self, idx: int, num_classes: int) -> torch.Tensor:
        vec = torch.zeros(num_classes)
        vec[idx] = 1.0
        return vec


class HierarchicalController:
    """
    层次化控制器：协调 Meta-Controller 和 Options
    
    训练流程：
    1. Meta-Controller 选择 Option
    2. Option 执行直到终止
    3. 用内在奖励更新 Option 策略
    4. 用环境奖励更新 Meta-Controller
    """
    def __init__(
        self,
        ldba,
        meta_controller: MetaController,
        options: Dict[int, AutomatonOption],
        gamma: float = 0.99,
        gamma_meta: float = 0.95
    ):
        self.ldba = ldba
        self.meta_controller = meta_controller
        self.options = options
        self.gamma = gamma  # Option 内折扣
        self.gamma_meta = gamma_meta  # Meta 层折扣
        
        # 优化器
        self.meta_optimizer = torch.optim.Adam(meta_controller.parameters(), lr=3e-4)
        self.option_optimizers = {
            opt_id: torch.optim.Adam(opt.parameters(), lr=3e-4)
            for opt_id, opt in options.items()
        }
    
    def train_step(
        self,
        transitions: List[OptionTransition],
        formula_emb: torch.Tensor
    ):
        """
        训练一个 episode 的 transitions
        
        Args:
            transitions: OptionTransition 列表
            formula_emb: 公式嵌入
        """
        # 1. 更新 Options（用内在奖励）
        self._update_options(transitions)
        
        # 2. 更新 Meta-Controller（用环境奖励）
        self._update_meta_controller(transitions, formula_emb)
    
    def _update_options(self, transitions: List[OptionTransition]):
        """用内在奖励更新 Option 策略"""
        # 按 Option 分组
        option_groups = {}
        for trans in transitions:
            # 假设 transition 记录了使用的 option_id
            # （需要在实际执行时记录）
            option_id = trans.option_id if hasattr(trans, 'option_id') else None
            if option_id is None:
                continue
            
            if option_id not in option_groups:
                option_groups[option_id] = []
            option_groups[option_id].append(trans)
        
        # 对每个 Option 计算策略梯度
        for option_id, trans_list in option_groups.items():
            option = self.options[option_id]
            optimizer = self.option_optimizers[option_id]
            
            # 计算 return（用内在奖励）
            returns = []
            G = 0
            for trans in reversed(trans_list):
                G = trans.intrinsic_reward + self.gamma * G
                returns.insert(0, G)
            
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # 策略梯度损失
            loss = 0
            for trans, G in zip(trans_list, returns):
                # 重新计算 log_prob（需要记录 action）
                # 这里简化为 placeholder
                # log_prob = option.policy.log_prob(trans.state, trans.action)
                # loss -= log_prob * G
                pass
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def _update_meta_controller(
        self, 
        transitions: List[OptionTransition],
        formula_emb: torch.Tensor
    ):
        """用环境奖励更新 Meta-Controller"""
        # 提取 Option 切换点
        option_transitions = []
        cumulative_reward = 0
        
        for i, trans in enumerate(transitions):
            cumulative_reward += trans.reward
            
            # Option 终止时记录
            if trans.option_status != OptionStatus.RUNNING:
                option_transitions.append({
                    'q': trans.automaton_state,
                    'next_q': trans.next_automaton_state,
                    'reward': cumulative_reward,
                    'done': trans.done
                })
                cumulative_reward = 0
        
        # Q-learning 更新
        for trans_dict in option_transitions:
            q = trans_dict['q']
            next_q = trans_dict['next_q']
            reward = trans_dict['reward']
            done = trans_dict['done']
            
            # 当前 Q 值
            # （需要记录使用的 option_id）
            # q_current = self.meta_controller.compute_q_value(q, option_id, formula_emb)
            
            # 目标 Q 值
            if not done:
                available_options = self._get_available_options(next_q)
                q_values_next = [
                    self.meta_controller.compute_q_value(next_q, opt_id, formula_emb)
                    for opt_id in available_options
                ]
                q_target = reward + self.gamma_meta * max(q_values_next)
            else:
                q_target = reward
            
            # TD 误差
            # loss = (q_current - q_target.detach()) ** 2
            
            # 反向传播
            # self.meta_optimizer.zero_grad()
            # loss.backward()
            # self.meta_optimizer.step()
    
    def _get_available_options(self, q: int) -> List[int]:
        """获取从状态 q 可用的 Options"""
        available = []
        for opt_id, option in self.options.items():
            if option.can_initiate(q):
                available.append(opt_id)
        return available


# 使用示例
if __name__ == "__main__":
    print("=== Hierarchical Options Framework ===")
    print("核心组件：")
    print("1. AutomatonOption: 封装子目标策略")
    print("2. MetaController: 高层 Option 选择")
    print("3. HierarchicalController: 协调训练")
    print("\n训练流程：")
    print("- Meta 选择 Option → Option 执行 → 内在奖励更新 Option → 环境奖励更新 Meta")
