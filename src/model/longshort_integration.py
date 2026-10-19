"""
集成模块：将长短期RL组件嵌入现有 deep-ltl 模型

使用方式：
    from model.longshort_integration import enhance_model_with_multiscale
    
    model = build_model(...)
    model = enhance_model_with_multiscale(
        model, 
        enable_hierarchical_value=True,
        enable_multiscale_gnn=True,
        enable_automaton_tracking=True
    )
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

from model.hierarchical_value import (
    AutomatonStateTracker,
    HierarchicalValueNetwork,
    MultiTimescaleLoss
)
from model.multiscale_gnn import (
    MultiScaleSemanticGNN,
    MultiTimescaleValueLoss
)


@dataclass
class LongShortConfig:
    """长短期RL配置"""
    # 层次化价值
    enable_hierarchical_value: bool = True
    lambda_short: float = 1.0
    lambda_mid: float = 0.5
    lambda_long: float = 0.3
    
    # 多尺度GNN
    enable_multiscale_gnn: bool = True
    gnn_num_layers: int = 3
    
    # 自动机追踪
    enable_automaton_tracking: bool = True
    
    # Options框架（可选，更复杂）
    enable_options: bool = False


class LongShortEnhancedModel(nn.Module):
    """
    增强模型：在原有 Model 基础上添加长短期RL组件
    
    保持向后兼容：
    - 如果所有开关关闭，行为与原模型一致
    - 渐进式启用：可单独开启任意组件
    """
    def __init__(
        self,
        base_model: nn.Module,
        ldba,
        config: LongShortConfig
    ):
        super().__init__()
        self.base_model = base_model
        self.ldba = ldba
        self.config = config
        
        # 代理关键属性到包装器（保持与原模型兼容）
        self.recurrent = getattr(base_model, 'recurrent', False)
        self.actor = getattr(base_model, 'actor', None)
        self.critic = getattr(base_model, 'critic', None)
        self.ltl_net = getattr(base_model, 'ltl_net', None)
        self.gnn_net = getattr(base_model, 'gnn_net', None)
        self.env_net = getattr(base_model, 'env_net', None)
        
        # 组件初始化
        self.automaton_tracker = None
        self.hierarchical_value_net = None
        self.multiscale_loss = None
        
        if config.enable_automaton_tracking:
            self.automaton_tracker = AutomatonStateTracker(ldba)
        
        if config.enable_hierarchical_value:
            # 获取维度（从 base_model 推断）- 直接内联以避免方法调用问题
            env_dim = 32  # 默认值
            if hasattr(base_model, 'actor'):
                actor = base_model.actor
                if hasattr(actor, 'input_features'):
                    env_dim = actor.input_features
                elif hasattr(actor, 'layers') and len(actor.layers) > 0:
                    first_layer = actor.layers[0]
                    if hasattr(first_layer, 'in_features'):
                        env_dim = first_layer.in_features
            
            formula_dim = base_model.gnn_net.hidden_dim if hasattr(base_model, 'gnn_net') else 64
            
            self.hierarchical_value_net = HierarchicalValueNetwork(
                env_state_dim=env_dim,
                formula_emb_dim=formula_dim
            )
            
            self.multiscale_loss = MultiTimescaleLoss(
                lambda_short=config.lambda_short,
                lambda_mid=config.lambda_mid,
                lambda_long=config.lambda_long
            )
        
        if config.enable_multiscale_gnn:
            # 替换 base_model 的 gnn_net（直接内联）
            if hasattr(base_model, 'gnn_net'):
                old_gnn = base_model.gnn_net
                
                # 提取原GNN配置
                input_dim = getattr(old_gnn, 'input_dim', 32)
                hidden_dim = getattr(old_gnn, 'hidden_dim', 64)
                edge_dim = getattr(old_gnn, 'edge_dim', 6)
                heads = getattr(old_gnn, 'heads', 4)
                dropout = getattr(old_gnn, 'dropout', 0.1)
                
                # 创建多尺度GNN（传递边特征与节点状态配置）
                use_edge_attr = getattr(old_gnn, 'use_edge_attr', True)
                node_state_dim = getattr(old_gnn, 'node_state_dim', 0)
                new_gnn = MultiScaleSemanticGNN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    num_layers=config.gnn_num_layers,
                    heads=heads,
                    dropout=dropout,
                    use_edge_attr=use_edge_attr,
                    node_state_dim=node_state_dim
                )
                
                # 尝试复制权重
                try:
                    if hasattr(old_gnn, 'convs') and hasattr(new_gnn, 'convs'):
                        with torch.no_grad():
                            min_layers = min(len(old_gnn.convs), len(new_gnn.convs))
                            for i in range(min_layers):
                                old_state = old_gnn.convs[i].state_dict()
                                new_state = new_gnn.convs[i].state_dict()
                                for key in old_state:
                                    if key in new_state and old_state[key].shape == new_state[key].shape:
                                        new_state[key].copy_(old_state[key])
                                new_gnn.convs[i].load_state_dict(new_state)
                    print("  [成功] 已将原GNN权重迁移至多尺度GNN")
                except Exception as e:
                    print(f"  [提示] 无法迁移GNN权重（将随机初始化）: {e}")
                
                # 替换GNN
                base_model.gnn_net = new_gnn
                self.gnn_net = new_gnn
            else:
                print("  [警告] base_model 没有 gnn_net 属性，跳过多尺度GNN替换")
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        return_multiscale: bool = False
    ):
        """
        前向传播（兼容原接口）
        
        Args:
            obs: 观测字典
            return_multiscale: 是否返回多尺度信息
        
        Returns:
            如果 return_multiscale=False: 与原模型相同的输出
            如果 return_multiscale=True: 增强输出（包含长短期信息）
        """
        # 1. 基础前向传播（直接调用，不再使用 __dict__ 回退）
        base_output = self.base_model(obs)
        
        if not return_multiscale:
            return base_output
        
        # 2. 增强信息提取
        enhanced_output = {
            'base': base_output,
            'multiscale': {}
        }
        
        # 获取配置和组件
        config = self.config
        automaton_tracker = self.automaton_tracker
        hierarchical_value_net = self.hierarchical_value_net
        
        # 自动机状态追踪
        if config.enable_automaton_tracking and automaton_tracker:
            # 从 obs 中提取命题
            props = self._extract_propositions(obs)
            automaton_states, is_accepting = automaton_tracker.step(props)
            automaton_features = automaton_tracker.get_state_features()
            
            enhanced_output['multiscale']['automaton_states'] = automaton_states
            enhanced_output['multiscale']['is_accepting'] = is_accepting
            enhanced_output['multiscale']['automaton_features'] = automaton_features
        
        # 层次化价值估计
        if config.enable_hierarchical_value and hierarchical_value_net:
            env_state = obs.get('features', obs.get('observation'))
            formula_emb = base_output.get('ltl_embedding', torch.zeros(env_state.shape[0], 64))
            automaton_features = enhanced_output['multiscale'].get(
                'automaton_features',
                torch.zeros(env_state.shape[0], 4)
            )
            
            value_dict = hierarchical_value_net(
                env_state,
                automaton_features,
                formula_emb
            )
            enhanced_output['multiscale']['hierarchical_values'] = value_dict
        
        return enhanced_output
    
    def compute_loss(
        self,
        predictions: Dict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        returns: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算多时间尺度损失
        
        Args:
            predictions: forward() 返回的增强输出
            rewards: [batch, seq_len] 奖励序列
            dones: [batch, seq_len] 终止标记
            returns: 可选的预计算回报
        
        Returns:
            损失字典
        """
        losses = {}
        
        # 基础损失（原有的 PPO/Actor-Critic 损失）
        # 这里假设 base_model 有自己的损失计算
        # losses['base_loss'] = ...
        
        # 获取组件（使用 __dict__ 直接访问）
        config = self.config
        multiscale_loss = self.multiscale_loss
        
        # 多时间尺度价值损失
        if config and config.enable_hierarchical_value and 'multiscale' in predictions:
            multiscale_info = predictions['multiscale']
            
            if 'hierarchical_values' in multiscale_info:
                v_dict = multiscale_info['hierarchical_values']
                v_preds = v_dict['v_total']
                
                # 扩展为序列
                if v_preds.dim() == 2 and v_preds.shape[1] == 1:
                    v_preds = v_preds.squeeze(1)
                
                # 计算多尺度损失  
                if multiscale_loss:
                    multiscale_losses = multiscale_loss(
                    v_preds.unsqueeze(1).expand(-1, rewards.shape[1]),
                    rewards,
                    dones
                )
                
                    losses.update({
                        'value_loss_short': multiscale_losses['loss_short'],
                        'value_loss_mid': multiscale_losses['loss_mid'],
                        'value_loss_long': multiscale_losses['loss_long'],
                        'value_loss_total': multiscale_losses['total_loss']
                    })
        
        return losses
    
    def reset_automaton_tracker(self, batch_size: int):
        """重置自动机追踪器（新 episode 开始时）"""
        if self.automaton_tracker:
            self.automaton_tracker.reset(batch_size)
    
    # 移除 __getattr__ 以避免 PyTorch 与自定义代理交互导致的属性缺失问题
    
    def _infer_env_dim(self) -> int:
        """从 base_model 推断环境状态维度"""
        # 直接使用 object.__getattribute__ 避免触发 __getattr__
        base_model = object.__getattribute__(self, 'base_model')
        
        # 尝试从 actor 输入维度推断
        if hasattr(base_model, 'actor'):
            actor = base_model.actor
            if hasattr(actor, 'input_features'):
                return actor.input_features
            # 尝试从第一层获取
            if hasattr(actor, 'layers') and len(actor.layers) > 0:
                first_layer = actor.layers[0]
                if hasattr(first_layer, 'in_features'):
                    return first_layer.in_features
        
        # 默认值（PointLtl2-v0 环境）
        return 32
    
    def _extract_propositions(self, obs: Dict) -> torch.Tensor:
        """从观测中提取命题真值"""
        # 1. 直接从观测字典获取
        if 'propositions' in obs:
            props = obs['propositions']
            if isinstance(props, torch.Tensor):
                return props
            return torch.tensor(props, dtype=torch.float32)
        
        # 2. 尝试从 'ltl_info' 或类似字段提取
        if 'ltl_info' in obs and 'propositions' in obs['ltl_info']:
            props = obs['ltl_info']['propositions']
            if isinstance(props, torch.Tensor):
                return props
            return torch.tensor(props, dtype=torch.float32)
        
        # 3. 返回零向量（无命题满足）- 兜底方案
        batch_size = obs.get('features', obs.get('observation')).shape[0]
        device = obs.get('features', obs.get('observation')).device
        ldba = object.__getattribute__(self, 'ldba')
        num_props = len(ldba.propositions) if hasattr(ldba, 'propositions') else 4
        return torch.zeros(batch_size, num_props, dtype=torch.float32, device=device)
    
    def _replace_gnn_with_multiscale(self):
        """用多尺度GNN替换原有GNN"""
        base_model = object.__getattribute__(self, 'base_model')
        
        if not hasattr(base_model, 'gnn_net'):
            print("  [警告] base_model 没有 gnn_net 属性，跳过多尺度GNN替换")
            return
        
        old_gnn = base_model.gnn_net
        
        # 提取原GNN配置
        input_dim = getattr(old_gnn, 'input_dim', 32)
        hidden_dim = getattr(old_gnn, 'hidden_dim', 64)
        edge_dim = getattr(old_gnn, 'edge_dim', 6)
        heads = getattr(old_gnn, 'heads', 4)
        dropout = getattr(old_gnn, 'dropout', 0.1)
        
        # 获取 config（使用 object.__getattribute__）
        config = object.__getattribute__(self, 'config')
        
        # 创建多尺度GNN（保持相同配置）
        use_edge_attr = getattr(old_gnn, 'use_edge_attr', True)
        node_state_dim = getattr(old_gnn, 'node_state_dim', 0)
        new_gnn = MultiScaleSemanticGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_layers=config.gnn_num_layers,
            heads=heads,
            dropout=dropout,
            use_edge_attr=use_edge_attr,
            node_state_dim=node_state_dim
        )
        
        # 尝试复制权重（如果结构兼容）
        try:
            if hasattr(old_gnn, 'convs') and hasattr(new_gnn, 'convs'):
                with torch.no_grad():
                    min_layers = min(len(old_gnn.convs), len(new_gnn.convs))
                    for i in range(min_layers):
                        old_state = old_gnn.convs[i].state_dict()
                        new_state = new_gnn.convs[i].state_dict()
                        # 复制形状匹配的参数
                        for key in old_state:
                            if key in new_state and old_state[key].shape == new_state[key].shape:
                                new_state[key].copy_(old_state[key])
                        new_gnn.convs[i].load_state_dict(new_state)
            print("  [成功] 已将原GNN权重迁移至多尺度GNN")
        except Exception as e:
            print(f"  [提示] 无法迁移GNN权重（将随机初始化）: {e}")
        
        # 使用 object.__getattribute__ 安全访问 base_model
        base_model = object.__getattribute__(self, 'base_model')
        base_model.gnn_net = new_gnn
        # 同步到包装器属性
        object.__setattr__(self, 'gnn_net', new_gnn)


def enhance_model_with_multiscale(
    model: nn.Module,
    ldba,
    config: Optional[LongShortConfig] = None,
    enable_hierarchical_value: bool = True,
    enable_multiscale_gnn: bool = True,
    enable_automaton_tracking: bool = True,
    enable_options: bool = False,
    **kwargs
) -> LongShortEnhancedModel:
    """
    便捷函数：增强现有模型
    
    Args:
        model: 原始 deep-ltl 模型
        ldba: LDBA 对象
        config: LongShortConfig 对象（如果提供则优先使用）
        enable_*: 各组件开关（当 config=None 时使用）
        **kwargs: 其他配置参数
    
    Returns:
        增强后的模型
    """
    if config is None:
        config = LongShortConfig(
            enable_hierarchical_value=enable_hierarchical_value,
            enable_multiscale_gnn=enable_multiscale_gnn,
            enable_automaton_tracking=enable_automaton_tracking,
            enable_options=enable_options,
            **kwargs
        )
    
    enhanced_model = LongShortEnhancedModel(model, ldba, config)
    
    print(f"✓ 模型已增强为长短期RL架构")
    print(f"  - 层次化价值: {'启用' if config.enable_hierarchical_value else '禁用'}")
    print(f"  - 多尺度GNN: {'启用' if config.enable_multiscale_gnn else '禁用'}")
    print(f"  - 自动机追踪: {'启用' if config.enable_automaton_tracking else '禁用'}")
    print(f"  - Options框架: {'启用' if config.enable_options else '禁用'}")
    
    return enhanced_model


def load_checkpoint_compatible(model: nn.Module, state_dict: dict, strict: bool = False) -> None:
    """
    向后兼容的 checkpoint 加载器
    
    处理两种情况：
    1. 基础模型 checkpoint -> 加载到 base_model
    2. 增强模型 checkpoint -> 直接加载
    
    Args:
        model: 当前模型（可能是基础模型或增强模型）
        state_dict: 保存的模型权重
        strict: 是否严格匹配（默认 False）
    """
    try:
        # 尝试直接加载（增强模型 checkpoint）
        model.load_state_dict(state_dict, strict=strict)
        print("[Checkpoint] 成功加载增强模型 checkpoint")
    except RuntimeError as e:
        # 检查是否是基础模型的 checkpoint
        if isinstance(model, LongShortEnhancedModel):
            # 尝试加载到 base_model
            try:
                model.base_model.load_state_dict(state_dict, strict=False)
                print("[Checkpoint] 检测到基础模型 checkpoint，已加载到 base_model（增强组件将重新训练）")
            except RuntimeError as e2:
                raise RuntimeError(
                    f"无法加载 checkpoint：\n"
                    f"  - 作为增强模型加载失败: {str(e)[:100]}\n"
                    f"  - 作为基础模型加载失败: {str(e2)[:100]}"
                )
        else:
            # 普通模型无法加载增强模型的 checkpoint
            raise RuntimeError(
                f"Checkpoint 格式不匹配。可能原因：\n"
                f"  1. Checkpoint 来自增强模型，但当前未启用增强\n"
                f"  2. 模型结构发生变化\n"
                f"  错误详情: {str(e)[:200]}"
            )


# 使用示例
if __name__ == "__main__":
    print("=== 长短期RL集成示例 ===\n")
    print("集成方式：")
    print("```python")
    print("# 1. 构建原始模型")
    print("model = build_model(config)")
    print("")
    print("# 2. 增强为长短期架构")
    print("enhanced_model = enhance_model_with_multiscale(")
    print("    model, ldba,")
    print("    enable_hierarchical_value=True,")
    print("    enable_multiscale_gnn=True,")
    print("    enable_automaton_tracking=True")
    print(")")
    print("")
    print("# 3. 训练时使用")
    print("output = enhanced_model(obs, return_multiscale=True)")
    print("losses = enhanced_model.compute_loss(output, rewards, dones)")
    print("```")
