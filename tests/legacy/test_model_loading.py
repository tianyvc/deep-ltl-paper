import sys
sys.path.insert(0, 'src')

import torch
from utils.model_store import ModelStore
from config import model_configs
from model.model import build_model
from envs import make_env
from ltl import FixedSampler

print("=" * 60)
print("检查模型加载问题")
print("=" * 60)

env_name = 'PointLtl2-v0'
config_name = 'test'
seed = 1

# 创建环境
sampler = FixedSampler.partial('F blue & F green')
env = make_env(env_name, sampler, render_mode=None)

# 加载配置和训练状态
config = model_configs[env_name]
model_store = ModelStore(env_name, config_name, seed)

print(f"\n1. 加载训练状态...")
training_status = model_store.load_training_status(map_location='cpu')
print(f"   训练状态键值: {list(training_status.keys())}")
print(f"   是否有 'model_state': {'model_state' in training_status}")

if 'model_state' in training_status:
    model_state = training_status['model_state']
    print(f"   模型状态中的层数: {len(model_state)}")
    print(f"   示例层: {list(model_state.keys())[:5]}")

print(f"\n2. 加载词汇表...")
model_store.load_vocab()
print(f"   词汇表大小: {len(model_store.vocab)}")

print(f"\n3. 构建模型...")
model = build_model(env, training_status, config)
print(f"   模型构建成功")
print(f"   模型参数总数: {sum(p.numel() for p in model.parameters())}")

# 检查是否模型权重被加载
print(f"\n4. 验证权重加载...")
if 'model_state' in training_status:
    # 比较一个权重参数
    first_param_key = list(training_status['model_state'].keys())[0]
    saved_weight = training_status['model_state'][first_param_key]
    
    # 获取对应的模型参数
    model_key_parts = first_param_key.split('.')
    model_param = model
    for part in model_key_parts:
        if part.isdigit():
            model_param = model_param[int(part)]
        else:
            model_param = getattr(model_param, part)
    
    # 比较
    if torch.allclose(model_param, saved_weight, atol=1e-6):
        print(f"   ✓ 权重已正确加载")
    else:
        print(f"   ✗ 权重不匹配！")
        print(f"     应该检查权重加载逻辑")
else:
    print(f"   ✗ 没有保存的权重！")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
