#!/bin/bash
# GNN 增强功能快速测试脚本

echo "======================================================================"
echo "GNN 增强功能测试"
echo "======================================================================"
echo ""

# 环境检查
echo "1. 检查PyTorch Geometric扩展..."
python -c "
import torch
try:
    from torch_scatter import scatter_add
    print('✓ torch-scatter 已安装')
except:
    print('✗ torch-scatter 未安装，需要安装加速扩展')
    print('  运行: pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html')
"

echo ""
echo "2. 测试基础GNN配置（仅链式边）..."
PYTHONPATH=src/ python src/evaluation/simulate.py \
  --env PointLtl2-v0 \
  --exp gnn_baseline \
  --seed 1 \
  --formula '(!blue U green) & F yellow' \
  --finite \
  --deterministic 2>&1 | tail -5

echo ""
echo "======================================================================"
echo "基础测试完成！"
echo ""
echo "后续步骤："
echo "1. 查看 docs/GNN_Enhancement_Guide.md 了解增强功能详情"
echo "2. 修改配置启用边特征和节点状态特征"
echo "3. 运行 python test_enhanced_gnn.py 进行完整对比测试"
echo "======================================================================"
