import torch
from torch_geometric.data import Data, Batch
from preprocessing.vocab import VOCAB

def preprocess_ltl_to_graph(ltl_tokens):
    """
    将LTL tokens转换为图结构
    
    Args:
        ltl_tokens: LTL公式的token序列
        
    Returns:
        Data: PyTorch Geometric的Data对象，包含图结构信息
    """
    # 创建节点特征
    x = torch.tensor([VOCAB[token] for token in ltl_tokens], dtype=torch.long)
    
    # 创建边连接关系（这里使用简单的顺序连接）
    edge_index = []
    num_nodes = len(ltl_tokens)
    
    # 创建基本的顺序连接
    for i in range(num_nodes-1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])  # 双向边
        
    # 添加特殊的语义连接
    # 例如：连接操作符与其参数
    for i, token in enumerate(ltl_tokens):
        if token in ['AND', 'OR', 'UNTIL']:
            # 连接操作符与其后的两个参数
            if i + 2 < num_nodes:
                edge_index.append([i, i+1])
                edge_index.append([i, i+2])
                edge_index.append([i+1, i])
                edge_index.append([i+2, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index)
    return data

def preprocess_batch_to_graphs(batch_ltl_tokens):
    """
    处理批量的LTL公式
    
    Args:
        batch_ltl_tokens: 批量的LTL token序列
        
    Returns:
        Batch: PyTorch Geometric的Batch对象
    """
    graphs = [preprocess_ltl_to_graph(tokens) for tokens in batch_ltl_tokens]
    return Batch.from_data_list(graphs)