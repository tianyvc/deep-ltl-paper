import torch
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple, Dict, Optional
from ltl.automata import LDBA

class GraphDataProcessor:
    """图数据增强与预处理工具"""
    
    @staticmethod
    def add_virtual_nodes(data: Data, num_virtual: int = 2) -> Data:
        """添加虚拟节点以增强图的全局信息传递"""
        num_nodes = data.x.size(0)
        device = data.x.device
        
        # 创建虚拟节点特征
        virtual_x = torch.zeros((num_virtual, data.x.size(1)), device=device)
        virtual_x[:, -1] = 1.0  # 标记为虚拟节点
        
        # 扩展节点特征
        x = torch.cat([data.x, virtual_x], dim=0)
        
        # 连接虚拟节点到所有其他节点
        virtual_edges = []
        for v in range(num_virtual):
            v_idx = num_nodes + v
            for i in range(num_nodes):
                virtual_edges.extend([[v_idx, i], [i, v_idx]])
                
        # 扩展边索引
        edge_index = torch.cat([
            data.edge_index,
            torch.tensor(virtual_edges, dtype=torch.long, device=device).t()
        ], dim=1)
        
        # 扩展边特征（如果存在）
        if hasattr(data, 'edge_attr'):
            virtual_edge_attr = torch.zeros((len(virtual_edges), data.edge_attr.size(1)), device=device)
            virtual_edge_attr[:, -1] = 1.0  # 标记为虚拟边
            edge_attr = torch.cat([data.edge_attr, virtual_edge_attr], dim=0)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        return Data(x=x, edge_index=edge_index)
    
    @staticmethod
    def add_subgraph_edges(data: Data, subgraph_size: int = 3, overlap: int = 1) -> Data:
        """添加子图级别的边连接以捕获局部结构"""
        num_nodes = data.x.size(0)
        device = data.x.device
        
        # 创建滑动窗口的子图连接
        subgraph_edges = []
        for i in range(0, num_nodes - subgraph_size + 1, subgraph_size - overlap):
            nodes = list(range(i, min(i + subgraph_size, num_nodes)))
            # 子图内部完全连接
            for n1 in nodes:
                for n2 in nodes:
                    if n1 != n2:
                        subgraph_edges.extend([[n1, n2]])
                        
        # 添加新边
        if len(subgraph_edges) > 0:
            new_edges = torch.tensor(subgraph_edges, dtype=torch.long, device=device).t()
            edge_index = torch.cat([data.edge_index, new_edges], dim=1)
            
            if hasattr(data, 'edge_attr'):
                subgraph_attr = torch.zeros((len(subgraph_edges), data.edge_attr.size(1)), device=device)
                subgraph_attr[:, -2] = 1.0  # 标记为子图边
                edge_attr = torch.cat([data.edge_attr, subgraph_attr], dim=0)
                return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr)
                
            return Data(x=data.x, edge_index=edge_index)
        return data
    
    @staticmethod
    def add_semantic_edges(data: Data, 
                          semantic_info: Dict[int, List[str]], 
                          similarity_threshold: float = 0.7) -> Data:
        """基于语义信息添加边"""
        device = data.x.device
        semantic_edges = []
        
        # 计算节点间的语义相似度并添加边
        for i, info_i in semantic_info.items():
            for j, info_j in semantic_info.items():
                if i >= j:
                    continue
                    
                # 计算语义相似度（示例：使用共同元素比例）
                common = len(set(info_i) & set(info_j))
                total = len(set(info_i) | set(info_j))
                similarity = common / total if total > 0 else 0
                
                if similarity > similarity_threshold:
                    semantic_edges.extend([[i, j], [j, i]])
                    
        # 添加新边
        if len(semantic_edges) > 0:
            new_edges = torch.tensor(semantic_edges, dtype=torch.long, device=device).t()
            edge_index = torch.cat([data.edge_index, new_edges], dim=1)
            
            if hasattr(data, 'edge_attr'):
                semantic_attr = torch.zeros((len(semantic_edges), data.edge_attr.size(1)), device=device)
                semantic_attr[:, -3] = 1.0  # 标记为语义边
                edge_attr = torch.cat([data.edge_attr, semantic_attr], dim=0)
                return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr)
                
            return Data(x=data.x, edge_index=edge_index)
        return data
        
    @staticmethod
    def augment_features(data: Data, 
                        degree_centrality: bool = True,
                        closeness_centrality: bool = True,
                        clustering_coeff: bool = True) -> Data:
        """使用图论特征增强节点表示"""
        num_nodes = data.x.size(0)
        device = data.x.device
        new_features = []
        
        if degree_centrality:
            # 计算度中心性
            edge_index = data.edge_index
            degrees = torch.zeros(num_nodes, device=device)
            unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
            degrees[unique_nodes] = counts.float()
            degrees = degrees / (num_nodes - 1)  # 归一化
            new_features.append(degrees.unsqueeze(1))
            
        if closeness_centrality:
            # 近似计算接近中心性（使用随机游走）
            walks = []
            num_walks = 10
            walk_length = 3
            for _ in range(num_walks):
                current = torch.randint(0, num_nodes, (1,), device=device)
                walk = [current]
                for _ in range(walk_length):
                    neighbors = data.edge_index[1][data.edge_index[0] == current]
                    if len(neighbors) > 0:
                        current = neighbors[torch.randint(0, len(neighbors), (1,))]
                        walk.append(current)
                walks.append(torch.cat(walk))
            closeness = torch.tensor([len(set(w.tolist())) for w in walks], device=device).float()
            closeness = closeness / num_nodes
            new_features.append(closeness.mean().unsqueeze(0).repeat(num_nodes, 1))
            
        if clustering_coeff:
            # 计算聚类系数
            edge_index = data.edge_index
            clustering = torch.zeros(num_nodes, device=device)
            for node in range(num_nodes):
                neighbors = edge_index[1][edge_index[0] == node]
                if len(neighbors) < 2:
                    continue
                neighbor_edges = 0
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if any((edge_index[0] == neighbors[i]) & (edge_index[1] == neighbors[j])):
                            neighbor_edges += 1
                possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                clustering[node] = neighbor_edges / possible_edges if possible_edges > 0 else 0
            new_features.append(clustering.unsqueeze(1))
            
        # 合并新特征
        if new_features:
            additional_features = torch.cat(new_features, dim=1)
            x = torch.cat([data.x, additional_features], dim=1)
            return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None)
            
        return data
        
    @classmethod
    def process_graph(cls, 
                     data: Data,
                     semantic_info: Optional[Dict[int, List[str]]] = None,
                     add_virtual: bool = True,
                     add_subgraph: bool = True,
                     add_features: bool = True) -> Data:
        """应用完整的图处理流程"""
        if add_virtual:
            data = cls.add_virtual_nodes(data)
        if add_subgraph:
            data = cls.add_subgraph_edges(data)
        if semantic_info is not None:
            data = cls.add_semantic_edges(data, semantic_info)
        if add_features:
            data = cls.augment_features(data)
        return data