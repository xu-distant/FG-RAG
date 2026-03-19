
from torch_geometric.data import Data
import numpy as np

import torch
def generate_random_graph(min_nodes=5, max_nodes=20, num_features=5, edge_prob=0.3):
    """
    生成一个随机图，节点数在给定范围内随机选择。

    参数:
    min_nodes (int): 最小节点数
    max_nodes (int): 最大节点数
    num_features (int): 每个节点的特征数量
    edge_prob (float): 两个节点之间存在边的概率

    返回:
    Data: PyTorch Geometric的Data对象，表示生成的随机图
    """
    # 随机选择节点数量
    num_nodes = np.random.randint(min_nodes, max_nodes + 1)

    # 生成随机节点特征
    x = torch.randn(num_nodes, num_features)

    # 生成随机边
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.random() < edge_prob:
                # 添加双向边
                edge_index.append([i, j])
                edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 创建Data对象
    data = Data(x=x, edge_index=edge_index)

    return data