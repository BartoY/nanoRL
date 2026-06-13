import torch

def convert_state_to_pyg_data(adj, fea, n_j, n_m, machines_np):
    """
    将 JSSP 环境的状态转换为PyG的Data对象

    Args:
        adj: 邻接矩阵 [n_tasks, n_tasks] (包含了工序先后约束)
        fea: 节点特征
        machines_np: 机器分配矩阵 [n_j, n_m], machines_np[j, s] 表示第 j 个作业第 s 道工序使用的机器 ID
        n_j: 作业数
        n_m: 机器数

    Returns:
        PyG Data 对象
    """
    from torch_geometric.data import Data

    # 处理基础特征和工序约束边，转换为 torch tensor
    adj_tensor = torch.from_numpy(adj).float()
    fea_tensor = torch.from_numpy(fea).float()

    # 从邻接矩阵提取边索引
    edge_index_conj = (adj_tensor > 0).nonzero(as_tuple=False).t().contiguous().long()

    # 构建机器边 (Disjunctive Edges) - 全连接/双向
    machine_groups = {}  # {machine_id: [node_idx1, node_idx2, ...]}

    for j in range(n_j):
        for s in range(n_m):
            # 获取当前工序对应的机器ID
            m_id = machines_np[j, s]
            # 获取当前工序在图中的全局索引
            node_idx = j * n_m + s

            if m_id not in machine_groups:
                machine_groups[m_id] = []
            machine_groups[m_id].append(node_idx)

    src_list = []
    dst_list = []

    # 对每一台机器，构建全连接 (Clique)
    # 也就是：该机器上的每一个工序都指向该机器上的其他所有工序
    for m_id, nodes in machine_groups.items():
        # nodes 是一个列表，例如 [0, 15, 23, ...]
        for u in nodes:
            for v in nodes:
                if u != v:  # 避免自环 (通常GAT会自己加自环)
                    src_list.append(u)
                    dst_list.append(v)
                    # 这样就构成了 u->v 和 v->u (双向)

    if len(src_list) > 0:
        edge_index_mach = torch.tensor([src_list, dst_list], dtype=torch.long)
        # 3. 合并两种边
        edge_index = torch.cat([edge_index_conj, edge_index_mach], dim=1)
    else:
        edge_index = edge_index_conj

    return Data(x=fea_tensor, edge_index=edge_index)


if __name__ == "__main__":
    # "D:\Desktop\RL\RL-JSSP\models_save\\best_model.pth"
    with open(".\models_save\\best_model2.txt", 'w') as file:
        file.write('Hello, world!')
    # from uniform_instance_gen import uni_instance_gen
    # from JSSP_Env import SJSSP
    # from data_utils import get_initial_adj_and_fea
    #
    # env = SJSSP(n_j=10, n_m=10)
    # times, machines = uni_instance_gen(n_j=10, n_m=10, low=0, high=1)
    # instance_data = (times, machines)
    # # print(instance_data)
    # adj, fea, _, _ = env.reset(instance_data)
    # print(adj, fea)
    # a1,f =  get_initial_adj_and_fea(10,10,instance_data)
    # print(a1, f)

    # 打开一个文件用于写入。如果文件不存在，将会被创建

