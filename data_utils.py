from uniform_instance_gen import uni_instance_gen
import torch
import numpy as np
from Params import configs
from multiprocessing import Pool
from functools import partial
from torch_geometric.data import HeteroData


def get_initial_input(n_j, n_m, max_n_op, proc_times, compat_mask, job_length):
    number_of_tasks = n_j * max_n_op
    # --- Adjacency Matrix ---
    first_col = np.arange(start=0, stop=number_of_tasks, step=max_n_op)
    conj_nei_up_stream = np.eye(number_of_tasks, k=-1, dtype=np.single)
    conj_nei_up_stream[first_col] = 0

    for j in range(n_j):
        if job_length[j] < max_n_op:
            dummy_start_idx = j * max_n_op + job_length[j]
            conj_nei_up_stream[dummy_start_idx] = 0

    self_as_nei = np.eye(number_of_tasks, dtype=np.single)
    adj = self_as_nei + conj_nei_up_stream

    # --- Feature Matrix ---
    mask_flat_float = compat_mask.reshape(number_of_tasks, n_m).astype(np.single)
    times_flat = proc_times.reshape(number_of_tasks, n_m)

    # 平均加工时间
    est_dur_flat = np.sum(times_flat * mask_flat_float, axis=1) / np.maximum(np.sum(mask_flat_float, axis=1), 1)
    est_dur = est_dur_flat.reshape(n_j, max_n_op)

    # 最短加工时间
    times_masked_inf = np.where(mask_flat_float > 0, times_flat, np.inf)
    min_pt_flat = np.min(times_masked_inf, axis=1)
    min_pt_flat[min_pt_flat == np.inf] = 0.0

    # 机器柔性度
    flex_ratio_flat = np.sum(mask_flat_float, axis=1) / n_m

    # 预估剩余总时间
    est_remain_pt = np.cumsum(est_dur[:, ::-1], axis=1)[:, ::-1]
    # 剩余工序数比例
    ops_left = np.zeros((n_j, max_n_op), dtype=np.single)
    ops_left_ratio = np.zeros((n_j, max_n_op), dtype=np.single)
    for j in range(n_j):
        if job_length[j] > 0:
            ops_left[j, :job_length[j]] = np.arange(job_length[j], 0, -1)
            ops_left_ratio[j, :job_length[j]] = ops_left[j, :job_length[j]] / job_length[j]

    LBs = np.cumsum(est_dur, axis=1, dtype=np.single)
    fea = np.concatenate((
        LBs.reshape(-1, 1) / configs.et_normalize_coef,
        est_remain_pt.reshape(-1, 1) / configs.et_normalize_coef,
        ops_left_ratio.reshape(-1, 1).astype(np.single),
        est_dur_flat.reshape(-1, 1) / configs.et_normalize_coef,
        min_pt_flat.reshape(-1, 1) / configs.et_normalize_coef,
        flex_ratio_flat.reshape(-1, 1).astype(np.single)
    ), axis=1)

    mask_flat = compat_mask.reshape(number_of_tasks, n_m)

    mach_fea = np.zeros((n_m, 3), dtype=np.single)
    for m in range(n_m):
        comp_ops = np.where(mask_flat[:, m] == True)[0]
        if len(comp_ops) > 0:
            # 机器柔性度
            mach_fea[m, 0] = len(comp_ops) / number_of_tasks
            # 机器潜在负载潜力
            mach_fea[m, 1] = np.mean(times_flat[comp_ops, m]) / configs.et_normalize_coef
            # 机器相对效率，当前机器时间 / 所有兼容该工序的机器平均时间
            op_mean_times = np.sum(times_flat[comp_ops, :] * mask_flat_float[comp_ops, :], axis=1) / \
                            np.maximum(np.sum(mask_flat_float[comp_ops, :], axis=1), 1)
            mach_fea[m, 2] = np.mean(times_flat[comp_ops, m] / (op_mean_times + 1e-5))

    return adj, fea, mach_fea


def convert_to_pyg_data(adj_conj, fea, mach_fea, proc_times, compat_mask, n_m, job_length):
    """
    将FJSP环境的状态转换为PyG的HeteroData对象
    """
    data = HeteroData()

    # 定义节点特征
    data['operation'].x = torch.from_numpy(fea).float()
    data['machine'].x = torch.from_numpy(mach_fea).float()

    # 同工件的先后约束边
    adj_tensor = torch.from_numpy(adj_conj).float()
    edge_index_conj = (adj_tensor == 1).nonzero(as_tuple=False).t().contiguous().long()
    data['operation', 'precedes', 'operation'].edge_index = edge_index_conj
    data['operation', 'precedes', 'operation'].edge_attr = torch.zeros((edge_index_conj.shape[1], 3), dtype=torch.float)

    # 工序与机器的兼容边
    number_of_tasks = proc_times.shape[0] * proc_times.shape[1]
    mask_flat = compat_mask.reshape(number_of_tasks, n_m)
    times_flat = proc_times.reshape(number_of_tasks, n_m)

    op_indices, mach_indices = np.where(mask_flat == True)
    edge_index_compat = torch.tensor(np.vstack((op_indices, mach_indices)), dtype=torch.long)
    data['operation', 'compatible_with', 'machine'].edge_index = edge_index_compat

    mask_flat_float = mask_flat.astype(np.single)
    pt_absolute = times_flat[op_indices, mach_indices]
    # 获取每个相关工序的 min 和 mean 加工时间
    times_masked_inf = np.where(mask_flat_float > 0, times_flat, np.inf)
    pt_min_all = np.min(times_masked_inf, axis=1)  # 每个工序的最短时间

    pt_sum_all = np.sum(times_flat * mask_flat_float, axis=1)
    pt_count_all = np.maximum(np.sum(mask_flat_float, axis=1), 1.0)
    pt_mean_all = pt_sum_all / pt_count_all  # 每个工序的平均时间

    # 映射回边列表对应的维度
    pt_min_edges = pt_min_all[op_indices]
    pt_mean_edges = pt_mean_all[op_indices]

    # 拼接为3列特征：[绝对时间, 最优比例, 平均比例]
    edge_attr_compat_np = np.column_stack((
        pt_absolute / configs.et_normalize_coef,
        pt_absolute / (pt_min_edges + 1e-5),  # 若等于最优机器，值为1；否则 > 1
        pt_absolute / (pt_mean_edges + 1e-5)  # 相对平均水平的快慢
    ))

    edge_attr_compat = torch.tensor(edge_attr_compat_np, dtype=torch.float)
    data['operation', 'compatible_with', 'machine'].edge_attr = edge_attr_compat

    data['machine', 'processed_by', 'operation'].edge_index = torch.flip(edge_index_compat, dims=[0])
    data['machine', 'processed_by', 'operation'].edge_attr = edge_attr_compat

    data.proc_times = torch.from_numpy(proc_times).view(-1, n_m).float()
    data.compat_mask = torch.from_numpy(compat_mask).view(-1, n_m).bool()

    data.job_length = torch.tensor(job_length, dtype=torch.long)

    return data


def _generate_single_instance(idx, n_j, n_m, min_op, max_op, return_pyg=False):
    proc_times, compat_mask, job_length = uni_instance_gen(n_j, n_m, min_op, max_op, 0.01, 1.0)

    adj, fea, mach_fea = get_initial_input(n_j, n_m, max_op, proc_times, compat_mask, job_length)
    if return_pyg:
        return convert_to_pyg_data(adj, fea, mach_fea, proc_times, compat_mask, n_m, job_length)

    return adj, fea, mach_fea, proc_times, compat_mask, job_length


def epoch_dataset_gen(n_samples, n_j, n_m, min_op, max_op):
    func = partial(_generate_single_instance, n_j=n_j, n_m=n_m, min_op=min_op, max_op=max_op, return_pyg=False)
    with Pool() as pool:
        numpy_results = pool.map(func, range(n_samples))

    data_list = []
    for adj_np, fea_np, mach_fea_np, proc_times_np, compat_mask_np, job_length_np in numpy_results:
        data = convert_to_pyg_data(adj_np, fea_np, mach_fea_np, proc_times_np, compat_mask_np, n_m, job_length_np)
        data_list.append(data)
    return data_list
