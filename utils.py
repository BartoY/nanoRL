import torch
from torch_geometric.data import Data, Dataset
from scipy.stats import ttest_rel


def chk_upd_bl(policy_costs, baseline_costs, alpha=0.05):
    """
    使用配对T检验判断Policy是否显著优于Baseline

    Args:
        policy_costs: 当前模型的验证集Makespan数组
        baseline_costs: Baseline模型的验证集Makespan数组
        alpha: 显著性水平

    Returns:
        bool: 是否应该更新Baseline
    """
    if torch.is_tensor(policy_costs):
        policy_costs = policy_costs.detach().cpu().numpy()
    if torch.is_tensor(baseline_costs):
        baseline_costs = baseline_costs.detach().cpu().numpy()

    # 如果当前模型均值比 Baseline 还差，直接不更新，省去 T-test
    if policy_costs.mean() >= baseline_costs.mean():
        return False

    # 2. 配对 T 检验 (Paired T-test)
    # H0: 两个模型均值相同
    # H1: 两个模型均值不同
    t_stat, p_value = ttest_rel(baseline_costs, policy_costs)

    if t_stat > 0 and (p_value / 2) < alpha:
        return True

    return False


def fjsp_sched_bch(job_sequence, mach_sequence, proc_times, n_j, n_m, n_op, job_length=None):
    """
     将 FJSP 模型生成的序列和机器分配方案转换为具体调度并计算 Makespan。

    Args:
        job_sequence:  [B, Total_Ops] (LongTensor) 高层策略每步选出的 Job ID
        mach_sequence: [B, Total_Ops] (LongTensor) 底层策略每步为该 Job 选出的机器 ID
        proc_times:    [B, Total_Ops, n_m] (FloatTensor) 3D 加工时间张量
        n_j, n_m, n_op:      int, int, int

    Returns:
        batch_schedules: List[Dict] 长度为 B。{machine_id: [task_info, ...]}
        batch_makespans: Tensor [B] 每个样本的最终完工时间
    """

    # 1. 预处理：将 Tensor 转移到 CPU 并转为 Numpy，方便构建字典结构
    seq_np = job_sequence.detach().cpu().numpy()
    mach_np = mach_sequence.detach().cpu().numpy()
    dur_np = proc_times.detach().cpu().numpy()

    if job_length is not None:
        if torch.is_tensor(job_length):
            jl_np = job_length.detach().cpu().numpy()
        else:
            jl_np = job_length
    else:
        import numpy as np
        jl_np = np.full((seq_np.shape[0], n_j), n_op)

    batch_size = seq_np.shape[0]
    total_ops = seq_np.shape[1]

    batch_schedules = []
    batch_makespans = []

    # 2. 遍历 Batch 中的每一个样本
    for b in range(batch_size):
        # 初始化当前样本的状态
        machine_intervals = {m: [] for m in range(n_m)}
        job_free_time = {j: 0.0 for j in range(n_j)}
        job_op_idx = {j: 0 for j in range(n_j)}
        current_schedule = {m: [] for m in range(n_m)}

        # --- 单个样本的调度模拟 ---
        for i in range(total_ops):
            job_id = seq_np[b, i]
            m_id = mach_np[b, i]
            op_k = job_op_idx[job_id]

            # 防止越界
            actual_len = jl_np[b, job_id]
            if op_k >= actual_len:
                continue

            global_node_idx = job_id * n_op + op_k
            d = dur_np[b, global_node_idx, m_id]
            ready_t = job_free_time[job_id]

            # 寻找机器上可以插入的最早空隙
            intervals = machine_intervals[m_id]
            intervals.sort(key=lambda x: x[0])

            start_time = ready_t
            for idx_int, (s, e) in enumerate(intervals):
                if start_time + d <= s:
                    # 找到了一个完美的缝隙
                    break
                start_time = max(start_time, e)

            end_time = start_time + d

            machine_intervals[m_id].append((start_time, end_time))
            job_free_time[job_id] = end_time
            job_op_idx[job_id] += 1

            # --- 记录到调度表 ---
            current_schedule[m_id].append({
                'job': int(job_id),
                'op_idx': int(op_k),
                'start': float(start_time),
                'end': float(end_time),
                'duration': float(d)
            })

        # 计算当前样本的 Makespan
        makespan = 0
        for m, intervals in machine_intervals.items():
            if intervals:
                makespan = max(makespan, max(e for s, e in intervals))

        batch_schedules.append(current_schedule)
        batch_makespans.append(makespan)

    # 将 Makespan 转回 Tensor
    return batch_schedules, torch.tensor(batch_makespans, device=job_sequence.device, dtype=torch.float32)