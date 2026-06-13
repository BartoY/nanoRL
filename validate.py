import torch
from utils import fjsp_sched_bch
import numpy as np


def validate_model(model, val_loader, device, n_j, n_m, max_op):
    """
    在验证集上评估模型，并返回所有样本的 Makespan 数组
    """
    model.eval()  # 切换到评估模式
    all_costs = []

    with torch.no_grad():  # 禁用梯度计算，节省显存和时间
        for batch in val_loader:
            batch = batch.to(device)

            # 维度处理
            bsz = batch.num_graphs
            n_node = n_j * max_op
            op_proc_time = batch.proc_times.view(bsz, n_node, n_m).float()
            mask_machine_compat = batch.compat_mask.view(bsz, n_node, n_m).bool()

            job_length = batch.job_length.view(bsz, n_j)

            job_seq, mach_assign, *_ = model(
                batch,
                mask_machine_compat,
                op_proc_time,
                job_length,
                rollout=True  # 验证集使用 Rollout 贪婪策略，不使用随机采样
            )

            _, costs = fjsp_sched_bch(
                job_sequence=job_seq,
                mach_sequence=mach_assign,
                proc_times=op_proc_time,
                n_j=n_j,
                n_m=n_m,
                n_op=max_op,
                job_length=job_length)

            all_costs.append(costs.cpu().numpy())

    return np.concatenate(all_costs)