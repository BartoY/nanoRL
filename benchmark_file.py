import torch
import numpy as np
import pandas as pd
import time
import os
from torch_geometric.loader import DataLoader

from model import FJSPActor
from utils import fjsp_sched_bch
from data_utils import convert_to_pyg_data, get_initial_input, _generate_single_instance

from ortools_solver import fjsp_solver

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_J = 10
N_M = 10
MIN_OP = 6
MAX_OP = 10
BATCH_SIZE = 128
seed = 200
OR_TOOLS_LIMIT = 1800.0

NUM_SAMPLES = 512

FILENAME = os.path.join("data_gen", f"GenData_{N_J}_{N_M}_{MAX_OP}_Seed{seed}_bsz{BATCH_SIZE}.npz")


def run_benchmark():
    print(f"Loading data from {FILENAME}...")
    if not os.path.exists(FILENAME):
        print(f"Error: 文件 {FILENAME} 不存在！请检查路径。")
        return

    # 1. 加载 .npz 数据
    data_archive = np.load(FILENAME)
    all_proc_times = data_archive['proc_times']  # [Batch, n_j, n_op, n_m]
    all_compat_masks = data_archive['compat_masks']  # [Batch, n_j, n_op, n_m]

    if 'job_length' in data_archive:
        all_job_lengths = data_archive['job_length']
    else:
        all_job_lengths = np.full((all_proc_times.shape[0], N_J), MAX_OP)

    batch_size = all_proc_times.shape[0]
    n_op = all_proc_times.shape[2]  # 获取工序数
    print(f"Data shape: Proc={all_proc_times.shape}, Mask={all_compat_masks.shape} (Batch Size: {batch_size})")

    # 2. 准备 RL 模型 (改用 FJSPActor)
    dummy_data = _generate_single_instance(0, N_J, N_M, MIN_OP,MAX_OP, return_pyg=True)
    metadata = dummy_data.metadata()
    model = FJSPActor(op_input_dim=6, mach_input_dim=3, hidden_dim=128, n_j=N_J, n_m=N_M, max_op=MAX_OP, metadata=metadata).to(DEVICE)
    try:
        model_path = os.path.join("models_save", f"{N_J}_{N_M}_best_model_{296}.pth")
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("成功加载已训练的 FJSP 模型！")
    except Exception as e:
        print(f"未找到模型 ({e})，将使用【未训练】模型进行随机测试！")
    model.eval()

    # 3. 数据预处理 (构建 PyG Batch)
    pyg_data_list = []
    print("Preparing PyG data...")
    for i in range(batch_size):
        pt = all_proc_times[i]
        mask = all_compat_masks[i]
        jl = all_job_lengths[i]
        # 调用新的数据处理函数
        adj, fea, mach_fea = get_initial_input(n_j=N_J, n_m=N_M, max_n_op=n_op, proc_times=pt, compat_mask=mask,
                                               job_length=jl)
        data = convert_to_pyg_data(adj, fea, mach_fea, pt, mask, N_M, jl)
        pyg_data_list.append(data)

    test_loader = DataLoader(pyg_data_list, batch_size=batch_size, shuffle=False)

    # 4. 运行 RL
    print(f">>> Running RL on {batch_size} instances...")
    rl_start = time.time()
    rl_makespans = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            bsz = batch.num_graphs
            n_node = N_J * n_op

            # 还原 3D 张量
            op_proc_time = batch.proc_times.view(bsz, n_node, N_M).float()
            mask_machine_compat = batch.compat_mask.view(bsz, n_node, N_M).bool()
            job_length_t = batch.job_length.view(bsz, N_J)
            job_seq_g, mach_assign_g, *_ = model(
                batch, mask_machine_compat, op_proc_time, job_length_t, rollout=True
            )
            _, best_costs = fjsp_sched_bch(job_seq_g, mach_assign_g, op_proc_time,
                                           n_j=N_J, n_m=N_M, n_op=MAX_OP, job_length=job_length_t)

            for s in range(NUM_SAMPLES):
                job_seq_s, mach_assign_s, *_ = model(
                    batch,
                    mask_machine_compat,
                    op_proc_time,
                    job_length_t,
                    rollout=True,
                    temperature=1.3
                )
                # 计算 Makespan
                _, sample_costs = fjsp_sched_bch(job_seq_s, mach_assign_s, op_proc_time,
                                                 n_j=N_J, n_m=N_M, n_op=MAX_OP, job_length=job_length_t)
                best_costs = torch.minimum(best_costs, sample_costs)

            rl_makespans.extend(best_costs.cpu().tolist())

    rl_end = time.time()
    rl_total_time = rl_end - rl_start
    print(f"RL Finished. Total Time: {rl_total_time:.4f}s")

    # ---------------------------------------------------------
    # 5. 运行 OR-Tools
    # ---------------------------------------------------------
    print(f">>> Running OR-Tools (Time Limit={OR_TOOLS_LIMIT}s)...")
    ortools_makespans = []
    ortools_times = []

    for i in range(batch_size):
        pt = all_proc_times[i]
        mask = all_compat_masks[i]
        jl = all_job_lengths[i]

        pt_or = pt.copy()
        mask_or = mask.copy()
        for j in range(N_J):
            actual_len = jl[j]
            if actual_len < n_op:
                # 强制让 dummy 工序兼容 0 号机器，且耗时为 0.0
                mask_or[j, actual_len:, 0] = True
                pt_or[j, actual_len:, 0] = 0.0
        t_start = time.time()
        val, status = fjsp_solver(pt, mask, job_length=jl, time_limit_seconds=OR_TOOLS_LIMIT)
        t_end = time.time()

        ortools_makespans.append(val)
        ortools_times.append(t_end - t_start)
        val_str = f"{val:.4f}" if val is not None else "N/A"
        print(f"  Instance {i + 1}/{batch_size}: OR-Tools={val_str} (Status: {status})")

    # 6. 生成报告
    results = []
    for i in range(batch_size):
        rl_val = rl_makespans[i]
        or_val = ortools_makespans[i]

        gap = ((rl_val - or_val) / or_val * 100) if or_val > 0 else 0.0

        results.append({
            "Instance": i,
            "RL_Makespan": rl_val,
            "OR_Makespan": or_val,
            "Gap(%)": gap,
            "OR_Time": ortools_times[i]
        })

    df = pd.DataFrame(results)
    valid_ortools = [v for v in ortools_makespans if v is not None]
    valid_gaps = df['Gap(%)'].dropna()

    print("\n" + "=" * 50)
    print("             COMPARISON REPORT             ")
    print("=" * 50)
    print(f"Avg RL Makespan:       {np.mean(rl_makespans):.4f}")
    if ortools_makespans[0] > 0:
        print(f"Avg OR-Tools Makespan: {np.mean(valid_ortools):.4f}")
        print(f"Average Gap:           {valid_gaps.mean():.2f}%")
    else:
        print("OR-Tools not implemented for FJSP yet. Gap skipped.")
    print("-" * 50)
    print(f"RL Avg Time/Instance:  {rl_total_time / batch_size:.6f} s")
    print(f"OR Avg Time/Instance:  {np.mean(ortools_times):.6f} s")
    print("=" * 50)

    df.to_csv("comparison_results.csv", index=False)


if __name__ == "__main__":
    run_benchmark()