import torch
import numpy as np
import pandas as pd
import time
import os
from torch_geometric.loader import DataLoader

from model import FJSPActor
from utils import fjsp_sched_bch
from data_utils import convert_to_pyg_data, get_initial_input
from ortools_solver import fjsp_solver

# --- 测试超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

HIDDEN_DIM = 768
N_LAYERS = 12
N_HEADS = 12

NUM_SAMPLES = 128
# OR_TOOLS_LIMIT = 1800.0


# ====== 1. 统一的公共 FJSP 数据集解析器 (动态规模自适应) ======
def parse_fjsp_file(filepath):
    """
    读取标准 FJSP (Brandimarte/Hurink) 文本格式。
    自动推断当前实例的真实规模 (actual_j, actual_m, actual_max_op)，
    并返回贴合该规模的紧凑张量，不再进行全局 Padding！
    """
    with open(filepath, 'r') as f:
        tokens = []
        for line in f.readlines():
            tokens.extend(line.strip().split())

    if not tokens: return None

    actual_j = int(tokens[0])
    actual_m = int(tokens[1])

    ptr = 3 if len(tokens) > 3 and float(tokens[2]) < actual_m else 2

    # 第一遍扫描：确定最大工序数 max_op
    job_lengths = []
    temp_ptr = ptr
    for i in range(actual_j):
        n_ops = int(tokens[temp_ptr])
        job_lengths.append(n_ops)
        temp_ptr += 1
        for j in range(n_ops):
            num_alt = int(tokens[temp_ptr])
            temp_ptr += 1 + 2 * num_alt

    actual_max_op = max(job_lengths)

    # 按照真实的紧凑规模初始化张量
    proc_times = np.zeros((actual_j, actual_max_op, actual_m), dtype=np.float32)
    compat_mask = np.zeros((actual_j, actual_max_op, actual_m), dtype=bool)
    job_length = np.array(job_lengths, dtype=np.int32)

    # 第二遍扫描：填充数据
    for i in range(actual_j):
        n_ops = int(tokens[ptr])
        ptr += 1
        for j in range(n_ops):
            num_alt = int(tokens[ptr])
            ptr += 1
            for _ in range(num_alt):
                mach = int(tokens[ptr]) - 1
                ptime = float(tokens[ptr + 1])
                ptr += 2

                proc_times[i, j, mach] = ptime
                compat_mask[i, j, mach] = True

    return proc_times, compat_mask, job_length, actual_j, actual_m, actual_max_op


# ====== 2. 主测试流程 ======
def run_public_benchmark():
    print("=" * 60)
    print("  FJSP-RL 公共数据集对比测试 ")
    print("=" * 60)



    # 1. 初始化并加载 RL 模型
    model_path = os.path.join("models_save", f"15_10_best_model_64.pth")
    if not os.path.exists(model_path):
        print(f"找不到模型权重文件: {model_path}")
        return

    # 构建极小假数据仅用于获取网络图拓扑结构的 metadata
    import data_utils
    pt, mask, jl = data_utils.uni_instance_gen(1, 1, 1, 1)
    adj, fea, mach_fea = get_initial_input(1, 1, 1, pt, mask, jl)
    dummy_data = convert_to_pyg_data(adj, fea, mach_fea, pt, mask, 1, jl)

    model = FJSPActor(op_input_dim=6, mach_input_dim=3, hidden_dim=HIDDEN_DIM,
                      metadata=dummy_data.metadata(), n_layers=N_LAYERS, n_heads=N_HEADS).to(DEVICE)

    # 剥离 DDP 外壳加载权重
    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        print("成功加载 RL 模型！\n")
    except Exception as e:
        print(f"模型加载失败，请检查参数维度是否对齐: {e}")
        return

    # 2. 遍历数据集文件夹
    datasets = [
        "BenchData/Brandimarte",
        "BenchData/Hurink_edata",
        "BenchData/Hurink_rdata",
        "BenchData/Hurink_vdata"
    ]
    results = []

    for ds_folder in datasets:
        if not os.path.exists(ds_folder):
            print(f"未找到文件夹 {ds_folder}，跳过...")
            continue

        print(f"\n--- 正在处理数据集: {ds_folder} ---")

        for filename in sorted(os.listdir(ds_folder)):
            filepath = os.path.join(ds_folder, filename)
            if not os.path.isfile(filepath): continue

            # ================= 解析动态规模数据 =================
            parsed = parse_fjsp_file(filepath)
            if parsed is None:
                continue

            pt, mask, jl, actual_j, actual_m, actual_max_op = parsed
            print(f">>> 测试实例: {filename} (规模: {actual_j}工件 x {actual_m}机器, 最高 {actual_max_op}道工序)")

            adj, fea, mach_fea = get_initial_input(n_j=actual_j, n_m=actual_m, max_n_op=actual_max_op,
                                                   proc_times=pt, compat_mask=mask, job_length=jl)
            data = convert_to_pyg_data(adj, fea, mach_fea, pt, mask, actual_m, jl)
            batch = next(iter(DataLoader([data], batch_size=1))).to(DEVICE)

            # 动态展开张量
            bsz = 1
            n_node = actual_j * actual_max_op
            op_proc_time = batch.proc_times.view(bsz, n_node, actual_m).float()
            mask_machine_compat = batch.compat_mask.view(bsz, n_node, actual_m).bool()
            job_length_t = batch.job_length.view(bsz, actual_j)

            with torch.no_grad():
                # ================= a. 贪婪搜索 (Greedy) =================
                t_start_g = time.time()
                job_seq_g, mach_assign_g, *_ = model(
                    batch, mask_machine_compat, op_proc_time, job_length_t, rollout=True
                )
                _, greedy_costs = fjsp_sched_bch(job_seq_g, mach_assign_g, op_proc_time,
                                                 n_j=actual_j, n_m=actual_m, n_op=actual_max_op,
                                                 job_length=job_length_t)
                greedy_mksp = greedy_costs.item()
                greedy_time = time.time() - t_start_g

                # ================= b. 采样搜索 (Sampling) =================
                t_start_s = time.time()
                best_sample_cost = float('inf')
                for _ in range(NUM_SAMPLES):
                    job_seq_s, mach_assign_s, *_ = model(
                        batch, mask_machine_compat, op_proc_time, job_length_t,
                        rollout=False, temperature=1.2
                    )
                    _, sample_costs = fjsp_sched_bch(job_seq_s, mach_assign_s, op_proc_time,
                                                     n_j=actual_j, n_m=actual_m, n_op=actual_max_op,
                                                     job_length=job_length_t)
                    if sample_costs.item() < best_sample_cost:
                        best_sample_cost = sample_costs.item()

                sampling_mksp = best_sample_cost
                sampling_time = time.time() - t_start_s

            print(f"    Greedy Mksp:   {greedy_mksp:.1f}  (耗时: {greedy_time:.4f}s)")
            print(f"    Sampling Mksp: {sampling_mksp:.1f}  (耗时: {sampling_time:.4f}s, {NUM_SAMPLES} 次采样)")
            print("-" * 50)

            results.append({
                "Dataset": ds_folder.split('/')[-1],
                "Instance": filename,
                "Greedy_Makespan": greedy_mksp,
                "Greedy_Time(s)": greedy_time,
                "Sampling_Makespan": sampling_mksp,
                "Sampling_Time(s)": sampling_time,
            })

            # 3. 输出汇总统计
        if not results:
            print("没有成功测试任何实例！请检查数据路径是否正确。")
            return

        df = pd.DataFrame(results)
        print("\n" + "=" * 50)
        print("             FINAL SUMMARY REPORT             ")
        print("=" * 50)
        print(f"总计测试实例数: {len(df)}")

        print(f"平均 Greedy Makespan:   {df['Greedy_Makespan'].mean():.2f}")
        print(f"平均 Sampling Makespan: {df['Sampling_Makespan'].mean():.2f}")
        print("")
        print(f"平均 Greedy 推理耗时:   {df['Greedy_Time(s)'].mean():.4f} s")
        print(f"平均 Sampling 推理耗时: {df['Sampling_Time(s)'].mean():.4f} s")
        print("=" * 50)

        # 按照不同数据集分组汇报
        print("\n--- 按数据集分类汇总 ---")
        summary_df = df.groupby('Dataset').agg(
            Count=('Instance', 'count'),
            Avg_Greedy_Mksp=('Greedy_Makespan', 'mean'),
            Avg_Sampling_Mksp=('Sampling_Makespan', 'mean'),
            Avg_Sampling_Time=('Sampling_Time(s)', 'mean')
        ).round(2)
        print(summary_df)

    df.to_csv("public_dataset_comparison.csv", index=False)
    print("\n详细结果已保存至 public_dataset_comparison.csv")


if __name__ == "__main__":
    run_public_benchmark()