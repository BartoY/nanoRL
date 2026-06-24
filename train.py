import os
import math  # 新增
import numpy as np
import random
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from copy import deepcopy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from model import FJSPActor
from utils import fjsp_sched_bch, chk_upd_bl
from plot import plot_learning_curves
from data_utils import epoch_dataset_gen, _generate_single_instance
from validate import validate_model
import wandb

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# --- 超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
MAX_LR = 3e-4  # [修改] 配合 OneCycleLR 的最大学习率，替代固定的小 LR
BATCH_SIZE = 256
EPOCHS = 10  # [建议] 数据量如此之大，实际上不需要30个Epoch，这里建议改小，10已经非常充分
N_J = 10
N_M = 5
MIN_OP = int(N_M * 0.8)
MAX_OP = int(N_M * 1.2)
n_simple = 1024000
ENTROPY_COEF = 0.01  # 熵正则化系数
TEMP_START = 1.2  # 初始温度
TEMP_END = 0.8

VAL_FREQ = 500  # [新增] 每 500 个 batch (Step) 进行一次验证和 Baseline 更新检查


def main():
    # 初始化 DDP 进程组
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f"cuda:{local_rank}")

    dummy_data = _generate_single_instance(0, N_J, N_M, MIN_OP, MAX_OP, return_pyg=True)
    metadata = dummy_data.metadata()
    # 初始化模型
    policy_model = FJSPActor(op_input_dim=6, mach_input_dim=3, hidden_dim=768, metadata=metadata, n_layers=12,
                             n_heads=12).to(DEVICE)
    baseline_model = deepcopy(policy_model)
    baseline_model.eval()

    policy_model = DDP(policy_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # [修改] 基础优化器设置（学习率由 Scheduler 完全接管）
    optimizer = optim.AdamW(policy_model.parameters(), lr=MAX_LR, weight_decay=1e-4)
    scaler = GradScaler('cuda')

    world_size = dist.get_world_size()
    local_n_simple = n_simple // world_size

    # [新增] 提前计算总 Step 数，用于 OneCycleLR 初始化
    steps_per_epoch = math.ceil(local_n_simple / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS

    # [修改] 使用带有 Warmup 的 OneCycleLR 按 Step 衰减
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        total_steps=total_steps,
        pct_start=0.05,  # 前 5% 的 Step 用于预热(Warmup)
        anneal_strategy='cos',  # 余弦平滑衰减
        div_factor=10.0,  # 初始LR = MAX_LR / 10
        final_div_factor=100.0  # 最终LR = 初始LR / 100
    )

    if local_rank == 0:
        # 1. 初始化 wandb，记录超参数
        wandb.init(
            project="nanoRL-FJSP",
            name=f"Run_NJ{N_J}_NM{N_M}_BSZ{BATCH_SIZE}",
            config={
                "batch_size": BATCH_SIZE,
                "max_lr": MAX_LR,
                "epochs": EPOCHS,
                "entropy_coef": ENTROPY_COEF,
                "n_j": N_J,
                "n_m": N_M,
                "val_freq": VAL_FREQ
            }
        )

    # 验证集
    if local_rank == 0:
        val_data_list = epoch_dataset_gen(n_samples=1024, n_j=N_J, n_m=N_M, min_op=MIN_OP, max_op=MAX_OP)
        val_loader = DataLoader(val_data_list, batch_size=BATCH_SIZE, shuffle=False)
        print("Evaluating initial baseline...")
        baseline_val_costs = validate_model(baseline_model, val_loader, DEVICE, N_J, N_M, MAX_OP)
        print(f"Initial Baseline Avg Mksp: {baseline_val_costs.mean():.2f}")

        history_loss = []
        history_train_mksp = []
        history_val_mksp = []
        best_val_makespan = float('inf')

    dist.barrier()

    global_step = 0  # [新增] 全局 Step 计数器

    # 用于收集在 VAL_FREQ 期间的平均训练指标
    running_loss = 0.0
    running_train_mksp = 0.0

    # 训练循环
    for epoch in range(EPOCHS):
        seed = 42 + epoch * world_size + local_rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        train_data_list = epoch_dataset_gen(n_samples=local_n_simple, n_j=N_J, n_m=N_M, min_op=MIN_OP, max_op=MAX_OP)
        train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)

        policy_model.train()

        for step_idx, batch in enumerate(train_loader):
            global_step += 1

            # [修改] 每一步平滑衰减温度，而不是每个 Epoch 才变
            current_temp = TEMP_END + (TEMP_START - TEMP_END) * (1.0 - global_step / total_steps)

            batch = batch.to(DEVICE)
            bsz = batch.num_graphs
            n_node = N_J * MAX_OP

            op_proc_time = batch.proc_times.view(bsz, n_node, N_M).float()
            mask_machine_compat = batch.compat_mask.view(bsz, n_node, N_M).bool()
            job_length = batch.job_length.view(bsz, N_J)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                job_seq, mach_assign, log_probs, entropy = policy_model(
                    batch, mask_machine_compat, op_proc_time, job_length,
                    rollout=False, temperature=current_temp
                )

                with torch.no_grad():
                    base_job_seq, base_mach_assign, *_ = baseline_model(
                        batch, mask_machine_compat, op_proc_time, job_length, rollout=True
                    )
                    _, costs = fjsp_sched_bch(job_sequence=job_seq, mach_sequence=mach_assign,
                                              proc_times=op_proc_time, n_j=N_J, n_m=N_M, n_op=MAX_OP,
                                              job_length=job_length)
                    _, base_costs = fjsp_sched_bch(job_sequence=base_job_seq, mach_sequence=base_mach_assign,
                                                   proc_times=op_proc_time, n_j=N_J, n_m=N_M, n_op=MAX_OP,
                                                   job_length=job_length)
                    advantage = (costs - base_costs).detach()

                # Advantage归一化
                if advantage.numel() > 1:
                    advantage = advantage / (advantage.std(unbiased=False) + 1e-8)
                rl_loss = (advantage * log_probs).mean()
                entropy_bonus = entropy.mean() * ENTROPY_COEF
                loss = rl_loss - entropy_bonus

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # [极其重要] Scheduler 挪到了每个 Step 之后
            scheduler.step()

            # 累计当前局部 Step 的数据
            running_loss += loss.item()
            running_train_mksp += costs.mean().item()

            # ==============================================================
            # [核心修改] 每 VAL_FREQ 步，进行验证和 Baseline 切换评估
            # ==============================================================
            if global_step % VAL_FREQ == 0 or global_step == total_steps:
                # 跨进程汇总过去 VAL_FREQ 步的累积数据
                metrics = torch.tensor([running_loss, running_train_mksp], device=DEVICE)
                dist.reduce(metrics, dst=0, op=dist.ReduceOp.SUM)

                update_flag = torch.tensor([0], device=DEVICE)

                if local_rank == 0:
                    # 计算这段时间内的真实平均值
                    avg_loss = metrics[0].item() / (VAL_FREQ * world_size)
                    avg_train_mksp = metrics[1].item() / (VAL_FREQ * world_size)
                    current_lr = scheduler.get_last_lr()[0]

                    # 模型验证
                    policy_model.eval()
                    policy_val_costs = validate_model(policy_model, val_loader, DEVICE, N_J, N_M, MAX_OP)
                    policy_model.train()  # 验证完切回 train

                    avg_val_mksp = policy_val_costs.mean()

                    wandb.log({
                        "Step": global_step,
                        "Epoch": epoch + 1,
                        "Loss/Train": avg_loss,
                        "Makespan/Train": avg_train_mksp,
                        "Makespan/Validation": avg_val_mksp,
                        "Temperature": current_temp,
                        "Learning_Rate": current_lr
                    }, step=global_step)

                    print(
                        f"Epoch {epoch + 1} | Step {global_step}/{total_steps} | LR {current_lr:.6f} | Loss {avg_loss:.4f} | Train Mksp {avg_train_mksp:.2f} | Val Mksp {avg_val_mksp:.2f}")

                    history_loss.append(avg_loss)
                    history_train_mksp.append(avg_train_mksp)
                    history_val_mksp.append(avg_val_mksp)

                    # 保存 Best Model
                    if avg_val_mksp < best_val_makespan:
                        best_val_makespan = avg_val_mksp
                        # torch.save(policy_model.module.state_dict(), f"/home/yifan/hang/nanoRL/models_save/{N_J}_{N_M}_best_model_{BATCH_SIZE}.pth")
                        torch.save(policy_model.module.state_dict(),
                                   f"/raid/hangy/nanoRL/models_save/{N_J}_{N_M}_best_model_{BATCH_SIZE}.pth")
                        print(f"  >>> New Best Model Saved! (Val Mksp: {best_val_makespan:.2f})")

                    # 判定是否升级 Baseline
                    should_update = chk_upd_bl(policy_val_costs, baseline_val_costs)
                    if should_update:
                        update_flag[0] = 1
                        baseline_val_costs = policy_val_costs

                # 广播判定结果：只有 rank 0 算了是不是要更新，告诉所有人一起更新
                dist.broadcast(update_flag, src=0)

                if update_flag.item() == 1:
                    if local_rank == 0:
                        print("  >>> Updating Baseline...")
                    # 所有人同步 Baseline 权重
                    baseline_model.load_state_dict(policy_model.module.state_dict())

                # 重置累积数据供下一个 VAL_FREQ 使用
                running_loss = 0.0
                running_train_mksp = 0.0

    if local_rank == 0:
        print("Training finished. Plotting curves...")
        plot_learning_curves(history_loss, history_train_mksp, history_val_mksp)
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()