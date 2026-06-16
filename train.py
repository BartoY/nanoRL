import os
import numpy as np
import random
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from copy import deepcopy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import FJSPActor
from utils import fjsp_sched_bch, chk_upd_bl
from plot import plot_learning_curves
from data_utils import epoch_dataset_gen, _generate_single_instance
from validate import validate_model
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# --- 超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
LR = 1e-5
BATCH_SIZE = 12
EPOCHS = 30
N_J = 15
N_M = 10
MIN_OP = 5
MAX_OP = 15
n_simple = 1024000
ENTROPY_COEF = 0.01   # 熵正则化系数
TEMP_START = 1.5     # 初始温度
TEMP_END = 1.0


def main():
    # 初始化 DDP 进程组
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f"cuda:{local_rank}")

    dummy_data = _generate_single_instance(0, N_J, N_M, MIN_OP, MAX_OP, return_pyg=True)
    metadata = dummy_data.metadata()
    # 初始化模型
    policy_model = FJSPActor(op_input_dim=6, mach_input_dim=3, hidden_dim=768, metadata=metadata, n_layers=12, n_heads=12).to(DEVICE)
    baseline_model = deepcopy(policy_model)
    baseline_model.eval()

    policy_model = DDP(policy_model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(policy_model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

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

    world_size = dist.get_world_size()
    local_n_simple = n_simple // world_size
    dist.barrier()

    # 训练循环
    for epoch in range(EPOCHS):
        # torch.cuda.empty_cache()
        seed = 42 + epoch * world_size + local_rank
        torch.manual_seed(seed)

        np.random.seed(seed)
        random.seed(seed)

        current_temp = TEMP_END + (TEMP_START - TEMP_END) * (1.0 - epoch / EPOCHS)

        train_data_list = epoch_dataset_gen(n_samples=local_n_simple, n_j=N_J, n_m=N_M, min_op=MIN_OP, max_op=MAX_OP)
        train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)

        policy_model.train()
        total_loss = 0
        total_train_mksp = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            bsz = batch.num_graphs
            n_node = N_J * MAX_OP

            op_proc_time = batch.proc_times.view(bsz, n_node, N_M).float()
            mask_machine_compat = batch.compat_mask.view(bsz, n_node, N_M).bool()

            job_length = batch.job_length.view(bsz, N_J)

            job_seq, mach_assign, log_probs, entropy = policy_model(
                batch,
                mask_machine_compat,
                op_proc_time,
                job_length,
                rollout=False,
                temperature=current_temp
            )

            with torch.no_grad():
                base_job_seq, base_mach_assign, *_ = baseline_model(
                    batch,
                    mask_machine_compat,
                    op_proc_time,
                    job_length,
                    rollout=True
                )
                _, costs = fjsp_sched_bch(job_sequence=job_seq,
                                      mach_sequence=mach_assign,
                                      proc_times=op_proc_time, n_j=N_J, n_m=N_M, n_op=MAX_OP, job_length=job_length)
                _, base_costs = fjsp_sched_bch(job_sequence=base_job_seq,
                                           mach_sequence=base_mach_assign,
                                           proc_times=op_proc_time, n_j=N_J, n_m=N_M, n_op=MAX_OP, job_length=job_length)

                # --- Loss ---
                advantage = (costs - base_costs).detach()

                # Advantage归一化
                # if advantage.numel() > 1:
                #     advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

            rl_loss = (advantage * log_probs).mean()

            entropy_bonus = entropy.mean() * ENTROPY_COEF
            loss = rl_loss - entropy_bonus
            # loss = rl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_train_mksp += costs.mean().item()

        metrics = torch.tensor([total_loss, total_train_mksp], device=DEVICE)
        dist.reduce(metrics, dst=0, op=dist.ReduceOp.SUM)

        scheduler.step()
        update_flag = torch.tensor([0], device=DEVICE)
        if local_rank == 0:
            global_batches = len(train_loader) * world_size
            avg_loss = metrics[0].item() / global_batches
            avg_train_mksp = metrics[1].item() / global_batches

            policy_val_costs = validate_model(policy_model, val_loader, DEVICE, N_J, N_M, MAX_OP)
            avg_val_mksp = policy_val_costs.mean()

            print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f} | Train Mksp {avg_train_mksp:.2f} | Val Mksp {avg_val_mksp:.2f}")

            # 记录当前 Epoch 的平均数据
            history_loss.append(avg_loss)
            history_train_mksp.append(avg_train_mksp)
            history_val_mksp.append(avg_val_mksp)

            # --- 保存最佳模型 ---
            if avg_val_mksp < best_val_makespan:
                best_val_makespan = avg_val_mksp
                # torch.save(policy_model.module.state_dict(), f"/home/yifan/hang/nanoRL/models_save/{N_J}_{N_M}_best_model_{BATCH_SIZE}.pth")
                torch.save(policy_model.module.state_dict(), f"/raid/hangy/nanoRL/models_save/{N_J}_{N_M}_best_model_{BATCH_SIZE}.pth")
                print(f"  >>> New Best Model Saved! (Val Mksp: {best_val_makespan:.2f})")

            # --- Update Baseline ---
            should_update = chk_upd_bl(policy_val_costs, baseline_val_costs)

            if should_update:
                update_flag[0] = 1
                baseline_val_costs = policy_val_costs
        dist.broadcast(update_flag, src=0)
        if update_flag.item() == 1:
            if local_rank == 0:
                print("Updating Baseline...")
            baseline_model.load_state_dict(policy_model.module.state_dict())
    if local_rank == 0:
        print("Training finished. Plotting curves...")
        plot_learning_curves(history_loss, history_train_mksp, history_val_mksp)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()