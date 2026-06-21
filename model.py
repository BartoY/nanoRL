import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import LayerNorm
from torch_geometric.nn import GATv2Conv, to_hetero
# from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_dense_batch
import math
import contextlib


class Attention(nn.Module):
    def __init__(self,
                 q_hidden_dim,
                 k_dim,
                 v_dim,
                 n_head,
                 k_hidden_dim=None,
                 v_hidden_dim=None):
        super().__init__()
        self.q_hidden_dim = q_hidden_dim
        self.k_hidden_dim = k_hidden_dim if k_hidden_dim else q_hidden_dim
        self.v_hidden_dim = v_hidden_dim if v_hidden_dim else q_hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.proj_q = nn.Linear(q_hidden_dim, k_dim * n_head, bias=False)
        self.proj_k = nn.Linear(self.k_hidden_dim, k_dim * n_head, bias=False)
        self.proj_v = nn.Linear(self.v_hidden_dim, v_dim * n_head, bias=False)
        self.proj_output = nn.Linear(v_dim * n_head,
                                     self.v_hidden_dim,
                                     bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None: k = q
        if v is None: v = k

        bsz, n_node, _ = k.size()

        # 计算 Q, K, V
        qs = torch.stack(torch.chunk(self.proj_q(q), self.n_head, dim=-1), dim=1)
        ks = torch.stack(torch.chunk(self.proj_k(k), self.n_head, dim=-1), dim=1)
        vs = torch.stack(torch.chunk(self.proj_v(v), self.n_head, dim=-1), dim=1)

        normalizer = self.k_dim ** 0.5
        u = torch.matmul(qs, ks.transpose(2, 3)) / normalizer

        if mask is not None:
            # mask shape 转换: [bsz, n_node] -> [bsz, 1, 1, n_node]
            mask = mask.unsqueeze(1).unsqueeze(1)
            u = u.masked_fill(mask, float('-inf'))

        att = torch.matmul(torch.softmax(u, dim=-1), vs)
        att = att.transpose(1, 2).reshape(bsz, -1, self.v_dim * self.n_head)
        att = self.proj_output(att)
        return att


class GPTLayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = GPTLayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = GPTLayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig:
    def __init__(self, n_embd, n_head, dropout, block_size=10000):
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.block_size = block_size # 最大序列长度 n_j * max_op
        self.bias = False


class BaseGNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False, edge_dim=3)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False, edge_dim=3)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False, edge_dim=3)

        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.norm1(F.relu(h))

        h_in = h
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(F.relu(h) + h_in)  # 残差连接

        h_in2 = h
        h = self.conv3(h, edge_index, edge_attr=edge_attr)
        h = self.norm3(F.relu(h) + h_in2)  # 残差连接
        return h


class GPTDecoder(nn.Module):
    def __init__(self, hidden_dim, n_layers=12, n_heads=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        config = GPTConfig(n_embd=hidden_dim, n_head=n_heads, dropout=0.0, block_size=2000)

        # 位置编码
        self.wpe = nn.Embedding(config.block_size, hidden_dim)
        # 丢弃层
        self.drop = nn.Dropout(config.dropout)
        # nanoGPT 的多层 Transformer Block
        self.h = nn.ModuleList([Block(config) for _ in range(n_layers)])
        self.ln_f = GPTLayerNorm(hidden_dim, bias=config.bias)

        # 工序选择
        self.op_pointer_att = Attention(q_hidden_dim=hidden_dim,
                                        k_dim=hidden_dim // n_heads,
                                        v_dim=hidden_dim // n_heads,
                                        n_head=n_heads)
        self.q_mach_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        # 机器选择
        self.mach_pointer_att = Attention(q_hidden_dim=hidden_dim,
                                          k_dim=hidden_dim // n_heads,
                                          v_dim=hidden_dim // n_heads,
                                          n_head=n_heads,
                                          k_hidden_dim=hidden_dim,  # 显式声明 k 的维度
                                          v_hidden_dim=hidden_dim)
        self.start_token = nn.Parameter(torch.randn(1, hidden_dim))
        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dyn_dim = 7
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim + self.dyn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dyn_mach_proj = nn.Linear(hidden_dim + 4, hidden_dim)
        self.tanh_clipping = 10.0
        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, encoder_out, mach_dense, mask_padding, mask_machine_compat, op_proc_time, job_length, rollout=False, temperature=1.0):
        """
            encoder_out:[B, n_tasks, H] (工序的异构图特征)
            mach_dense:[B, n_m, H] (机器的异构图特征)
        """
        bsz, n_node, _ = encoder_out.size()
        device = encoder_out.device

        n_j = job_length.size(1)
        n_m = mach_dense.size(1)
        max_op = n_node // n_j

        k_logits_all = self.proj_k(encoder_out)
        valid_mask = (~mask_padding).unsqueeze(-1).float()
        graph_context = (encoder_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)

        curr_input = graph_context + self.start_token.expand(bsz, -1)  # [B, H]
        seq_inputs = []

        job_next_op_local_idx = torch.zeros(bsz, n_j, dtype=torch.long, device=device)
        mask_job_finished = (job_length == 0)

        job_ready_time = torch.zeros(bsz, n_j, device=device)
        machine_avail_time = torch.zeros(bsz, n_m, device=device)

        # 每个Job当前待选工序的全局Node Index, base_indices: [0, max_op, 2*max_op, ...]
        base_indices = torch.arange(0, n_j * max_op, max_op, device=device).unsqueeze(0).expand(bsz, -1)
        avg_op_pt = (op_proc_time * mask_machine_compat.float()
                     ).sum(-1) / mask_machine_compat.float().sum(-1).clamp(min=1.0)
        max_steps = int(job_length.sum(dim=1).max().item())

        actions_job_list = []
        actions_mach_list = []

        sel_node_idx_list = []
        safe_indices_list = []
        dyn_feats_list = []
        mask_job_finished_list = []
        batch_is_done_list = []
        curr_op_mach_compat_list = []
        k_mach_dyn_list = []

        seq_inputs = []

        with torch.no_grad() if not rollout else contextlib.nullcontext():
            for i in range(max_steps):
                # 将当前的 input 存入序列
                seq_inputs.append(curr_input)   # list of [B, H]

                # 将历史记录拼接成[B, T, H]
                x_seq = torch.stack(seq_inputs, dim=1)
                T_curr = x_seq.size(1)

                # 生成位置编码
                pos = torch.arange(0, T_curr, dtype=torch.long, device=device)  # [T]
                pos_emb = self.wpe(pos)  # [T, H]

                # 输入 GPT
                x = self.drop(x_seq + pos_emb)
                for block in self.h:
                    x = block(x)
                x = self.ln_f(x)  # [B, T, H]
                hx = x[:, -1, :]  # [B, H]
                q_op = hx.unsqueeze(1)
                glimpse_op = self.op_pointer_att(q=q_op, k=encoder_out, v=encoder_out, mask=mask_padding)

                current_global_indices = base_indices + job_next_op_local_idx
                safe_indices = current_global_indices.clamp(max=n_node - 1)
                safe_indices_list.append(safe_indices)  # [Cache]

                # k_logits_all: [B, N_tasks, H] -> [B, n_j, H]
                safe_indices_expanded = safe_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
                k_candidates = torch.gather(k_logits_all, 1, safe_indices_expanded)

                # 当前候选工序的机器兼容矩阵
                cand_pt_est = torch.gather(avg_op_pt, 1, safe_indices)
                cand_mach_mask = torch.gather(mask_machine_compat, 1, safe_indices.unsqueeze(-1).expand(-1, -1, n_m))
                avail_time_expanded = machine_avail_time.unsqueeze(1).expand(-1, n_j, -1)
                valid_avail_time = avail_time_expanded.masked_fill(~cand_mach_mask, float('inf'))
                min_comp_mach_avail = valid_avail_time.min(dim=-1)[0]  # [B, n_j]
                min_comp_mach_avail = min_comp_mach_avail.masked_fill(min_comp_mach_avail == float('inf'), 0.0)

                e_start = torch.max(job_ready_time, min_comp_mach_avail)  # 最早开始时间
                e_comp = e_start + cand_pt_est  # 预计完成时间
                wait_time = e_start - job_ready_time  # 工件等待时间
                idle_time = e_start - min_comp_mach_avail  # 机器空闲时间
                ops_left_ratio = (job_length - job_next_op_local_idx) / job_length.clamp(min=1)  # 剩余工序比例
                pt_ratio = cand_pt_est / (cand_pt_est.mean(dim=-1, keepdim=True) + 1e-5)  # 剩余加工时间

                job_progress = 1.0 - ops_left_ratio.float()
                valid_job_mask = (job_length > 0).float()
                mean_progress = (job_progress * valid_job_mask).sum(dim=1, keepdim=True) / \
                                valid_job_mask.sum(dim=1,keepdim=True).clamp(min=1.0)
                progress_diff = job_progress - mean_progress  # [B, n_j]

                norm_factor = torch.max(machine_avail_time, dim=1, keepdim=True)[0].clamp(min=1.0)
                dyn_feats = torch.stack([
                    e_start / norm_factor,
                    e_comp / norm_factor,
                    wait_time / norm_factor,
                    idle_time / norm_factor,
                    ops_left_ratio.float(),
                    pt_ratio,
                    progress_diff
                ], dim=-1)  # [B, n_j, 7]
                dyn_feats_list.append(dyn_feats)  # [Cache]

                cat_feats = torch.cat([k_candidates, dyn_feats], dim=-1)
                k_candidates_fused = self.fusion_net(cat_feats)  # [B, n_j, H]

                # ---计算 Logits---
                u_op = torch.matmul(glimpse_op, k_candidates_fused.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
                u_op = torch.tanh(u_op) * self.tanh_clipping / temperature
                u_op = u_op.squeeze(1)

                batch_is_done = mask_job_finished.all(dim=-1)  # [B]
                batch_is_done_list.append(batch_is_done)  # [Cache]

                is_all_masked = mask_job_finished.all(dim=-1, keepdim=True)
                mask_job_finished = mask_job_finished.masked_fill(is_all_masked, False)
                mask_job_finished_list.append(mask_job_finished)  # [Cache]

                # ---Mask掉已完成的Job---
                u_op = u_op.masked_fill(mask_job_finished, float('-inf'))
                u_op_safe = u_op.masked_fill(batch_is_done.unsqueeze(-1), 0.0)
                u_op_safe = u_op_safe.masked_fill(mask_job_finished & ~batch_is_done.unsqueeze(-1), float('-inf'))

                # 采样/贪婪选择
                if rollout:
                    selected_job = u_op_safe.max(-1)[1]
                else:
                    probs = F.softmax(u_op_safe, dim=-1)
                    selected_job = Categorical(probs).sample()
                    # 只记录有效 batch 的对数概率和熵

                selected_job = selected_job.masked_fill(batch_is_done, 0)
                actions_job_list.append(selected_job)

                # ---------------------------------------------------------
                # Low-Level: 为选出的工序分配机器
                # ---------------------------------------------------------
                sel_job_unsq = selected_job.unsqueeze(1)
                sel_node_idx = torch.gather(current_global_indices, 1, sel_job_unsq)
                sel_node_idx_list.append(sel_node_idx)  # [Cache]

                idx_exp = sel_node_idx.unsqueeze(-1)
                selected_op_emb = torch.gather(encoder_out, 1,
                                               idx_exp.expand(-1, -1, self.hidden_dim)
                                               ).squeeze(1)

                q_mach_raw = torch.cat([hx, selected_op_emb], dim=-1)
                q_mach = self.q_mach_proj(q_mach_raw).unsqueeze(1)

                # base_k_mach = self.machine_embeds.expand(bsz, -1, -1)
                # base_k_mach = mach_dense  # [B, n_m, H]

                curr_op_mach_compat = torch.gather(
                    mask_machine_compat, 1, idx_exp.expand(-1, -1, n_m)
                ).squeeze(1)
                curr_op_mach_compat_list.append(curr_op_mach_compat)  # [Cache]

                curr_op_pt = torch.gather(op_proc_time, 1, idx_exp.expand(-1, -1, n_m)).squeeze(1)
                curr_op_pt = curr_op_pt.masked_fill(~curr_op_mach_compat, 0.0)

                chosen_job_ready = torch.gather(job_ready_time, 1, sel_job_unsq)  # [B, 1]
                est_start = torch.max(chosen_job_ready, machine_avail_time)  # [B, n_m]
                est_comp = est_start + curr_op_pt  # [B, n_m]
                est_comp_on_mach = est_comp.masked_fill(~curr_op_mach_compat, 0.0)

                max_avail = torch.max(machine_avail_time, dim=1, keepdim=True)[0].clamp(min=1.0)
                norm_mach_avail = (machine_avail_time / max_avail).unsqueeze(-1)  # 机器空闲时间 [B, n_m, 1]
                norm_curr_op_pt = (curr_op_pt / max_avail).unsqueeze(-1)  # 耗时代价 [B, n_m, 1]
                norm_est_comp = (est_comp_on_mach / max_avail).unsqueeze(-1)  # 最终完工时间[B, n_m, 1]
                mach_idle_relative = ((max_avail - machine_avail_time) / max_avail).unsqueeze(-1)

                k_mach_dyn = torch.cat([norm_mach_avail, norm_curr_op_pt, norm_est_comp, mach_idle_relative], dim=-1)
                k_mach_dyn_list.append(k_mach_dyn)  # [Cache]

                k_mach_concat = torch.cat([mach_dense, k_mach_dyn], dim=-1)
                k_mach = self.dyn_mach_proj(k_mach_concat)

                # 底层 Glimpse 和 Logits
                glimpse_mach = self.mach_pointer_att(q=q_mach, k=k_mach, v=k_mach)
                u_mach = torch.matmul(glimpse_mach, k_mach.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
                u_mach = u_mach.squeeze(1)  # [B, n_m]

                u_mach_safe = u_mach.masked_fill(batch_is_done.unsqueeze(-1), 0.0)
                u_mach_safe = u_mach_safe.masked_fill(~curr_op_mach_compat & ~batch_is_done.unsqueeze(-1), float('-inf'))

                if rollout:
                    selected_mach = u_mach_safe.max(-1)[1]
                else:
                    selected_mach = Categorical(F.softmax(u_mach_safe, dim=-1)).sample()
                selected_mach = selected_mach.masked_fill(batch_is_done, 0)
                actions_mach_list.append(selected_mach)

                batch_idx = torch.arange(bsz, device=device)
                chosen_pt = op_proc_time[batch_idx, sel_node_idx.squeeze(1), selected_mach]  # [B]

                chosen_job_ready = torch.gather(job_ready_time, 1, sel_job_unsq).squeeze(1)  # [B]
                chosen_mach_avail = torch.gather(machine_avail_time, 1, selected_mach.unsqueeze(1)).squeeze(1)  # [B]

                actual_start = torch.max(chosen_job_ready, chosen_mach_avail)
                actual_comp = actual_start + chosen_pt
                actual_comp = torch.where(batch_is_done, chosen_job_ready, actual_comp)

                job_ready_time.scatter_(1, sel_job_unsq, actual_comp.unsqueeze(1))
                machine_avail_time.scatter_(1, selected_mach.unsqueeze(1), actual_comp.unsqueeze(1))

                one_hot = F.one_hot(selected_job, num_classes=n_j)  # [B, n_j]
                step_active = (~batch_is_done).unsqueeze(-1).long()
                job_next_op_local_idx = job_next_op_local_idx + one_hot * step_active
                mask_job_finished = (job_next_op_local_idx >= job_length)

                sel_mach_expanded = selected_mach.unsqueeze(1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
                selected_mach_emb = torch.gather(mach_dense, 1, sel_mach_expanded).squeeze(1)  # [B, hidden_dim]
                combined_context = torch.cat([selected_op_emb, selected_mach_emb], dim=-1)

                curr_input = self.context_proj(combined_context)
        priority_job_list = torch.stack(actions_job_list, dim=1)
        machine_assign_tensor = torch.stack(actions_mach_list, dim=1)

        if rollout:
            return priority_job_list, machine_assign_tensor, None, None

        T = priority_job_list.size(1)

        sel_node_idx_tensor = torch.stack(sel_node_idx_list, dim=1).squeeze(2)  # [B, T]
        safe_indices_tensor = torch.stack(safe_indices_list, dim=1)  # [B, T, n_j]
        dyn_feats_tensor = torch.stack(dyn_feats_list, dim=1)  # [B, T, n_j, 7]
        mask_job_finished_tensor = torch.stack(mask_job_finished_list, dim=1)  # [B, T, n_j]
        batch_is_done_tensor = torch.stack(batch_is_done_list, dim=1)  # [B, T]
        curr_op_mach_compat_tensor = torch.stack(curr_op_mach_compat_list, dim=1)  # [B, T, n_m]
        k_mach_dyn_tensor = torch.stack(k_mach_dyn_list, dim=1)  # [B, T, n_m, 4]

        # 还原并重构整个序列 GPT 输入
        selected_op_emb_seq = torch.gather(encoder_out, 1,
                                           sel_node_idx_tensor.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        selected_mach_emb_seq = torch.gather(mach_dense, 1,
                                             machine_assign_tensor.unsqueeze(-1).expand(-1, -1, self.hidden_dim))

        combined_context_seq = torch.cat([selected_op_emb_seq, selected_mach_emb_seq], dim=-1)  # [B, T, 2H]
        curr_input_from_actions = self.context_proj(combined_context_seq)  # [B, T, H]

        # 序列构建: [初始 token, action_1 token, ..., action_T-1 token]
        curr_input_0 = (graph_context + self.start_token.expand(bsz, -1)).unsqueeze(1)  # [B, 1, H]
        x_seq_full = torch.cat([curr_input_0, curr_input_from_actions[:, :-1, :]], dim=1)  # [B, T, H]

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.wpe(pos)
        x = self.drop(x_seq_full + pos_emb.unsqueeze(0))
        for block in self.h:
            x = block(x)
        hx_seq = self.ln_f(x)

        #  并行计算所有的工序(Job) 指针网络
        q_op_seq = hx_seq  # [B, T, H]
        glimpse_op_seq = self.op_pointer_att(q=q_op_seq, k=encoder_out, v=encoder_out, mask=mask_padding)

        k_candidates_seq = torch.gather(k_logits_all.unsqueeze(1).expand(-1, T, -1, -1), 2,
                                        safe_indices_tensor.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim))
        cat_feats_seq = torch.cat([k_candidates_seq, dyn_feats_tensor], dim=-1)
        k_candidates_fused_seq = self.fusion_net(cat_feats_seq)

        # einsum 矩阵乘法替代
        glimpse_op_unsq = glimpse_op_seq.unsqueeze(2)  # [B, T, 1, H]
        u_op_seq = torch.matmul(glimpse_op_unsq, k_candidates_fused_seq.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        u_op_seq = torch.tanh(u_op_seq) * self.tanh_clipping / temperature
        u_op_seq = u_op_seq.squeeze(2)  # [B, T, n_j]

        u_op_seq = u_op_seq.masked_fill(mask_job_finished_tensor, float('-inf'))
        u_op_safe_seq = u_op_seq.masked_fill(batch_is_done_tensor.unsqueeze(-1), 0.0)
        u_op_safe_seq = u_op_safe_seq.masked_fill(mask_job_finished_tensor & ~batch_is_done_tensor.unsqueeze(-1),
                                                  float('-inf'))

        m_op_seq = Categorical(probs=F.softmax(u_op_safe_seq, dim=-1))
        # 记录对数概率
        log_probs_op_seq = m_op_seq.log_prob(priority_job_list) * (~batch_is_done_tensor).float()
        entropy_op_seq = m_op_seq.entropy() * (~batch_is_done_tensor).float()

        # 并行计算所有的机器(Machine) 指针网络
        q_mach_raw_seq = torch.cat([hx_seq, selected_op_emb_seq], dim=-1)
        q_mach_seq = self.q_mach_proj(q_mach_raw_seq)  # [B, T, H]

        base_k_mach_seq = mach_dense.unsqueeze(1).expand(-1, T, -1, -1)
        k_mach_concat_seq = torch.cat([base_k_mach_seq, k_mach_dyn_tensor], dim=-1)
        k_mach_seq = self.dyn_mach_proj(k_mach_concat_seq)  # [B, T, n_m, H]

        q_mach_flat = q_mach_seq.view(bsz * T, 1, self.hidden_dim)  # [B*T, 1, H]
        k_mach_flat = k_mach_seq.view(bsz * T, n_m, self.hidden_dim)  # [B*T, n_m, H]
        glimpse_mach_flat = self.mach_pointer_att(q=q_mach_flat, k=k_mach_flat, v=k_mach_flat)
        glimpse_mach_seq = glimpse_mach_flat.view(bsz, T, self.hidden_dim)  # 还原回 [B, T, H]

        glimpse_mach_unsq = glimpse_mach_seq.unsqueeze(2)  # [B, T, 1, H]
        u_mach_seq = torch.matmul(glimpse_mach_unsq, k_mach_seq.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        u_mach_seq = u_mach_seq.squeeze(2)  # [B, T, n_m]

        u_mach_safe_seq = u_mach_seq.masked_fill(batch_is_done_tensor.unsqueeze(-1), 0.0)
        u_mach_safe_seq = u_mach_safe_seq.masked_fill(~curr_op_mach_compat_tensor & ~batch_is_done_tensor.unsqueeze(-1),
                                                      float('-inf'))

        m_mach_seq = Categorical(probs=F.softmax(u_mach_safe_seq, dim=-1))
        # 记录机器的对数概率
        log_probs_mach_seq = m_mach_seq.log_prob(machine_assign_tensor) * (~batch_is_done_tensor).float()
        entropy_mach_seq = m_mach_seq.entropy() * (~batch_is_done_tensor).float()

        # 直接聚合求解
        sum_log_probs = log_probs_op_seq.sum(dim=1) + log_probs_mach_seq.sum(dim=1)
        sum_entropy = entropy_op_seq.sum(dim=1) + entropy_mach_seq.sum(dim=1)

        return priority_job_list, machine_assign_tensor, sum_log_probs, sum_entropy


class FJSPActor(nn.Module):
    def __init__(self, op_input_dim, mach_input_dim, hidden_dim, metadata, n_layers=12, n_heads=12):
        super().__init__()
        self.op_emb = nn.Linear(op_input_dim, hidden_dim)
        self.mach_emb = nn.Linear(mach_input_dim, hidden_dim)

        self.encoder = to_hetero(BaseGNN(hidden_dim), metadata=metadata, aggr='sum')

        self.decoder = GPTDecoder(hidden_dim, n_layers=n_layers, n_heads=n_heads)

    def forward(self, pyg_hetero_batch, mask_machine_compat, op_proc_time, job_length, rollout=False, temperature=1.0):
        """
        pyg_batch: torch_geometric.data.Batch
        """
        # 图编码
        # node_emb_flat = self.encoder(pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr)
        x_dict = {
            'operation': self.op_emb(pyg_hetero_batch['operation'].x),
            'machine': self.mach_emb(pyg_hetero_batch['machine'].x)
        }
        edge_index_dict = pyg_hetero_batch.edge_index_dict
        edge_attr_dict = pyg_hetero_batch.edge_attr_dict if hasattr(pyg_hetero_batch, 'edge_attr_dict') else None

        node_emb_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)

        op_emb_flat = node_emb_dict['operation']
        mach_emb_flat = node_emb_dict['machine']

        x_op_dense, mask_op = to_dense_batch(op_emb_flat, pyg_hetero_batch['operation'].batch)
        mask_padding = ~mask_op

        x_mach_dense, _ = to_dense_batch(mach_emb_flat, pyg_hetero_batch['machine'].batch)

        # 解码
        return self.decoder(x_op_dense, x_mach_dense, mask_padding, mask_machine_compat,
                            op_proc_time, job_length, rollout, temperature)
