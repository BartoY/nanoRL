import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import LayerNorm
from torch_geometric.nn import GATv2Conv, to_hetero
from torch_geometric.utils import to_dense_batch
import math


class Attention(nn.Module):
    def __init__(self, q_hidden_dim, k_dim, v_dim, n_head, k_hidden_dim=None, v_hidden_dim=None):
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
        self.proj_output = nn.Linear(v_dim * n_head, self.v_hidden_dim, bias=False)
        # 支持 PyTorch 2.0+ 的 Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, q, k=None, v=None, mask=None):
        if k is None: k = q
        if v is None: v = k

        bsz, q_len, _ = q.size()
        _, k_len, _ = k.size()

        # 使用 view 和 transpose 替代原本的 chunk 和 stack，避免不必要的显存拷贝
        qs = self.proj_q(q).view(bsz, q_len, self.n_head, self.k_dim).transpose(1, 2)  # [B, n_head, q_len, k_dim]
        ks = self.proj_k(k).view(bsz, k_len, self.n_head, self.k_dim).transpose(1, 2)  # [B, n_head, k_len, k_dim]
        vs = self.proj_v(v).view(bsz, k_len, self.n_head, self.v_dim).transpose(1, 2)  # [B, n_head, k_len, v_dim]

        if self.flash:
            # flash attention 的 mask 要求：True 表示保留，False 表示 mask
            # 原 mask_padding 里 True 表示需要被 mask 的 padded 部分，因此这里取反
            attn_mask = None
            if mask is not None:
                attn_mask = (~mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, k_len]
            att = torch.nn.functional.scaled_dot_product_attention(
                qs, ks, vs, attn_mask=attn_mask, is_causal=False
            )
        else:
            normalizer = self.k_dim ** 0.5
            u = torch.matmul(qs, ks.transpose(2, 3)) / normalizer

            if mask is not None:
                # mask shape 转换: [bsz, k_len] -> [bsz, 1, 1, k_len]
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)
                u = u.masked_fill(mask_expanded, float('-inf'))

            att = torch.matmul(torch.softmax(u, dim=-1), vs)

        att = att.transpose(1, 2).contiguous().view(bsz, q_len, self.v_dim * self.n_head)
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
    """nanoGPT-style with Flash Attention + KV cache support"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, cache=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if T > 1:  # causal mask only when needed
                mask = self.bias[:, :, :T, :k.size(2)]
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_cache


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

    def forward(self, x, cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPTConfig:
    def __init__(self, n_embd, n_head, dropout, block_size=2048, bias=False):
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.block_size = block_size
        self.bias = bias


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
        h = self.norm2(F.relu(h) + h_in)

        h_in2 = h
        h = self.conv3(h, edge_index, edge_attr=edge_attr)
        h = self.norm3(F.relu(h) + h_in2)
        return h


class GPTDecoder(nn.Module):
    def __init__(self, hidden_dim=256, n_layers=6, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        config = GPTConfig(n_embd=hidden_dim, n_head=n_heads, dropout=0.0, block_size=2048)

        self.wpe = nn.Embedding(config.block_size, hidden_dim)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(n_layers)])
        self.ln_f = GPTLayerNorm(hidden_dim, bias=config.bias)

        # Pointer networks (lighter)
        head_dim = hidden_dim // n_heads
        self.op_pointer_att = Attention(hidden_dim, head_dim, head_dim, n_heads)
        self.q_mach_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.mach_pointer_att = Attention(hidden_dim, head_dim, head_dim, n_heads,
                                          k_hidden_dim=hidden_dim, v_hidden_dim=hidden_dim)

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

    def forward(self, encoder_out, mach_dense, mask_padding, mask_machine_compat,
                op_proc_time, job_length, rollout=False, temperature=1.0):
        bsz, n_node, _ = encoder_out.shape
        device = encoder_out.device
        n_j = job_length.size(1)
        n_m = mach_dense.size(1)
        max_op = n_node // n_j

        k_logits_all = self.proj_k(encoder_out)
        valid_mask = (~mask_padding).unsqueeze(-1).float()
        graph_context = (encoder_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

        curr_input = graph_context + self.start_token.expand(bsz, -1)

        # KV caches for each layer
        caches = [None] * len(self.h)

        job_next_op_local_idx = torch.zeros(bsz, n_j, dtype=torch.long, device=device)
        mask_job_finished = (job_length == 0)
        job_ready_time = torch.zeros(bsz, n_j, device=device)
        machine_avail_time = torch.zeros(bsz, n_m, device=device)

        job_indices_seq = []
        machine_assign_list = []
        log_probs_op = []
        log_probs_mach = []
        entropy_op = []
        entropy_mach = []

        base_indices = torch.arange(0, n_j * max_op, max_op, device=device).unsqueeze(0).expand(bsz, -1)
        avg_op_pt = (op_proc_time * mask_machine_compat.float()).sum(-1) / mask_machine_compat.float().sum(-1).clamp(min=1.0)

        max_steps = int(job_length.sum(dim=1).max().item())

        for step in range(max_steps):
            # === Decoder step with KV cache ===
            pos = torch.tensor([step], dtype=torch.long, device=device)
            pos_emb = self.wpe(pos).unsqueeze(0).expand(bsz, -1, -1)  # [B, 1, H]

            x = curr_input.unsqueeze(1) + pos_emb          # [B, 1, H]
            x = self.drop(x)

            new_caches = []
            for i, block in enumerate(self.h):
                x, cache = block(x, caches[i])
                new_caches.append(cache)
            caches = new_caches

            hx = self.ln_f(x).squeeze(1)                   # [B, H]

            # === High-level: Job (Operation) selection ===
            q_op = hx.unsqueeze(1)
            glimpse_op = self.op_pointer_att(q=q_op, k=encoder_out, v=encoder_out, mask=mask_padding)

            current_global_indices = base_indices + job_next_op_local_idx
            safe_indices = current_global_indices.clamp(max=n_node - 1)
            safe_idx_exp = safe_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            k_candidates = torch.gather(k_logits_all, 1, safe_idx_exp)

            # Dynamic features (kept but can be further optimized)
            cand_pt_est = torch.gather(avg_op_pt, 1, safe_indices)

            # avg_mach_avail = machine_avail_time.mean(dim=1, keepdim=True).expand(-1, self.n_j)
            # 当前候选工序的机器兼容矩阵
            cand_mach_mask = torch.gather(mask_machine_compat, 1, safe_indices.unsqueeze(-1).expand(-1, -1, n_m))
            avail_time_expanded = machine_avail_time.unsqueeze(1).expand(-1, n_j, -1)
            valid_avail_time = avail_time_expanded.masked_fill(~cand_mach_mask, float('inf'))
            min_comp_mach_avail = valid_avail_time.min(dim=-1)[0]  # [B, n_j]

            min_comp_mach_avail = min_comp_mach_avail.masked_fill(min_comp_mach_avail == float('inf'), 0.0)

            e_start = torch.max(job_ready_time, min_comp_mach_avail)  # 最早开始时间
            e_comp = e_start + cand_pt_est  # 预计完成时间
            wait_time = e_start - job_ready_time  # 工件等待时间
            idle_time = e_start - min_comp_mach_avail  # 机器空闲时间
            ops_left_ratio = (job_length - job_next_op_local_idx) / job_length.clamp(min=1)
            ops_left_ratio = ops_left_ratio.float()  # 剩余工序比例
            pt_ratio = cand_pt_est / (cand_pt_est.mean(dim=-1, keepdim=True) + 1e-5)  # 剩余加工时间

            job_progress = 1.0 - ops_left_ratio.float()
            valid_job_mask = (job_length > 0).float()
            mean_progress = (job_progress * valid_job_mask).sum(dim=1, keepdim=True) / \
                            valid_job_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
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

            cat_feats = torch.cat([k_candidates, dyn_feats], dim=-1)   # define dyn_feats as in original
            k_candidates_fused = self.fusion_net(cat_feats)

            u_op = torch.matmul(glimpse_op, k_candidates_fused.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
            u_op = torch.tanh(u_op) * self.tanh_clipping / temperature
            u_op = u_op.squeeze(1)

            # Masking logic (same as original)
            batch_is_done = mask_job_finished.all(dim=-1)
            u_op = u_op.masked_fill(mask_job_finished, float('-inf'))
            u_op_safe = u_op.masked_fill(batch_is_done.unsqueeze(-1), 0.0)

            if rollout:
                selected_job = u_op_safe.max(-1)[1]
            else:
                probs = F.softmax(u_op_safe, dim=-1)
                m_op = Categorical(probs)
                selected_job = m_op.sample()
                log_probs_op.append(m_op.log_prob(selected_job) * (~batch_is_done).float())
                entropy_op.append(m_op.entropy() * (~batch_is_done).float())

            selected_job = selected_job.masked_fill(batch_is_done, 0)
            job_indices_seq.append(selected_job)

            # === Low-level: Machine selection (same structure) ===
            sel_job_unsq = selected_job.unsqueeze(1)
            sel_node_idx = torch.gather(current_global_indices, 1, sel_job_unsq)
            idx_exp = sel_node_idx.unsqueeze(-1)

            selected_op_emb = torch.gather(encoder_out, 1,
                                           idx_exp.expand(-1, -1, self.hidden_dim)
                                           ).squeeze(1)

            q_mach_raw = torch.cat([hx, selected_op_emb], dim=-1)
            q_mach = self.q_mach_proj(q_mach_raw).unsqueeze(1)

            # base_k_mach = self.machine_embeds.expand(bsz, -1, -1)
            base_k_mach = mach_dense  # [B, n_m, H]

            curr_op_mach_compat = torch.gather(
                mask_machine_compat, 1, idx_exp.expand(-1, -1, n_m)
            ).squeeze(1)

            curr_op_pt = torch.gather(
                op_proc_time, 1, idx_exp.expand(-1, -1, n_m)
            ).squeeze(1)
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

            k_mach_concat = torch.cat([base_k_mach, norm_mach_avail, norm_curr_op_pt,
                                       norm_est_comp, mach_idle_relative], dim=-1)
            k_mach = self.dyn_mach_proj(k_mach_concat)

            # 底层 Glimpse 和 Logits
            glimpse_mach = self.mach_pointer_att(q=q_mach, k=k_mach, v=k_mach)
            u_mach = torch.matmul(glimpse_mach, k_mach.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
            u_mach = u_mach.squeeze(1)  # [B, n_m]

            curr_op_mach_compat = torch.gather(mask_machine_compat, 1,
                                               sel_node_idx.unsqueeze(-1).expand(-1, -1, n_m)).squeeze(1)

            u_mach_safe = u_mach.masked_fill(batch_is_done.unsqueeze(-1), 0.0)
            u_mach_safe = u_mach_safe.masked_fill(~curr_op_mach_compat & ~batch_is_done.unsqueeze(-1), float('-inf'))

            if rollout:
                selected_mach = u_mach_safe.max(-1)[1]
            else:
                m_mach = Categorical(F.softmax(u_mach_safe, dim=-1))
                selected_mach = m_mach.sample()
                log_probs_mach.append(m_mach.log_prob(selected_mach) * (~batch_is_done).float())
                entropy_mach.append(m_mach.entropy() * (~batch_is_done).float())
            selected_mach = selected_mach.masked_fill(batch_is_done, 0)
            machine_assign_list.append(selected_mach)

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

            combined = torch.cat([selected_op_emb, selected_mach_emb], dim=-1)
            curr_input = self.context_proj(combined)

        # Final return (same as original)
        priority_job_list = torch.stack(job_indices_seq, dim=1)
        machine_assign_tensor = torch.stack(machine_assign_list, dim=1)

        if not rollout:
            sum_log_probs = torch.stack(log_probs_op, dim=1).sum(dim=1) + torch.stack(log_probs_mach, dim=1).sum(dim=1)
            sum_entropy = torch.stack(entropy_op, dim=1).sum(dim=1) + torch.stack(entropy_mach, dim=1).sum(dim=1)
            return priority_job_list, machine_assign_tensor, sum_log_probs, sum_entropy
        return priority_job_list, machine_assign_tensor, None, None


class FJSPActor(nn.Module):
    def __init__(self, op_input_dim, mach_input_dim, hidden_dim, metadata, n_layers=12, n_heads=12):
        super().__init__()
        self.op_emb = nn.Linear(op_input_dim, hidden_dim)
        self.mach_emb = nn.Linear(mach_input_dim, hidden_dim)
        self.encoder = to_hetero(BaseGNN(hidden_dim), metadata=metadata, aggr='sum')
        self.decoder = GPTDecoder(hidden_dim, n_layers=n_layers, n_heads=n_heads)

    def forward(self, pyg_hetero_batch, mask_machine_compat, op_proc_time, job_length, rollout=False, temperature=1.0):
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

        return self.decoder(x_op_dense, x_mach_dense, mask_padding, mask_machine_compat,
                            op_proc_time, job_length, rollout, temperature)