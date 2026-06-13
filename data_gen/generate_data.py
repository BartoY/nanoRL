import numpy as np
from uniform_instance_gen import uni_instance_gen

j = 10
m = 10
min_op = 6
max_op = 10
l = 0.01
h = 1.00
flexibility = 0.5
batch_size = 128
seed = 200

np.random.seed(seed)

batch_proc_times = []
batch_compat_masks = []
batch_j_len = []

for _ in range(batch_size):
    pt, mask, jl = uni_instance_gen(n_j=j, n_m=m, min_op=min_op, max_op=max_op, low=l, high=h, flexibility=flexibility)

    batch_proc_times.append(pt)
    batch_compat_masks.append(mask)
    batch_j_len.append(jl)
batch_proc_times = np.array(batch_proc_times, dtype=np.float32)
batch_compat_masks = np.array(batch_compat_masks, dtype=np.bool_)
batch_job_lengths = np.array(batch_j_len, dtype=np.int32)

# print(batch_compat_masks,batch_proc_times)
file_name = f'GenData_{j}_{m}_{max_op}_Seed{seed}_bsz{batch_size}.npz'
np.savez(file_name, proc_times=batch_proc_times, compat_masks=batch_compat_masks, job_length=batch_job_lengths)
