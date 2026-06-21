import numpy as np


def uni_instance_gen(n_j, n_m, min_op=None, max_op=None, low=0.01, high=1.0, flexibility=None):
    job_ops = np.random.randint(min_op, max_op + 1, size=n_j)
    proc_times = np.zeros((n_j, max_op, n_m), dtype=np.float32)

    for i in range(n_j):
        n_op_i = job_ops[i]
        for j in range(n_op_i):
            if flexibility is None:
                num_capable_machines = np.random.randint(1, n_m + 1)
            else:
                num_capable_machines = max(1, int(n_m * flexibility))

            capable_machines = np.random.choice(n_m, num_capable_machines, replace=False)

            for m in capable_machines:
                proc_times[i, j, m] = np.random.uniform(low=low, high=high)

    # 兼容掩码
    compat_mask = (proc_times > 0)

    return proc_times, compat_mask, job_ops


def override(fn):
    """
    override decorator
    """
    return fn

if __name__ == '__main__':
    print(uni_instance_gen(n_j=6, n_m=6, min_op=1, max_op=6, low=0.01, high=1.0))
