from ortools.sat.python import cp_model
import collections

from ortools.sat.python import cp_model
import collections


def fjsp_solver(proc_times, compat_mask, job_length=None, time_limit_seconds=30.0, multiplier=10000):
    n_j, n_op, n_m = proc_times.shape

    model = cp_model.CpModel()

    # 1. 解析数据并转换为整数、计算 Horizon
    horizon = 0
    jobs = []
    for i in range(n_j):
        job = []

        # 【核心修改】：如果有 job_length，只对有效工序进行建模，彻底舍弃 dummy 工序
        actual_op_len = job_length[i] if job_length is not None else n_op

        for j in range(actual_op_len):
            task = []
            max_duration = 0
            for m in range(n_m):
                if compat_mask[i, j, m]:
                    # 浮点数转整数
                    duration = int(round(proc_times[i, j, m] * multiplier))
                    task.append((m, duration))
                    max_duration = max(max_duration, duration)
            job.append(task)
            horizon += max_duration
        jobs.append(job)

    # 2. 建立决策变量
    machine_to_intervals = collections.defaultdict(list)
    starts = {}
    ends = {}
    job_ends = []

    for job_id, job in enumerate(jobs):
        for task_id, task in enumerate(job):
            start = model.NewIntVar(0, horizon, f'start_{job_id}_{task_id}')
            end = model.NewIntVar(0, horizon, f'end_{job_id}_{task_id}')
            starts[(job_id, task_id)] = start
            ends[(job_id, task_id)] = end

            task_presences = []
            for alt_id, (machine, duration) in enumerate(task):
                suffix = f'_j{job_id}_t{task_id}_a{alt_id}_m{machine}'

                l_presence = model.NewBoolVar(f'presence{suffix}')
                l_start = model.NewIntVar(0, horizon, f'start{suffix}')
                l_end = model.NewIntVar(0, horizon, f'end{suffix}')

                l_interval = model.NewOptionalIntervalVar(
                    l_start, duration, l_end, l_presence, f'interval{suffix}')

                task_presences.append(l_presence)
                machine_to_intervals[machine].append(l_interval)

                model.Add(start == l_start).OnlyEnforceIf(l_presence)
                model.Add(end == l_end).OnlyEnforceIf(l_presence)

            # 约束：一个工序必须选一台机器
            model.AddExactlyOne(task_presences)

    # 3. 添加约束
    # 优先级约束
    for job_id, job in enumerate(jobs):
        # 兼容处理：只有 1 道工序的工件不需要前后约束
        if len(job) > 1:
            for task_id in range(len(job) - 1):
                model.Add(ends[(job_id, task_id)] <= starts[(job_id, task_id + 1)])
        # 【重要】：将完工目标的结束变量，指向该工件真实的最后一道工序
        job_ends.append(ends[(job_id, len(job) - 1)])

    # 机器不重叠约束
    for machine in range(n_m):
        model.AddNoOverlap(machine_to_intervals[machine])

    # 4. 目标函数
    makespan = model.NewIntVar(0, horizon, 'makespan')
    for end_var in job_ends:
        model.Add(makespan >= end_var)
    model.Minimize(makespan)

    # 5. 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.Solve(model)

    # 6. 返回结果
    status_name = solver.StatusName(status)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # 将求得的整数 makespan 缩小回原来的浮点数级别
        final_makespan = solver.ObjectiveValue() / multiplier
        return final_makespan, status_name
    else:
        return None, status_name


# --- 测试运行 ---
if __name__ == '__main__':
    from uniform_instance_gen import uni_instance_gen
    proc_times, compat_mask = uni_instance_gen(n_j=10, n_m=10, n_op=10)

    # 2. 传入求解器并获取结果
    makespan, status = fjsp_solver(proc_times, compat_mask)

    # 3. 极简输出
    # print(f"proc_times: {proc_times}")
    print(f"Status: {status}")
    print(f"Makespan: {makespan}")