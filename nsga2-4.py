# QSNSGA2_scheduling_rule_optimizer.py
# Python 3.8+
# pip install numpy matplotlib seaborn

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
from collections import defaultdict

# ------------------------------
# 0. 可配置项
# ------------------------------
RANDOM_SEED = 42
POP_SIZE = 40  # 每个种群大小
GENERATIONS = 60
SCENARIOS_PER_EVAL = 4
STABILITY_EVAL_RUNS = 12
COMPARE_EVAL_RUNS = 200
WEIGHTS_FOR_FINAL_SCORE = np.array([1 / 3, 1 / 3, 1 / 3])

# Q-learning参数
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # 折扣因子
EPSILON = 0.5  # 探索率

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 设置全局字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

# ------------------------------
# 1) 仿真数据(保持不变)
# ------------------------------
machines = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
M_idx = {m: i for i, m in enumerate(machines)}

ops = [
    {'job': 1, 'op': 1, 'machines': [M_idx['M1']], 'p': [3], 'e': [8.2], 'c': [8.2]},
    {'job': 1, 'op': 2, 'machines': [M_idx['M2'], M_idx['M3']], 'p': [6, 4], 'e': [9.1, 7.8], 'c': [9.1, 7.8]},
    {'job': 1, 'op': 3, 'machines': [M_idx['M4']], 'p': [7], 'e': [8.9], 'c': [8.9]},
    {'job': 1, 'op': 4, 'machines': [M_idx['M5']], 'p': [6], 'e': [7.5], 'c': [7.5]},
    {'job': 1, 'op': 5, 'machines': [M_idx['M6']], 'p': [6], 'e': [8.9], 'c': [8.9]},
    {'job': 2, 'op': 1, 'machines': [M_idx['M2'], M_idx['M6']], 'p': [8, 6], 'e': [9.2, 9.5], 'c': [9.2, 9.5]},
    {'job': 2, 'op': 2, 'machines': [M_idx['M1']], 'p': [5], 'e': [7.6], 'c': [7.6]},
    {'job': 2, 'op': 3, 'machines': [M_idx['M3']], 'p': [6], 'e': [8.2], 'c': [8.2]},
    {'job': 2, 'op': 4, 'machines': [M_idx['M2'], M_idx['M5']], 'p': [8, 7], 'e': [9.1, 9.4], 'c': [9.1, 9.4]},
    {'job': 3, 'op': 1, 'machines': [M_idx['M1'], M_idx['M2']], 'p': [4, 6], 'e': [8.9, 9.1], 'c': [8.9, 9.1]},
    {'job': 3, 'op': 2, 'machines': [M_idx['M5']], 'p': [6], 'e': [7.5], 'c': [7.5]},
    {'job': 3, 'op': 3, 'machines': [M_idx['M2']], 'p': [8], 'e': [9.2], 'c': [9.2]},
    {'job': 3, 'op': 4, 'machines': [M_idx['M4']], 'p': [7], 'e': [8.9], 'c': [8.9]},
    {'job': 3, 'op': 5, 'machines': [M_idx['M3']], 'p': [7], 'e': [9.0], 'c': [9.0]},
    {'job': 3, 'op': 6, 'machines': [M_idx['M5'], M_idx['M6']], 'p': [8, 7], 'e': [9.2, 9.8], 'c': [9.2, 9.8]},
]

job_to_ops = {1: [0, 1, 2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12, 13, 14]}
D = {1: 20, 2: 18, 3: 22}
priority = {1: 1.0, 2: 1.0, 3: 1.0}
arrival = {1: 0, 2: 0, 3: 0}
Cmax_dict = {i: 2.0 for i in range(len(machines))}


# ------------------------------
# 2) 规则优先级函数(保持不变)
# ------------------------------
def f_SPT(op_idx, m_idx, now): return 1.0 / (ops[op_idx]['p'][ops[op_idx]['machines'].index(m_idx)] + 1e-9)


def f_EDD(op_idx, m_idx, now): job = ops[op_idx]['job']; return 1.0 / max(1.0, D[job] - now)


def f_CR(op_idx, m_idx, now):
    job = ops[op_idx]['job'];
    p = ops[op_idx]['p'][ops[op_idx]['machines'].index(m_idx)]
    return max(1e-9, (D[job] - now) / (p + 1e-9))


def f_FIFO(op_idx, m_idx, now): job = ops[op_idx]['job']; return 1.0 / (1.0 + arrival[job])


def f_LPT(op_idx, m_idx, now): return float(ops[op_idx]['p'][ops[op_idx]['machines'].index(m_idx)])


def f_SRT(op_idx, m_idx, now): return 1.0 / (ops[op_idx]['p'][ops[op_idx]['machines'].index(m_idx)] + 1e-9)


def f_MWKR(op_idx, m_idx, now):
    job = ops[op_idx]['job'];
    rem = 0.0
    for oi in job_to_ops[job]: rem += min(ops[oi]['p'])
    return float(rem)


def f_NPT(op_idx, m_idx, now):
    p = ops[op_idx]['p'][ops[op_idx]['machines'].index(m_idx)]
    e = ops[op_idx]['e'][ops[op_idx]['machines'].index(m_idx)]
    return 1.0 / (p * e + 1e-9)


rule_funcs = [f_SPT, f_EDD, f_CR, f_FIFO, f_LPT, f_SRT, f_MWKR, f_NPT]
RULE_NAMES = ['SPT', 'EDD', 'CR', 'FIFO', 'LPT', 'SRT', 'MWKR', 'NPT']
K = len(rule_funcs)


def compute_rule_vector_for_op(op_idx, now=0.0):
    out = {}
    for m_idx in ops[op_idx]['machines']:
        vec = np.array([fn(op_idx, m_idx, now) for fn in rule_funcs], dtype=float)
        out[m_idx] = vec
    return out


# ------------------------------
# 3) 仿真器(保持不变)
# ------------------------------
def schedule_simulation(weight_vec, disturbances=None):
    w = np.array(weight_vec, dtype=float)
    if w.sum() == 0:
        w = np.ones(K) / K
    else:
        w = w / w.sum()

    NOW = 0.0
    machine_free_time = [0.0] * len(machines)
    machine_current_op = [None] * len(machines)
    op_status = {i: 'pending' for i in range(len(ops))}
    op_start = {i: None for i in range(len(ops))}
    op_end = {i: None for i in range(len(ops))}
    energy_used = {i: 0.0 for i in range(len(machines))}
    compute_used = {i: 0.0 for i in range(len(machines))}
    ready = set()
    for j, op_idxs in job_to_ops.items(): ready.add(op_idxs[0])
    total_ops = len(ops);
    done_count = 0
    disturbances = disturbances or []
    machine_downtime_until = [0.0] * len(machines)
    max_iter = 10000;
    iter_count = 0

    while done_count < total_ops and iter_count < max_iter:
        iter_count += 1
        for ev in list(disturbances):
            if abs(ev[0] - NOW) < 1e-9 and ev[1] == 'break':
                m_idx, downtime = ev[2], ev[3]
                if machine_free_time[m_idx] <= NOW + 1e-9:
                    machine_free_time[m_idx] = NOW + downtime
                machine_downtime_until[m_idx] = NOW + downtime
                disturbances.remove(ev)

        for j, op_idxs in job_to_ops.items():
            for idx_pos, op_idx in enumerate(op_idxs):
                if op_status[op_idx] == 'pending':
                    if idx_pos == 0:
                        ready.add(op_idx)
                    else:
                        prev = op_idxs[idx_pos - 1]
                        if op_status[prev] == 'done':
                            ready.add(op_idx)

        avail_machines = [i for i, ft in enumerate(machine_free_time)
                          if ft <= NOW + 1e-9 and machine_downtime_until[i] <= NOW + 1e-9]
        if len(avail_machines) == 0:
            next_free = min([ft for ft in machine_free_time if ft > NOW + 1e-9], default=float('inf'))
            next_break = min([ev[0] for ev in disturbances], default=float('inf'))
            next_event = min(next_free, next_break)
            if next_event == float('inf'): break
            NOW = next_event
            continue

        random.shuffle(avail_machines)
        assigned = False
        for m in avail_machines:
            candidates = [op for op in list(ready) if m in ops[op]['machines'] and op_status[op] == 'pending']
            if not candidates: continue
            best_op = None;
            best_score = -1e9
            for op_idx in candidates:
                vec = compute_rule_vector_for_op(op_idx, now=NOW)[m]
                base = vec / (vec.sum() + 1e-9)
                fused = float(np.dot(w, base))
                p = ops[op_idx]['p'][ops[op_idx]['machines'].index(m)]
                score = fused + 0.001 * (1.0 / (p + 1e-9))
                if score > best_score:
                    best_score = score;
                    best_op = op_idx
            if best_op is None: continue
            assigned = True
            m_local_idx = ops[best_op]['machines'].index(m)
            p = ops[best_op]['p'][m_local_idx];
            e = ops[best_op]['e'][m_local_idx];
            c = ops[best_op]['c'][m_local_idx]
            start_t = NOW;
            end_t = NOW + p
            op_start[best_op] = start_t;
            op_end[best_op] = end_t
            op_status[best_op] = 'running'
            machine_current_op[m] = best_op
            machine_free_time[m] = end_t
            if best_op in ready: ready.remove(best_op)

        if not assigned:
            next_free = min([ft for ft in machine_free_time if ft > NOW + 1e-9], default=float('inf'))
            if next_free == float('inf'): break
            NOW = next_free
            for m_idx in range(len(machines)):
                if abs(machine_free_time[m_idx] - NOW) < 1e-9:
                    op_idx = machine_current_op[m_idx]
                    if op_idx is not None and op_status[op_idx] == 'running':
                        op_status[op_idx] = 'done';
                        done_count += 1
                        local = ops[op_idx]['machines'].index(m_idx) if m_idx in ops[op_idx]['machines'] else 0
                        energy_used[m_idx] += ops[op_idx]['e'][local];
                        compute_used[m_idx] += ops[op_idx]['c'][local]
                        machine_current_op[m_idx] = None
            continue
        else:
            future_times = [ft for ft in machine_free_time if ft > NOW + 1e-9]
            if not future_times: break
            NEXT = min(future_times)
            NOW = NEXT
            for m_idx in range(len(machines)):
                if abs(machine_free_time[m_idx] - NOW) < 1e-9:
                    op_idx = machine_current_op[m_idx]
                    if op_idx is not None and op_status[op_idx] == 'running':
                        op_status[op_idx] = 'done';
                        done_count += 1
                        local = ops[op_idx]['machines'].index(m_idx) if m_idx in ops[op_idx]['machines'] else 0
                        energy_used[m_idx] += ops[op_idx]['e'][local];
                        compute_used[m_idx] += ops[op_idx]['c'][local]
                        machine_current_op[m_idx] = None

    if any(op_status[i] != 'done' for i in op_status):
        for i in op_status:
            if op_status[i] != 'done':
                op_end[i] = NOW + (min(ops[i]['p']) if len(ops[i]['p']) > 0 else 1.0)
                op_status[i] = 'done'

    Cmax_time = max([v for v in op_end.values() if v is not None])
    Etotal = sum(energy_used.values())
    avgU = 0.0
    for k in range(len(machines)):
        if Cmax_time > 0:
            avgU += (compute_used[k] / (Cmax_time * Cmax_dict[k]))
    avgU = avgU / len(machines)
    ECCR = float('inf') if avgU <= 0 else Etotal / (avgU + 1e-9)
    TD = 0.0
    max_end_known = max([v for v in op_end.values() if v is not None])
    for j in job_to_ops:
        last_op = job_to_ops[j][-1]
        Ci = op_end[last_op] if op_end[last_op] is not None else max_end_known
        TD += max(0.0, Ci - D[j]) * priority[j]

    return float(Cmax_time), float(ECCR), float(TD)


# ------------------------------
# 4) 种群多样性和收敛性指标
# ------------------------------
def calculate_SPAD(objs):
    """计算种群收敛性(标准差)"""
    if len(objs) <= 1: return 0.0
    objs_arr = np.array(objs)
    return np.std(objs_arr, axis=0).sum()


def calculate_VAR(objs):
    """计算种群多样性(方差)"""
    if len(objs) <= 1: return 0.0
    objs_arr = np.array(objs)
    return np.var(objs_arr, axis=0).sum()


# ------------------------------
# 5) Q-learning Agent
# ------------------------------
class QLearningAgent:
    def __init__(self, n_states=81, n_actions=9):  # 简化动作空间为9(3^2,因为p只增加)
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON

    def discretize_state(self, spad_ratio1, spad_ratio2, var_ratio1, var_ratio2):
        """将连续状态离散化为81个状态之一"""

        def categorize(ratio):
            if ratio < 0.95:
                return 0  # 减少
            elif ratio > 1.05:
                return 2  # 增加
            else:
                return 1  # 不变

        s1 = categorize(spad_ratio1)
        s2 = categorize(spad_ratio2)
        v1 = categorize(var_ratio1)
        v2 = categorize(var_ratio2)

        return s1 * 27 + s2 * 9 + v1 * 3 + v2

    def choose_action(self, state):
        """ε-贪心策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, 8)  # 探索
        else:
            return np.argmax(self.Q[state])  # 利用

    def decode_action(self, action):
        """将动作解码为迁徙参数调整 (只有q和m可以±1或0, p只能+1)"""
        # action: 0-8, 对应q和m的3x3组合
        q_change = (action // 3) - 1  # -1, 0, +1
        m_change = (action % 3) - 1  # -1, 0, +1
        p_change = 1  # p只能增加
        return p_change, q_change, m_change

    def update(self, state, action, reward, next_state):
        """更新Q值"""
        best_next = np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (reward + self.gamma * best_next - self.Q[state, action])


# ------------------------------
# 6) NSGA-II基础函数
# ------------------------------
def dominates(a, b):
    a = np.array(a);
    b = np.array(b)
    return np.all(a <= b) and np.any(a < b)


def nondominated_sort(objs):
    S = [[] for _ in objs];
    n = [0] * len(objs);
    fronts = [[]]
    for p in range(len(objs)):
        for q in range(len(objs)):
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0: fronts[0].append(p)
    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0: Q.append(q)
        i += 1;
        fronts.append(Q)
    if fronts and len(fronts[-1]) == 0: fronts.pop()
    return fronts


def crowding_distance(front_objs):
    l = len(front_objs)
    if l == 0: return []
    m = len(front_objs[0])
    dist = [0.0] * l
    for i in range(m):
        order = sorted(range(l), key=lambda x: front_objs[x][i])
        fmin = front_objs[order[0]][i];
        fmax = front_objs[order[-1]][i]
        dist[order[0]] = dist[order[-1]] = float('inf')
        if fmax - fmin == 0: continue
        for j in range(1, l - 1):
            dist[order[j]] += (front_objs[order[j + 1]][i] - front_objs[order[j - 1]][i]) / (fmax - fmin)
    return dist


def SBX_crossover(a, b, eta=20.0):
    c1 = np.zeros_like(a);
    c2 = np.zeros_like(b)
    for i in range(len(a)):
        u = random.random()
        beta = (2 * u) ** (1.0 / (eta + 1)) if u <= 0.5 else (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
        c1[i] = 0.5 * ((1 + beta) * a[i] + (1 - beta) * b[i])
        c2[i] = 0.5 * ((1 - beta) * a[i] + (1 + beta) * b[i])
    for c in [c1, c2]:
        np.clip(c, 0, 1, out=c)
        if c.sum() == 0:
            c[:] = 1.0 / len(c)
        else:
            c[:] = c / c.sum()
    return c1, c2


def gaussian_mutation(ind, sigma=0.06, pm=0.2):
    out = ind.copy()
    for i in range(len(out)):
        if random.random() < pm:
            out[i] += random.gauss(0, sigma)
    np.clip(out, 0, 1, out=out)
    if out.sum() == 0:
        out[:] = 1.0 / len(out)
    else:
        out[:] = out / out.sum()
    return out


def tournament_selection(pop, pop_objs):
    i1, i2 = random.sample(range(len(pop)), 2)
    if dominates(pop_objs[i1], pop_objs[i2]): return pop[i1]
    if dominates(pop_objs[i2], pop_objs[i1]): return pop[i2]
    return pop[random.choice([i1, i2])]


# ------------------------------
# 7) QSNSGA-II主算法(融合Q-learning和双种群)
# ------------------------------
def qsnsga2_run(pop_size=POP_SIZE, generations=GENERATIONS, scenarios=SCENARIOS_PER_EVAL):
    """
    Q-learning增强的双种群NSGA-II算法
    """
    # 初始化双种群
    pop1 = []  # 精英种群
    pop2 = []  # 随机种群
    for _ in range(pop_size):
        v = np.random.rand(K);
        v = v / v.sum();
        pop1.append(v)
        v = np.random.rand(K);
        v = v / v.sum();
        pop2.append(v)

    # 评估初始种群
    def evaluate_population(pop):
        pop_objs = []
        for ind in pop:
            acc = np.zeros(3)
            for _ in range(scenarios):
                dlist = []
                if random.random() < 0.3:
                    t = random.uniform(2, 8);
                    m = random.choice(range(len(machines)))
                    downtime = random.uniform(1, 3)
                    dlist.append((t, 'break', m, downtime))
                f1, f2, f3 = schedule_simulation(ind, disturbances=dlist)
                acc += np.array([f1, f2, f3])
            pop_objs.append(tuple(acc / scenarios))
        return pop_objs

    pop1_objs = evaluate_population(pop1)
    pop2_objs = evaluate_population(pop2)

    # 初始化Q-learning agent
    agent = QLearningAgent()

    # 迁徙参数初始化
    p = 5  # 迁徙代数(只能增加)
    q = 3  # 迁徙间隔
    m = max(1, pop_size // 10)  # 迁徙个体数

    # 记录初始指标
    spad0_1 = calculate_SPAD(pop1_objs)
    spad0_2 = calculate_SPAD(pop2_objs)
    var0_1 = calculate_VAR(pop1_objs)
    var0_2 = calculate_VAR(pop2_objs)

    # 历史记录
    history_best = []
    pareto_history = []
    diversity_history = []  # 记录多样性变化
    convergence_history = []  # 记录收敛性变化
    migration_history = []  # 记录迁徙事件

    print("开始QSNSGA-II优化...")
    for gen in range(generations):
        # === 种群1进化 ===
        offspring1 = []
        while len(offspring1) < pop_size:
            p1 = tournament_selection(pop1, pop1_objs)
            p2 = tournament_selection(pop1, pop1_objs)
            c1, c2 = SBX_crossover(p1, p2)
            c1 = gaussian_mutation(c1);
            c2 = gaussian_mutation(c2)
            offspring1 += [c1, c2]
        off1_objs = evaluate_population(offspring1)

        R1 = pop1 + offspring1;
        R1_objs = pop1_objs + off1_objs
        fronts1 = nondominated_sort(R1_objs)
        new_pop1 = [];
        new_objs1 = []
        for F in fronts1:
            F_objs = [R1_objs[i] for i in F]
            dists = crowding_distance(F_objs)
            order = sorted(range(len(F)), key=lambda i: dists[i], reverse=True)
            for idx in order:
                if len(new_pop1) < pop_size:
                    new_pop1.append(R1[F[idx]]);
                    new_objs1.append(R1_objs[F[idx]])
        pop1 = new_pop1;
        pop1_objs = new_objs1

        # === 种群2进化 ===
        offspring2 = []
        while len(offspring2) < pop_size:
            p1 = tournament_selection(pop2, pop2_objs)
            p2 = tournament_selection(pop2, pop2_objs)
            c1, c2 = SBX_crossover(p1, p2)
            c1 = gaussian_mutation(c1);
            c2 = gaussian_mutation(c2)
            offspring2 += [c1, c2]
        off2_objs = evaluate_population(offspring2)

        R2 = pop2 + offspring2;
        R2_objs = pop2_objs + off2_objs
        fronts2 = nondominated_sort(R2_objs)
        new_pop2 = [];
        new_objs2 = []
        for F in fronts2:
            F_objs = [R2_objs[i] for i in F]
            dists = crowding_distance(F_objs)
            order = sorted(range(len(F)), key=lambda i: dists[i], reverse=True)
            for idx in order:
                if len(new_pop2) < pop_size:
                    new_pop2.append(R2[F[idx]]);
                    new_objs2.append(R2_objs[F[idx]])
        pop2 = new_pop2;
        pop2_objs = new_objs2

        # === 计算当前状态 ===
        spad_t1 = calculate_SPAD(pop1_objs)
        spad_t2 = calculate_SPAD(pop2_objs)
        var_t1 = calculate_VAR(pop1_objs)
        var_t2 = calculate_VAR(pop2_objs)

        spad_ratio1 = spad_t1 / (spad0_1 + 1e-9)
        spad_ratio2 = spad_t2 / (spad0_2 + 1e-9)
        var_ratio1 = var_t1 / (var0_1 + 1e-9)
        var_ratio2 = var_t2 / (var0_2 + 1e-9)

        state = agent.discretize_state(spad_ratio1, spad_ratio2, var_ratio1, var_ratio2)

        # === Q-learning选择动作并调整参数 ===
        action = agent.choose_action(state)
        p_change, q_change, m_change = agent.decode_action(action)

        p += p_change  # p只增加
        q = max(1, min(10, q + q_change))
        m = max(1, min(pop_size // 4, m + m_change))

        # === 迁徙操作 ===
        if (gen + 1) % q == 0 and gen > 0:
            # 从种群1选择m个最优个体
            fronts1_cur = nondominated_sort(pop1_objs)
            elite1_indices = fronts1_cur[0][:m] if len(fronts1_cur[0]) >= m else fronts1_cur[0]

            # 随机选择种群2的m个个体进行替换
            rand2_indices = random.sample(range(len(pop2)), min(m, len(elite1_indices)))

            # 执行迁徙
            for i, elite_idx in enumerate(elite1_indices):
                if i < len(rand2_indices):
                    pop2[rand2_indices[i]] = pop1[elite_idx].copy()
                    pop2_objs[rand2_indices[i]] = pop1_objs[elite_idx]

            migration_history.append((gen, m, len(elite1_indices)))
            print(f"  Gen {gen}: 迁徙 {len(elite1_indices)} 个精英个体到随机种群")

        # === 计算奖励 ===
        reward = 0.0
        if spad_ratio1 < 1.0:
            reward += 0.5
        elif spad_ratio1 > 1.0:
            reward -= 1.0

        if spad_ratio2 < 1.0:
            reward += 0.5
        elif spad_ratio2 > 1.0:
            reward -= 1.0

        if var_ratio1 < 1.0:
            reward += 0.5
        elif var_ratio1 > 1.0:
            reward -= 1.0

        if var_ratio2 < 1.0:
            reward += 0.5
        elif var_ratio2 > 1.0:
            reward -= 1.0

        # === 更新Q值 ===
        spad_next1 = calculate_SPAD(pop1_objs)
        spad_next2 = calculate_SPAD(pop2_objs)
        var_next1 = calculate_VAR(pop1_objs)
        var_next2 = calculate_VAR(pop2_objs)

        next_state = agent.discretize_state(
            spad_next1 / (spad0_1 + 1e-9),
            spad_next2 / (spad0_2 + 1e-9),
            var_next1 / (var0_1 + 1e-9),
            var_next2 / (var0_2 + 1e-9)
        )
        agent.update(state, action, reward, next_state)

        # === 合并两个种群得到总Pareto前沿 ===
        combined_pop = pop1 + pop2
        combined_objs = pop1_objs + pop2_objs
        combined_fronts = nondominated_sort(combined_objs)
        pareto_idx = combined_fronts[0]

        # === 记录历史 ===
        sums = [sum(combined_objs[i]) for i in range(len(combined_objs))]
        history_best.append(min(sums))
        pareto_history.append([combined_objs[i] for i in pareto_idx])
        diversity_history.append((var_t1 + var_t2) / 2)
        convergence_history.append((spad_t1 + spad_t2) / 2)

        if gen % 10 == 0 or gen == generations - 1:
            print(f"Gen {gen}: best_sum={history_best[-1]:.3f}, Pareto_size={len(pareto_idx)}, "
                  f"div={diversity_history[-1]:.3f}, conv={convergence_history[-1]:.3f}, "
                  f"p={p}, q={q}, m={m}")

    # === 最终Pareto前沿 ===
    combined_pop = pop1 + pop2
    combined_objs = pop1_objs + pop2_objs
    final_fronts = nondominated_sort(combined_objs)
    pareto_idx = final_fronts[0]
    pareto_inds = [combined_pop[i] for i in pareto_idx]
    pareto_objs = [combined_objs[i] for i in pareto_idx]

    # === 稳定性评分 ===
    scored = []
    for ind, obj in zip(pareto_inds, pareto_objs):
        vals = []
        for _ in range(STABILITY_EVAL_RUNS):
            dlist = []
            if random.random() < 0.3:
                t = random.uniform(2, 8);
                m_idx = random.choice(range(len(machines)))
                downtime = random.uniform(1, 3)
                dlist.append((t, 'break', m_idx, downtime))
            f1, f2, f3 = schedule_simulation(ind, disturbances=dlist)
            vals.append([f1, f2, f3])
        vals = np.array(vals)
        meanv = vals.mean(axis=0)
        varv = vals.var(axis=0).sum()
        score = meanv.sum() + 0.5 * varv
        scored.append((score, meanv, varv, ind))

    scored = sorted(scored, key=lambda x: x[0])
    selected = scored[:3]

    return (pareto_inds, pareto_objs, selected, history_best,
            pareto_history, diversity_history, convergence_history, migration_history)


# ------------------------------
# 8) 可视化函数(改进版)
# ------------------------------
def plot_3d_pareto(pareto_objs, title="3D Pareto前沿"):
    """绘制3D Pareto前沿图"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if len(pareto_objs) > 0:
        pareto_arr = np.array(pareto_objs)
        scatter = ax.scatter(pareto_arr[:, 0], pareto_arr[:, 1], pareto_arr[:, 2],
                             c=pareto_arr[:, 0], cmap='viridis', s=100,
                             alpha=0.8, edgecolors='black', linewidth=0.5)
        fig.colorbar(scatter, ax=ax, label='Makespan (f1)', shrink=0.6)

    ax.set_xlabel('Makespan (f1)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ECCR (f2)', fontsize=11, fontweight='bold')
    ax.set_zlabel('TD (f3)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_convergence_curves(history_best, diversity_history, convergence_history):
    """绘制收敛曲线、多样性和收敛性指标"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 最佳加权和收敛曲线
    axes[0, 0].plot(history_best, linewidth=2, color='#2E86AB', label='Min Weighted Sum')
    axes[0, 0].set_xlabel('代数 (Generation)', fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel('最小加权和目标值', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('收敛曲线 (Convergence Curve)', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].legend(fontsize=9)

    # 2. 多样性变化
    axes[0, 1].plot(diversity_history, linewidth=2, color='#A23B72', label='Diversity (VAR)')
    axes[0, 1].set_xlabel('代数 (Generation)', fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel('多样性指标 (VAR)', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('种群多样性变化', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    axes[0, 1].legend(fontsize=9)

    # 3. 收敛性变化
    axes[1, 0].plot(convergence_history, linewidth=2, color='#F18F01', label='Convergence (SPAD)')
    axes[1, 0].set_xlabel('代数 (Generation)', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('收敛性指标 (SPAD)', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('种群收敛性变化', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].legend(fontsize=9)

    # 4. Pareto前沿规模变化
    pareto_sizes = [len(pf) for pf in history_best]  # 这里用history记录的长度模拟
    axes[1, 1].plot(range(len(history_best)),
                    [len(history_best)] * len(history_best),
                    linewidth=2, color='#6A994E', label='Population Size')
    axes[1, 1].set_xlabel('代数 (Generation)', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('种群规模', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('种群规模稳定性', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_objective_boxplots(all_results, labels):
    """绘制候选方案的目标函数箱线图对比"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    obj_names = ['Makespan (f1)', 'ECCR (f2)', 'TD (f3)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i in range(3):
        data = [all_results[j][:, i] for j in range(len(all_results))]
        bp = axes[i].boxplot(data, labels=labels, patch_artist=True,
                             notch=True, showmeans=True,
                             boxprops=dict(facecolor=colors[i], alpha=0.7),
                             medianprops=dict(color='darkred', linewidth=2),
                             meanprops=dict(marker='D', markerfacecolor='yellow',
                                            markeredgecolor='black', markersize=8))

        axes[i].set_ylabel(obj_names[i], fontsize=11, fontweight='bold')
        axes[i].set_title(f'{obj_names[i]} 对比', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[i].set_xlabel('候选方案', fontsize=10, fontweight='bold')

    plt.suptitle('Top-3候选方案多目标性能对比 (箱线图)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_radar_comparison(all_results):
    """绘制归一化雷达图对比"""
    all_means = np.array([all_results[i].mean(axis=0) for i in range(len(all_results))])
    mins = all_means.min(axis=0)
    maxs = all_means.max(axis=0)
    denom = np.where(maxs - mins == 0, 1e-9, maxs - mins)
    normalized = (all_means - mins) / denom

    angles = np.linspace(0, 2 * math.pi, 4)[:-1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    colors = ['#E63946', '#06FFA5', '#457B9D']
    for i in range(len(all_results)):
        vals = normalized[i, :].tolist()
        vals += vals[:1]
        angs = list(angles) + [angles[0]]
        ax.plot(angs, vals, linewidth=2.5, label=f'候选方案 {i}',
                color=colors[i], marker='o', markersize=8)
        ax.fill(angs, vals, alpha=0.15, color=colors[i])

    ax.set_xticks(angles)
    ax.set_xticklabels(['Makespan', 'ECCR', 'TD'], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('归一化目标雷达图对比\n(越小越好→归一化值越低)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_migration_events(migration_history, generations):
    """可视化迁徙事件"""
    if not migration_history:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    gens = [m[0] for m in migration_history]
    sizes = [m[1] for m in migration_history]

    ax.scatter(gens, sizes, s=150, c='#E63946', alpha=0.7,
               edgecolors='black', linewidth=1.5, marker='D', label='迁徙事件')
    ax.plot(gens, sizes, linestyle='--', color='#457B9D', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('代数 (Generation)', fontsize=11, fontweight='bold')
    ax.set_ylabel('迁徙个体数', fontsize=11, fontweight='bold')
    ax.set_title('双种群迁徙事件时间线', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    ax.set_xlim(-2, generations + 2)

    plt.tight_layout()
    plt.show()


# ------------------------------
# 9) 主流程
# ------------------------------
if __name__ == "__main__":
    start_time = time.time()

    print("=" * 70)
    print("  QSNSGA-II: Q-learning增强的双种群NSGA-II调度规则优化算法")
    print("=" * 70)

    (pareto_inds, pareto_objs, selected, history_best,
     pareto_history, diversity_history, convergence_history,
     migration_history) = qsnsga2_run()
    print(f"\n{'=' * 70}")
    print(f"QSNSGA-II优化完成! 用时: {time.time() - start_time:.1f}秒")
    print(f"最终Pareto前沿规模: {len(pareto_inds)}")
    print(f"{'=' * 70}\n")

    # === 可视化1: 3D Pareto前沿 ===
    plot_3d_pareto(pareto_objs, title="QSNSGA-II 3D Pareto前沿")


    # === 可视化2: 收敛曲线和指标 ===
    # 修复后的plot_convergence_curves函数
    def plot_convergence_curves(history_best, diversity_history, convergence_history):
        """绘制收敛曲线 - 修复版本"""
        generations = list(range(len(history_best)))

        # 修正：检查history_best中元素的类型
        if len(history_best) > 0 and hasattr(history_best[0], '__len__'):
            # 如果元素是序列类型（如列表），计算长度
            pareto_sizes = [len(pf) for pf in history_best]
            y_label = 'Pareto前沿规模'
            title_suffix = 'Pareto前沿演化'
        else:
            # 如果元素是数值类型，直接使用数值本身
            pareto_sizes = [float(pf) for pf in history_best]
            y_label = '适应度值'
            title_suffix = '最佳适应度演化'

        plt.figure(figsize=(15, 5))

        # 子图1: Pareto前沿规模或最佳个体适应度
        plt.subplot(1, 3, 1)
        plt.plot(generations, pareto_sizes, 'b-', linewidth=2, label=y_label)
        plt.xlabel('代数')
        plt.ylabel(y_label)
        plt.title(title_suffix)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 子图2: 多样性指标
        plt.subplot(1, 3, 2)
        plt.plot(generations, diversity_history, 'g-', linewidth=2, label='多样性')
        plt.xlabel('代数')
        plt.ylabel('多样性指标')
        plt.title('多样性演化')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 子图3: 收敛性指标
        plt.subplot(1, 3, 3)
        plt.plot(generations, convergence_history, 'r-', linewidth=2, label='收敛性')
        plt.xlabel('代数')
        plt.ylabel('收敛性指标')
        plt.title('收敛性演化')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()


    plot_convergence_curves(history_best, diversity_history, convergence_history)

    # === 可视化3: 迁徙事件 ===
    plot_migration_events(migration_history, GENERATIONS)

    # === 显示Top-3候选方案 ===
    print("\n" + "=" * 70)
    print("  Top-3 候选方案 (稳定性评分后)")
    print("=" * 70)
    selected_weights = [item[3] for item in selected]
    selected_means = [item[1] for item in selected]

    for i, (s_m, w) in enumerate(zip(selected_means, selected_weights)):
        print(f"\n候选方案 {i}:")
        print(f"  均值目标: f1={s_m[0]:.3f}, f2={s_m[1]:.3f}, f3={s_m[2]:.3f}")
        print(f"  规则权重: {dict(zip(RULE_NAMES, [round(float(x), 3) for x in w]))}")

    # === 大量场景评估对比 ===
    print(f"\n{'=' * 70}")
    print(f"  对Top-3候选方案进行{COMPARE_EVAL_RUNS}次场景评估...")
    print(f"{'=' * 70}")

    all_results = []
    for idx, w in enumerate(selected_weights):
        res = np.zeros((COMPARE_EVAL_RUNS, 3))
        for s in range(COMPARE_EVAL_RUNS):
            dlist = []
            if random.random() < 0.3:
                t = random.uniform(2, 8)
                m_idx = random.choice(range(len(machines)))
                downtime = random.uniform(1, 3)
                dlist.append((t, 'break', m_idx, downtime))
            f1, f2, f3 = schedule_simulation(w, disturbances=dlist)
            res[s, :] = [f1, f2, f3]
        all_results.append(res)
        print(f"候选方案 {idx} 评估完成")

    # === 可视化4: 箱线图对比 ===
    labels = ['候选方案0', '候选方案1', '候选方案2']
    plot_objective_boxplots(all_results, labels)

    # === 可视化5: 雷达图对比 ===
    plot_radar_comparison(all_results)

    # === 选择最佳方案 ===
    weighted_scores = []
    print(f"\n{'=' * 70}")
    print("  加权评分 (用户可调整权重)")
    print(f"{'=' * 70}")
    for i in range(3):
        means = all_results[i].mean(axis=0)
        ws = np.dot(WEIGHTS_FOR_FINAL_SCORE, means)
        weighted_scores.append(ws)
        print(f"候选方案 {i}: 均值 f1={means[0]:.3f}, f2={means[1]:.3f}, f3={means[2]:.3f}, "
              f"加权得分={ws:.3f}")

    best_idx = int(np.argmin(weighted_scores))
    print(f"\n{'=' * 70}")
    print(f"  ★ 最佳候选方案: 候选方案 {best_idx} (加权得分最低)")
    print(f"{'=' * 70}\n")


    # === 映射到TF-PDAG初始概率 ===
    def map_weights_to_P0(ind):
        P0 = {}
        for oi in range(len(ops)):
            machines_list = ops[oi]['machines']
            scores_m = []
            for m in machines_list:
                vec = np.array([fn(oi, m, 0.0) for fn in rule_funcs], dtype=float)
                base = vec / (vec.sum() + 1e-9)
                fused = float(np.dot(ind, base))
                scores_m.append(fused)
            scores_m = np.array(scores_m)
            probs = scores_m / (scores_m.sum() + 1e-9) if scores_m.sum() > 0 else np.ones_like(scores_m) / len(scores_m)
            P0[oi] = {machines_list[i]: float(probs[i]) for i in range(len(machines_list))}
        return P0


    best_P0 = map_weights_to_P0(selected_weights[best_idx])
    print("TF-PDAG初始机器选择概率 P0 (最佳方案):")
    print("-" * 70)
    for oi in range(len(ops)):
        job = ops[oi]['job']
        op_num = ops[oi]['op']
        probs_str = ", ".join([f"M{m + 1}:{best_P0[oi][m]:.3f}"
                               for m in sorted(best_P0[oi].keys())])
        print(f"  J{job}-O{op_num} → {probs_str}")

    print("\n" + "=" * 70)
    print("  优化完成! 可将上述P0用于初始化TF-PDAG或MAPPO Actor先验")
    print("=" * 70)