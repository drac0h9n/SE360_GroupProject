# backend.py
# 包含核心计算逻辑：贪心算法和CP-SAT（ILP）求解器
# 版本已修改，以支持“每个 j-子集至少被覆盖 y 次”的需求。

import random
import time
import math
import multiprocessing as mp
import queue # 用于进程间通信
from itertools import combinations, chain
from collections import Counter, defaultdict # 导入 defaultdict

# --- OR-Tools CP-SAT 相关导入 ---
try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    print("警告：未能导入 ortools.sat.python.cp_model。")
    print("CP-SAT (ILP) 求解器将不可用。请确保已安装 Google OR-Tools:")
    print("python -m pip install --upgrade --user ortools")
    HAS_ORTOOLS = False
# --- OR-Tools 导入结束 ---

# --- 辅助函数 ---

def comb(n, k):
    """计算组合数 C(n, k)"""
    if k < 0 or k > n:
        return 0
    try:
        # math.comb 在某些边界情况可能抛出 ValueError (例如非常大的数)
        return math.comb(n, k)
    except ValueError:
        print(f"警告: comb({n}, {k}) 计算溢出或参数无效，返回 0。")
        return 0 # 或抛出异常，取决于如何处理

def ksubsets(items, k):
    """生成 `items` 的所有大小为 `k` 的子集"""
    return list(combinations(items, k))

# --- 运行计数器 ---
class RunCounter:
    """跟踪每个 (m, n, k, j, s) 参数组合的运行次数"""
    def __init__(self):
        self._counts = defaultdict(int)
        self._lock = mp.Lock()

    def get_next_run_index(self, m, n, k, j, s):
        """获取给定参数组合的下一个运行索引 (从 1 开始)，并自增计数"""
        key = (m, n, k, j, s)
        with self._lock:
            self._counts[key] += 1
            return self._counts[key]
    # ... (reset 方法等可以保持不变) ...

# --- 贪心算法 (已修改以支持 y 次覆盖) ---

def greedy_cover(args):
    """
    贪心算法求解覆盖问题，要求每个 j-子集被覆盖至少 y 次。
    参数: (q, univ, k, j, s, y, run_idx)
         q: 结果队列
         univ: 元素全集 (n 个元素)
         k: 覆盖块大小
         j: 被覆盖子集大小
         s: 交集大小阈值 (j-子集与k-块交集 >= s)
         y: **必须覆盖的次数** (每个 j-子集需要被满足条件的 k-块覆盖至少 y 次)
         run_idx: 运行索引
    """
    q, univ, k, j, s, y, run_idx = args # 解包参数
    n = len(univ)
    start_time = time.time()
    result = {'alg': 'Greedy', 'status': 'INIT', 'sets': [], 'time': 0, 'j_subsets_total': 0, 'j_subsets_covered': 0, 'coverage_target': y} # 添加 coverage_target

    print(f"[Greedy-{run_idx}] 开始执行贪心算法 (覆盖目标 y={y})...")

    # 基本参数检查
    if y <= 0:
        print(f"[Greedy-{run_idx}] 错误：覆盖次数 y ({y}) 必须大于 0。")
        result['status'] = 'ERROR_INVALID_Y'
        result['error_message'] = f"覆盖次数 y ({y}) 必须大于 0。"
        result['time'] = time.time() - start_time
        q.put(result)
        return

    try:
        # 1. 生成所有可能的 k-子集 (潜在的覆盖块)
        all_k_subsets_tuples = list(combinations(univ, k))
        all_k_subsets = [set(subset) for subset in all_k_subsets_tuples]
        print(f"[Greedy-{run_idx}] 生成了 {len(all_k_subsets)} 个候选 k-子集。")

        # 2. 生成所有需要被覆盖的 j-子集
        target_j_subsets_tuples = list(combinations(univ, j))
        target_j_subsets = [set(subset) for subset in target_j_subsets_tuples]
        num_j_subsets = len(target_j_subsets)
        result['j_subsets_total'] = num_j_subsets
        print(f"[Greedy-{run_idx}] 需要覆盖 {num_j_subsets} 个 j-子集，每个至少 {y} 次。")

        if num_j_subsets == 0:
             print(f"[Greedy-{run_idx}] 没有需要覆盖的 j-子集，直接完成。")
             result['status'] = 'SUCCESS' # 没有目标，视为成功
             result['time'] = time.time() - start_time
             q.put(result)
             return

        # 3. 预计算每个 k-子集能覆盖哪些 j-子集 (基于交集大小 >= s)
        k_subset_covers_j_indices = defaultdict(list)
        j_subset_covered_by_k_indices = defaultdict(list) # 反向映射，j 被哪些 k 覆盖
        for idx_j, j_subset in enumerate(target_j_subsets):
            for idx_k, k_subset in enumerate(all_k_subsets):
                if len(j_subset.intersection(k_subset)) >= s:
                    k_subset_covers_j_indices[idx_k].append(idx_j)
                    j_subset_covered_by_k_indices[idx_j].append(idx_k)
        print(f"[Greedy-{run_idx}] 预计算覆盖关系完成。")

        # 检查是否有 j-子集根本无法被覆盖 y 次
        for idx_j in range(num_j_subsets):
             potential_covers = len(j_subset_covered_by_k_indices[idx_j])
             if potential_covers < y:
                 print(f"[Greedy-{run_idx}] 错误：j-子集 {idx_j} ({target_j_subsets[idx_j]}) 最多只能被 {potential_covers} 个 k-子集覆盖，无法满足 y={y} 的要求。问题不可行。")
                 result['status'] = 'INFEASIBLE_Y_TARGET'
                 result['error_message'] = f"j-子集 {idx_j} 无法被覆盖 {y} 次。"
                 result['time'] = time.time() - start_time
                 q.put(result)
                 return

        # 4. 贪心选择过程 (面向 y 次覆盖)
        selected_k_subset_indices = [] # 存储选中的 k-子集的索引
        # 跟踪每个 j-子集的当前覆盖次数
        j_subset_coverage_count = [0] * num_j_subsets
        # 跟踪哪些 j-子集还需要更多覆盖 (覆盖次数 < y)
        needs_more_coverage_j_indices = set(range(num_j_subsets))
        print(f"[Greedy-{run_idx}] 开始贪心选择，目标 y={y}...")

        iteration = 0
        # 循环直到所有 j-子集的覆盖次数都达到 y
        while needs_more_coverage_j_indices:
            iteration += 1
            best_k_subset_idx = -1
            max_newly_satisfied_benefit = -1 # 选择此 k-块能让多少个 j-子集 *新* 满足 y 次覆盖？(备用启发式)
            max_coverage_increase_count = -1 # 选择此 k-块能为 *未满足* 的 j-子集增加多少次覆盖？(主要启发式)
            indices_covered_by_best = [] # 最佳块能覆盖的、当前未满足的 j-子集索引

            # 遍历所有*尚未选择*的候选 k-子集（避免重复选择相同的块？）
            # 另一种策略是允许重复选择，如果选择一个块多次有助于满足目标
            # **当前策略：允许选择所有块，每次评估其边际贡献**
            # available_k_indices = set(range(len(all_k_subsets))) - set(selected_k_subset_indices) # 如果不允许重复选择

            for idx_k in range(len(all_k_subsets)):
                # 计算当前 k-子集能为哪些 *仍需覆盖* 的 j-子集提供覆盖
                # eligible_j_indices = k_subset_covers_j_indices[idx_k] # 所有能覆盖的
                relevant_j_indices = set(k_subset_covers_j_indices[idx_k]).intersection(needs_more_coverage_j_indices)

                current_coverage_increase_count = len(relevant_j_indices)

                # 贪心策略：选择能为当前未满足的 j-子集提供最多覆盖次数的 k-子集
                if current_coverage_increase_count > max_coverage_increase_count:
                    max_coverage_increase_count = current_coverage_increase_count
                    best_k_subset_idx = idx_k
                    # indices_covered_by_best = list(relevant_j_indices) # 记录这次选择能帮助哪些未满足的j集

            # 如果在一轮中找不到任何能增加覆盖的 k-子集 (理论上不应发生，除非前面检查疏漏)
            if best_k_subset_idx == -1:
                print(f"[Greedy-{run_idx}] 警告：在迭代 {iteration} 中找不到能为任何未满足覆盖的 j-子集增加覆盖的 k-子集。可能存在问题或算法逻辑错误。")
                result['status'] = 'FAILED_INCOMPLETE_COVER'
                break # 退出循环

            # 选择最佳的 k-子集
            selected_k_subset_indices.append(best_k_subset_idx)
            chosen_k_subset = all_k_subsets[best_k_subset_idx]
            print(f"[Greedy-{run_idx}] 迭代 {iteration}: 选择块 {best_k_subset_idx} {list(sorted(chosen_k_subset))}, "
                  f"为 {max_coverage_increase_count} 个未满足的 j-子集增加了覆盖。")

            # 更新受影响的 j-子集的覆盖计数
            newly_satisfied_count = 0
            j_indices_affected_this_round = k_subset_covers_j_indices[best_k_subset_idx]
            for idx_j in j_indices_affected_this_round:
                # 只更新那些还需要覆盖的
                if idx_j in needs_more_coverage_j_indices:
                    j_subset_coverage_count[idx_j] += 1
                    # 检查是否刚刚满足 y 次覆盖
                    if j_subset_coverage_count[idx_j] >= y:
                        needs_more_coverage_j_indices.remove(idx_j) # 从待覆盖集合中移除
                        newly_satisfied_count += 1

            print(f"[Greedy-{run_idx}]   -> 本轮选择后 {newly_satisfied_count} 个 j-子集达到 y={y} 覆盖目标。")
            print(f"[Greedy-{run_idx}]   -> 剩余 {len(needs_more_coverage_j_indices)} 个 j-子集待满足。")

            # 添加一个迭代次数上限防止死循环 (尽管理论上应该能结束)
            if iteration > num_j_subsets * y * 2 : # 一个比较宽松的估计上限
                  print(f"[Greedy-{run_idx}] 警告：迭代次数过多 ({iteration})，可能陷入循环或收敛缓慢。提前终止。")
                  result['status'] = 'FAILED_ITERATION_LIMIT'
                  break

        # 5. 形成结果
        if not needs_more_coverage_j_indices: # 如果所有 j-子集都满足了 y 次覆盖
            chosen_sets_list = [list(sorted(list(all_k_subsets[idx]))) for idx in selected_k_subset_indices]
            result['sets'] = chosen_sets_list
            result['status'] = 'SUCCESS' # **确认这个状态字符串与 run 方法中的检查一致**
            result['j_subsets_covered'] = num_j_subsets # 表示所有 j-子集都满足了条件
            print(f"[Greedy-{run_idx}] 贪心算法成功完成 y={y} 覆盖，共选择 {len(chosen_sets_list)} 个集合。")
        else:
            # 如果循环退出但仍有未满足的
            chosen_sets_list = [list(sorted(list(all_k_subsets[idx]))) for idx in selected_k_subset_indices]
            result['sets'] = chosen_sets_list
            result['j_subsets_covered'] = num_j_subsets - len(needs_more_coverage_j_indices) # 满足条件的 j-子集数量
            if result['status'] == 'INIT': # 如果状态未被其他失败原因覆盖
                 result['status'] = 'FAILED_INCOMPLETE_COVER' # 默认失败状态
            print(f"[Greedy-{run_idx}] 贪心算法结束，状态: {result['status']}。{result['j_subsets_covered']}/{num_j_subsets} 个 j-子集满足了 y={y} 覆盖。")

    except Exception as e:
        print(f"[Greedy-{run_idx}] 贪心算法执行出错: {e}")
        import traceback
        traceback.print_exc()
        result['status'] = 'ERROR'
        result['error_message'] = str(e)

    finally:
        # 计算耗时并放入队列
        result['time'] = time.time() - start_time
        try:
            q.put(result)
            print(f"[Greedy-{run_idx}] 结果已放入队列。耗时: {result['time']:.2f} 秒。")
        except Exception as qe:
             print(f"[Greedy-{run_idx}] 错误：无法将结果放入队列: {qe}")

# --- CP-SAT (ILP) 求解器 (已修改以支持 y 次覆盖) ---

def cpsat_cover(args):
    """
    使用 OR-Tools CP-SAT 求解器解决覆盖问题，要求每个 j-子集被覆盖至少 y 次。
    参数: (q, univ, k, j, s, y, run_idx, timeout_solver)
         q: 结果队列
         univ: 元素全集
         k: 覆盖块大小
         j: 被覆盖子集大小
         s: 交集大小阈值 (>= s)
         y: **必须覆盖的次数**
         run_idx: 运行索引
         timeout_solver: CP-SAT 求解器的内部超时时间（秒）
    """
    if not HAS_ORTOOLS:
        q.put({'alg': 'ILP', 'status': 'ERROR_MISSING_ORTOOLS', 'sets': [], 'time': 0, 'coverage_target': y})
        return

    q, univ, k, j, s, y, run_idx, timeout_solver = args # 解包参数
    n = len(univ)
    start_time = time.time()
    result = {'alg': 'ILP', 'status': 'INIT', 'sets': [], 'time': 0, 'coverage_target': y} # 添加 coverage_target

    print(f"[ILP-{run_idx}] 开始执行 CP-SAT 求解器 (覆盖目标 y={y})... Timeout={timeout_solver}s")

    # 基本参数检查
    if y <= 0:
        print(f"[ILP-{run_idx}] 错误：覆盖次数 y ({y}) 必须大于 0。")
        result['status'] = 'ERROR_INVALID_Y'
        result['error_message'] = f"覆盖次数 y ({y}) 必须大于 0。"
        result['time'] = time.time() - start_time
        q.put(result)
        return

    try:
        # 1. 生成所有可能的 k-子集 (变量)
        all_k_subsets_tuples = list(combinations(univ, k))
        all_k_subsets = [frozenset(subset) for subset in all_k_subsets_tuples]
        num_k_subsets = len(all_k_subsets)
        k_subset_indices = {subset: i for i, subset in enumerate(all_k_subsets)}
        print(f"[ILP-{run_idx}] 生成了 {num_k_subsets} 个候选 k-子集 (变量)。")

        # 2. 生成所有需要被覆盖的 j-子集 (约束)
        target_j_subsets_tuples = list(combinations(univ, j))
        target_j_subsets = [frozenset(subset) for subset in target_j_subsets_tuples]
        num_j_subsets = len(target_j_subsets)
        print(f"[ILP-{run_idx}] 需要覆盖 {num_j_subsets} 个 j-子集，每个至少 {y} 次。")

        if num_j_subsets == 0:
            print(f"[ILP-{run_idx}] 没有需要覆盖的 j-子集，直接完成。")
            result['status'] = 'OPTIMAL' # 无约束问题，目标为0，解为空集
            result['time'] = time.time() - start_time
            q.put(result)
            return

        # 3. 创建 CP-SAT 模型
        model = cp_model.CpModel()

        # 4. 定义变量: 每个 k-子集是否被选中 (0 或 1)
        x = [model.NewBoolVar(f'x_{i}') for i in range(num_k_subsets)]

        # 5. 定义约束: 每个 j-子集必须被至少 y 个满足条件的 k-子集覆盖
        print(f"[ILP-{run_idx}] 开始添加约束 (每个 j-子集 >= {y} 次覆盖)...")
        constraints_added = 0
        feasible = True # 标记模型是否可能可行
        for idx_j, j_subset in enumerate(target_j_subsets):
            # 找到所有与当前 j_subset 交集 >= s 的 k_subsets 的索引
            covering_k_indices = [
                k_subset_indices[k_subset]
                for k_subset in all_k_subsets
                if len(j_subset.intersection(k_subset)) >= s
            ]

            # 检查是否有足够的 k-子集来满足 y 次覆盖
            if len(covering_k_indices) < y:
                print(f"[ILP-{run_idx}] 错误：j-子集 {idx_j} ({set(j_subset)}) 最多只能被 {len(covering_k_indices)} 个 k-子集覆盖，无法满足 y={y} 的要求。问题不可行。")
                result['status'] = 'INFEASIBLE' # 直接标记为不可行
                feasible = False
                break # 不再添加后续约束，因为已经确定不可行

            # 添加约束：这些候选 k-子集被选中的数量必须 >= y
            model.Add(sum(x[i] for i in covering_k_indices) >= y) # <--- 核心修改：>= y
            constraints_added += 1

        if not feasible: # 如果在添加约束时检测到不可行
             result['time'] = time.time() - start_time
             q.put(result)
             return

        print(f"[ILP-{run_idx}] 为 {constraints_added}/{num_j_subsets} 个 j-子集添加了覆盖约束。")

        # 6. 定义目标函数: 最小化选中的 k-子集数量
        model.Minimize(sum(x))

        # 7. 创建求解器并设置参数
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(timeout_solver)
        # solver.parameters.log_search_progress = True # 可选：显示求解日志
        solver.parameters.num_search_workers = max(1, mp.cpu_count() // 2) # 使用多核
        print(f"[ILP-{run_idx}] 使用 {solver.parameters.num_search_workers} 个 worker 进行求解...")

        # 8. 求解模型
        status = solver.Solve(model)
        result['time'] = time.time() - start_time # 记录总时间

        # 9. 处理结果
        status_map = {
            cp_model.OPTIMAL: 'OPTIMAL',        # 找到最优解
            cp_model.FEASIBLE: 'FEASIBLE',      # 找到可行解（但可能未证明最优，常见于超时）
            cp_model.INFEASIBLE: 'INFEASIBLE',    # 证明无解
            cp_model.MODEL_INVALID: 'MODEL_INVALID',# 模型本身有问题
            cp_model.UNKNOWN: 'UNKNOWN'         # 未知状态（通常因为超时）
        }
        result['status'] = status_map.get(status, f'UNMAPPED_STATUS_{status}')
        obj_value = solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else float('inf')
        print(f"[ILP-{run_idx}] 求解完成。状态: {result['status']}, 目标值(集合数量): {obj_value if obj_value != float('inf') else 'N/A'}")

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            selected_indices = [i for i, var in enumerate(x) if solver.Value(var) == 1]
            chosen_sets = [all_k_subsets[i] for i in selected_indices]
            result['sets'] = [list(sorted(list(s))) for s in chosen_sets]
            print(f"[ILP-{run_idx}] 找到 {len(result['sets'])} 个集合。")
        elif status == cp_model.UNKNOWN:
             print(f"[ILP-{run_idx}] 求解器在 {timeout_solver} 秒内未能找到最优解或证明不可行。可能是超时。")
        elif status == cp_model.INFEASIBLE:
             print(f"[ILP-{run_idx}] 模型被证明不可行，不存在满足 y={y} 覆盖条件的解。")
        else:
             print(f"[ILP-{run_idx}] 求解器返回未处理状态：{result['status']}")

    except Exception as e:
        print(f"[ILP-{run_idx}] CP-SAT 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        result['status'] = 'ERROR'
        result['error_message'] = str(e)

    finally:
        result['time'] = time.time() - start_time # 确保记录最终时间
        try:
            q.put(result)
            print(f"[ILP-{run_idx}] 结果已放入队列。耗时: {result['time']:.2f} 秒。")
        except Exception as qe:
            print(f"[ILP-{run_idx}] 错误：无法将结果放入队列: {qe}")

# --- Sample 类：组织计算任务 ---
# （Sample 类的 __init__ 和 run 方法不需要修改，因为它们已经正确传递 y 参数）
class Sample:
    """
    代表一次计算实例，负责管理参数、运行算法、收集结果。
    """
    def __init__(self, m, n, k, j, s, y, run_idx, timeout=60, rand_instance=None):
        """
        初始化 Sample 实例。

        参数:
            m (int): 基础集合的大小 {1, ..., m}
            n (int): 选取的子集 (universe) 的大小
            k (int): 每个覆盖块 (Set) 的大小
            j (int): 需要被覆盖的子集的大小
            s (int): j-子集中必须与覆盖块 (Set) 共享的元素数量的最小值 (>=s)
            y (int): **覆盖次数要求** (每个 j-子集要被覆盖 >= y 次)
            run_idx (int): 运行唯一标识符
            timeout (int): 整个 run 方法的总超时时间 (秒)
            rand_instance (random.Random, 可选): 用于生成宇宙的随机数生成器实例
        """
        self.m = m
        self.n = n
        self.k = k
        self.j = j
        self.s = s
        self.y = y # 存储实际使用的覆盖次数要求
        self.run_idx = run_idx
        self.timeout = timeout
        self.rand = rand_instance if rand_instance else random

        # --- 结果属性初始化 ---
        self.univ = []        # 将使用的 n 个元素的宇宙 (可能被手动输入覆盖)
        self.q = mp.Queue()   # 用于接收算法结果的队列
        self.result = {}      # 存储选定算法的结果字典
        self.sets = []        # 存储选定算法找到的覆盖集列表
        self.ans = None       # 最终格式化的结果字符串

        print(f"[Sample-{self.run_idx}] 初始化实例: M={m}, N={n}, K={k}, J={j}, S={s}, Target_Y={y}, Timeout={timeout}s")

        # 1. 生成宇宙 (Universe) - 如果 main.py 使用随机模式，这里会生成
        #   如果 main.py 使用手动模式，它会在调用 run 之前覆盖 self.univ
        if n > m:
            raise ValueError(f"错误：N ({n}) 不能大于 M ({m})")
        # 只有在需要时才生成（例如，如果主调没有提供）？
        # 或者总是生成一个默认的，然后可能被主调覆盖？
        # 当前：总是生成，然后主调可以覆盖。
        if n > 0 and m >= n : # 确保 n, m 有效
             self.univ = sorted(self.rand.sample(range(1, m + 1), n))
             print(f"[Sample-{self.run_idx}] 内部生成了初始 Universe (大小 {n}): {self.univ} (可能被外部覆盖)")
        else:
             print(f"[Sample-{self.run_idx}] M或N无效，未生成 Universe。")
             if n <= 0 or m <= 0:
                 raise ValueError("M 和 N 必须为正数。")

        # 注意：这里不再预计算 self.blocks，因为算法内部会生成 k-子集

    def run(self):
        """
        启动 Greedy 和 CP-SAT (如果可用) 算法并行计算，并根据结果选择最佳方案。
        算法现在应该使用 self.y 作为覆盖次数目标。
        """
        print(f"[RUN-{self.run_idx}] 开始运行并行计算 (目标 y={self.y})...")

        # 检查 Universe 是否有效
        if not self.univ or len(self.univ) != self.n:
             print(f"[RUN-{self.run_idx}] 错误：Universe 无效或大小与 N 不符。Univ: {self.univ}, N: {self.n}")
             self.ans = f"Fail: Invalid Universe"
             self.result = {'status': 'Error', 'alg': 'Preprocessing', 'sets': [], 'time': 0, 'error': 'Invalid Universe'}
             self.sets = []
             return

        processes = []

        # -- 准备参数元组 --
        # y 值现在是关键参数，传递给算法
        common_args = (self.q, self.univ, self.k, self.j, self.s, self.y, self.run_idx)

        # -- 启动 Greedy 进程 --
        args_greedy = common_args
        p_greedy = mp.Process(target=greedy_cover, args=(args_greedy,), daemon=True)
        processes.append(p_greedy)
        p_greedy.start()
        print(f"[RUN-{self.run_idx}] 启动了 Greedy 进程 (PID: {p_greedy.pid}, 目标 y={self.y})")

        # -- 启动 CP-SAT 进程 (如果可用) --
        p_ilp = None
        if HAS_ORTOOLS:
            timeout_solver = max(1.0, self.timeout * 0.95)
            args_ilp = common_args + (timeout_solver,) # 添加 solver 超时
            p_ilp = mp.Process(target=cpsat_cover, args=(args_ilp,), daemon=True)
            processes.append(p_ilp)
            p_ilp.start()
            print(f"[RUN-{self.run_idx}] 启动了 CP-SAT (ILP) 进程 (PID: {p_ilp.pid}, 目标 y={self.y}), 内部超时 {timeout_solver:.1f}s")
        else:
            print(f"[RUN-{self.run_idx}] CP-SAT (ILP) 求解器不可用，仅运行 Greedy。")

        # -- 等待并处理结果 (选择逻辑保持不变，优先 ILP 的 OPTIMAL/FEASIBLE) --
        res1 = None
        res2 = None
        time_start_wait = time.time()
        selected_result = None

        try:
            print(f"[RUN-{self.run_idx}] 等待第一个结果到达 (总超时={self.timeout:.1f}s)...")
            res1 = self.q.get(timeout=self.timeout)
            time_elapsed = time.time() - time_start_wait
            res1_alg = res1.get('alg', 'N/A')
            res1_status = res1.get('status', 'N/A')
            print(f"[RUN-{self.run_idx}] 在 {time_elapsed:.2f}s 收到第一个结果，来自 {res1_alg}，状态: {res1_status}")

            timeout_rem = self.timeout - time_elapsed
            if timeout_rem > 0.1 and len(processes) > 1 :
                try:
                    print(f"[RUN-{self.run_idx}] 尝试在剩余 {timeout_rem:.2f}s 内获取第二个结果...")
                    res2 = self.q.get(timeout=timeout_rem)
                    time_elapsed_total = time.time() - time_start_wait
                    res2_alg = res2.get('alg', 'N/A')
                    res2_status = res2.get('status', 'N/A')
                    print(f"[RUN-{self.run_idx}] 在总计 {time_elapsed_total:.2f}s 收到第二个结果，来自 {res2_alg}，状态: {res2_status}")
                except queue.Empty:
                    print(f"[RUN-{self.run_idx}] 在剩余时间内未收到第二个结果。")
            elif len(processes) <= 1:
                print(f"[RUN-{self.run_idx}] 只有一个算法运行，无需等待第二个结果。")
            else:
                 print(f"[RUN-{self.run_idx}] 剩余时间不足 ({timeout_rem:.2f}s)，不再等待第二个结果。")

            # --- 核心决策逻辑：选择最优结果 ---
            print(f"[RUN-{self.run_idx}] --- 分析结果 ---")
            results_map = {}
            if res1 and 'alg' in res1: results_map[res1['alg']] = res1
            if res2 and 'alg' in res2: results_map[res2['alg']] = res2

            print(f"[RUN-{self.run_idx}] 可用结果: { {alg: data.get('status', 'N/A') for alg, data in results_map.items()} }")

            ilp_res = results_map.get('ILP')
            greedy_res = results_map.get('Greedy')

            # 优先级 1: ILP 得到 OPTIMAL 或 FEASIBLE 解
            if ilp_res and ilp_res.get('status') in ('OPTIMAL', 'FEASIBLE'):
                if 'sets' in ilp_res and ilp_res.get('sets') is not None:
                    print(f"[RUN-{self.run_idx}] **选择 ILP 结果** (状态: {ilp_res.get('status')}, 集合数: {len(ilp_res['sets'])}).")
                    selected_result = ilp_res
                else:
                    print(f"[RUN-{self.run_idx}] 警告：ILP 报告 {ilp_res.get('status')} 但缺少 'sets'。尝试回退。")

            # 优先级 2: 如果 ILP 未选中，且 Greedy 成功 ('SUCCESS')
            # **重要**: 确保 greedy_cover 成功时 status 是 'SUCCESS'
            if selected_result is None and greedy_res and greedy_res.get('status') == 'SUCCESS':
                if 'sets' in greedy_res and greedy_res.get('sets') is not None:
                    print(f"[RUN-{self.run_idx}] **选择 Greedy 结果** (状态: {greedy_res.get('status')}, 集合数: {len(greedy_res['sets'])}). (ILP 不适用或未成功)")
                    selected_result = greedy_res
                else:
                     print(f"[RUN-{self.run_idx}] 警告：Greedy 报告 SUCCESS 但缺少 'sets'。尝试回退。")

            # 优先级 3: 回退逻辑
            if selected_result is None:
                print(f"[RUN-{self.run_idx}] 未能基于最优/成功状态选择结果，进入回退...")
                # 优先选择任何 ILP 结果 (即使是 INFEASIBLE 或 UNKNOWN)
                if ilp_res:
                    print(f"[RUN-{self.run_idx}] **选择 ILP 结果作为回退** (状态: {ilp_res.get('status', 'N/A')})")
                    selected_result = ilp_res
                elif greedy_res: # 如果只有 Greedy
                    print(f"[RUN-{self.run_idx}] **选择 Greedy 结果作为回退** (状态: {greedy_res.get('status', 'N/A')})")
                    selected_result = greedy_res
                else:
                    print(f"[RUN-{self.run_idx}] 错误：无法从队列中获取任何有效结果。")
                    self.ans = f"Fail: No valid result obtained."
                    self.result = {'status': 'NoResult', 'alg': 'None', 'sets': [], 'time': time.time() - time_start_wait}
                    self.sets = []

        except queue.Empty:
            total_wait_time = time.time() - time_start_wait
            print(f"[RUN-{self.run_idx}] 错误：在 {total_wait_time:.2f} 秒内 (超时 {self.timeout:.1f}s) 未收到任何算法的结果。")
            self.ans = f"Fail: Timeout ({self.timeout:.1f}s)"
            self.result = {'status': 'Timeout', 'alg': 'None', 'sets': [], 'time': total_wait_time}
            self.sets = []

        except Exception as e:
            total_time = time.time() - time_start_wait
            print(f"[RUN-{self.run_idx}] 处理结果时发生意外错误: {e}")
            import traceback
            traceback.print_exc()
            self.ans = f"Error: Exception in result processing"
            self.result = {'status': 'RuntimeError', 'alg': 'Error', 'sets': [], 'time': total_time, 'error': str(e)}
            self.sets = []

        finally:
            # 清理队列
            while not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: break
            print(f"[RUN-{self.run_idx}] Run 方法结果处理和清理完成。")

        # --- 设置最终结果属性 ---
        if selected_result:
            self.result = selected_result
            self.sets = self.result.get('sets', [])
            num_results = len(self.sets) if isinstance(self.sets, list) else 0

            # 构建标准答案字符串: m-n-k-j-s-run_idx-num_sets
            self.ans = f"{self.m}-{self.n}-{self.k}-{self.j}-{self.s}-{self.run_idx}-{num_results}"

            final_alg = self.result.get('alg', 'N/A')
            final_status = self.result.get('status', 'N/A')
            final_time = self.result.get('time', 0)
            cov_target = self.result.get('coverage_target', self.y) # 获取算法实际使用的 y
            print(f"[RUN-{self.run_idx}] ---- Final Result Summary ----")
            print(f"[RUN-{self.run_idx}] Selected Algorithm: {final_alg}")
            print(f"[RUN-{self.run_idx}] Status: {final_status}")
            print(f"[RUN-{self.run_idx}] Coverage Target (y): {cov_target}")
            print(f"[RUN-{self.run_idx}] Time (internal): {final_time:.2f}s")
            print(f"[RUN-{self.run_idx}] Sets Found: {num_results}")
            print(f"[RUN-{self.run_idx}] Result ID (ans): {self.ans}")
            print(f"[RUN-{self.run_idx}] -----------------------------")

        elif self.ans is None: # 如果未在异常中设置 ans
             fail_status = self.result.get('status', 'UnknownFailure') if hasattr(self, 'result') and self.result else 'SetupFailure'
             self.ans = f"Fail({fail_status}):{self.m}-{self.n}-{self.k}-{self.j}-{self.s}-{self.run_idx}-0"
             if not hasattr(self, 'result'): self.result = {'status': fail_status, 'alg': 'None', 'sets': [], 'time': 0}
             if not hasattr(self, 'sets'): self.sets = []
             print(f"[RUN-{self.run_idx}] 最终未能选择结果。Ans 设置为: {self.ans}")

        return

# --- 主程序入口 (用于直接测试 backend.py) ---
if __name__ == '__main__':
    print("backend.py 被直接运行。进行 y 次覆盖测试...")

    # 设置测试参数
    test_m = 10
    test_n = 7
    test_k = 4
    test_j = 3
    test_s = 2
    test_y = 2 # 要求每个 j-子集被覆盖至少 2 次
    test_run_idx = 1
    test_timeout = 45 # 秒

    print(f"\n测试参数: M={test_m}, N={test_n}, K={test_k}, J={test_j}, S={test_s}, Y={test_y}, Timeout={test_timeout}s")

    try:
        test_random_instance = random.Random(0) # 固定种子
        sample_instance = Sample(test_m, test_n, test_k, test_j, test_s, test_y,
                                 test_run_idx, test_timeout, test_random_instance)

        # 手动设置一个 Universe (如果需要)
        # sample_instance.univ = [1, 2, 3, 4, 5, 6, 7]
        # print(f"手动设置 Universe: {sample_instance.univ}")

        sample_instance.run()

        print("\n--- 测试运行结果 ---")
        print(f"最终结果标识 (ans): {sample_instance.ans}")
        print(f"选择的算法结果 (result): {sample_instance.result}")
        print(f"找到的集合 (sets):")
        if sample_instance.sets:
            for i, found_set in enumerate(sample_instance.sets):
                print(f"  Set {i+1}: {found_set}")
        else:
            print("  未能找到任何集合。")

    except ValueError as ve:
         print(f"参数错误： {ve}")
    except Exception as e:
        print(f"测试过程中发生未预料的错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n后端测试结束。")