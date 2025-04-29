# backend.py
# 包含核心计算逻辑：贪心算法和CP-SAT（ILP）求解器
# ** 版本已修改，以支持“每个 j-子集至少被覆盖 c 次”的需求 (c 代替了 y)。**
# ** 移除了 RunCounter 类，因为 run_index 的管理移到了数据库 db.py **

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

# --- RunCounter 类已被移除 ---
# --- 贪心算法 (已修改以支持 c 次覆盖) ---

def greedy_cover(args):
    """
    贪心算法求解覆盖问题，要求每个 j-子集被覆盖至少 c 次。
    参数: (q, univ, k, j, s, c, run_idx)
         q: 结果队列
         univ: 元素全集 (n 个元素)
         k: 覆盖块大小
         j: 被覆盖子集大小
         s: 交集大小阈值 (j-子集与k-块交集 >= s)
         c: **必须覆盖的次数** (每个 j-子集需要被满足条件的 k-块覆盖至少 c 次)
         run_idx: 运行索引 (由调用者提供，通过数据库管理)
    """
    q, univ, k, j, s, c, run_idx = args # 解包参数 (c 代替 y)
    n = len(univ)
    start_time = time.time()
    # !!! 修改: coverage_target 使用 c
    result = {'alg': 'Greedy', 'status': 'INIT', 'sets': [], 'time': 0, 'j_subsets_total': 0, 'j_subsets_covered': 0, 'coverage_target': c, 'run_index': run_idx}

    print(f"[Greedy-{run_idx}] 开始执行贪心算法 (覆盖目标 c={c})...") 

    # 基本参数检查
    if c <= 0: 
        print(f"[Greedy-{run_idx}] 错误：覆盖次数 c ({c}) 必须大于 0。") 
        result['status'] = 'ERROR_INVALID_C' 
        result['error_message'] = f"覆盖次数 c ({c}) 必须大于 0。" 
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
        print(f"[Greedy-{run_idx}] 需要覆盖 {num_j_subsets} 个 j-子集，每个至少 {c} 次。") 

        if num_j_subsets == 0:
             print(f"[Greedy-{run_idx}] 没有需要覆盖的 j-子集，直接完成。")
             result['status'] = 'SUCCESS' # 没有目标，视为成功
             result['time'] = time.time() - start_time
             q.put(result)
             return

        # 3. Precompute which j-subsets each k-subset can cover (based on intersection size >= s).
        k_subset_covers_j_indices = defaultdict(list)
        j_subset_covered_by_k_indices = defaultdict(list)
        for idx_j, j_subset in enumerate(target_j_subsets):
            for idx_k, k_subset in enumerate(all_k_subsets):
                if len(j_subset.intersection(k_subset)) >= s:
                    k_subset_covers_j_indices[idx_k].append(idx_j)
                    j_subset_covered_by_k_indices[idx_j].append(idx_k)
        print(f"[Greedy-{run_idx}] Precomputation of covering relationships completed.")

        # Check if there exists a j-subset that cannot be covered c times at all.
        for idx_j in range(num_j_subsets):
             potential_covers = len(j_subset_covered_by_k_indices[idx_j])
             if potential_covers < c: 
                 print(f"[Greedy-{run_idx}] Error: j-subset {idx_j} ({target_j_subsets[idx_j]}) can be covered by at most {potential_covers} k-subsets, which does not meet the requirement of c={c}. Problem is infeasible.")
                 result['status'] = 'INFEASIBLE_C_TARGET' 
                 result['error_message'] = f"j-subset {idx_j} cannot be covered {c} times." 
                 result['time'] = time.time() - start_time
                 q.put(result)
                 return

        # 4. 贪心选择过程 (面向 c 次覆盖)
        selected_k_subset_indices = [] # 存储选中的 k-子集的索引
        # 跟踪每个 j-子集的当前覆盖次数 / j_subset_coverage_count是里面有 num_j_subsets 个0的列表 [0, 0, 0, ... 0, 0]
        j_subset_coverage_count = [0] * num_j_subsets
        # 跟踪哪些 j-子集还需要更多覆盖 (覆盖次数 < c)
        
        needs_more_coverage_j_indices = set(range(num_j_subsets))
        print(f"[Greedy-{run_idx}] 开始贪心选择，目标 c={c}...") 

        iteration = 0
        # Loop until the coverage count of all j-subsets reaches c.
        while needs_more_coverage_j_indices:
            iteration += 1
            best_k_subset_idx = -1
            max_coverage_increase_count = -1 # How many additional covers would selecting this k-block provide for *unsatisfied* j-subsets?
            for idx_k in range(len(all_k_subsets)):
                # Calculate which *still-to-be-covered* j-subsets the current k-subset can provide coverage for.
                relevant_j_indices = set(k_subset_covers_j_indices[idx_k]).intersection(needs_more_coverage_j_indices)
                current_coverage_increase_count = len(relevant_j_indices)
                # Greedy strategy: Select the k-subset that provides the most coverage for the currently unsatisfied j-subsets.
                if current_coverage_increase_count > max_coverage_increase_count:
                    max_coverage_increase_count = current_coverage_increase_count
                    best_k_subset_idx = idx_k
            if best_k_subset_idx == -1:
                # 检查是否真的没有任何 K-子集可以覆盖任何剩余的 J-子集
                can_cover_anything = False
                for idx_k_check in range(len(all_k_subsets)):
                    if set(k_subset_covers_j_indices[idx_k_check]).intersection(needs_more_coverage_j_indices):
                        can_cover_anything = True
                        break
                if not can_cover_anything and needs_more_coverage_j_indices:
                    # 如果确实没有任何块能覆盖剩下的了
                    
                    print(f"[Greedy-{run_idx}] 错误：在迭代 {iteration} 中，没有任何剩余的 k-块可以覆盖任何仍未满足 c={c} 覆盖的 j-子集。问题可能不可行或贪心策略无法找到解。")
                    result['status'] = 'FAILED_INFEASIBLE_REMAINING'
                    result['error_message'] = "Greedy search cannot find any k-subset to cover remaining j-subsets."
                else:
                    # 保持原来的警告，以防是其他逻辑问题
                    print(f"[Greedy-{run_idx}] 警告：在迭代 {iteration} 中找不到最佳 k-子集 (best_k_subset_idx = -1)。可能存在问题。")
                    result['status'] = 'FAILED_INCOMPLETE_COVER' # 或其他错误状态
                break # 退出循环
            # Pick the best k-subset found in this iteration. 
            selected_k_subset_indices.append(best_k_subset_idx)
            chosen_k_subset = all_k_subsets[best_k_subset_idx]
            print(f"[Greedy-{run_idx}] Iteration {iteration}: Selected block {best_k_subset_idx} {list(sorted(chosen_k_subset))}, "
                  f"increased coverage for {max_coverage_increase_count} unsatisfied j-subsets.")
            # Update the coverage count of the affected j-subsets
            newly_satisfied_count = 0
            j_indices_affected_this_round = k_subset_covers_j_indices[best_k_subset_idx]
            for idx_j in j_indices_affected_this_round:
                # Only update those that still need to be overwritten.
                if idx_j in needs_more_coverage_j_indices:
                    j_subset_coverage_count[idx_j] += 1
                    # Check if c covers have just been satisfied.
                    if j_subset_coverage_count[idx_j] >= c: 
                        needs_more_coverage_j_indices.remove(idx_j) # Remove from the set to be covered
                        newly_satisfied_count += 1
            print(f"[Greedy-{run_idx}]   -> After this round of selection, {newly_satisfied_count} j-subsets have reached the c={c} coverage target.") 
            print(f"[Greedy-{run_idx}]   -> {len(needs_more_coverage_j_indices)} j-subsets remain to be satisfied.")
            # Add an iteration limit to prevent infinite loops
            max_iterations = len(all_k_subsets) * c 
            if iteration > max_iterations and max_iterations > 0:
                  print(f"[Greedy-{run_idx}] 警告：迭代次数过多 ({iteration} > {max_iterations})，可能陷入循环或收敛缓慢。提前终止。")
                  result['status'] = 'FAILED_ITERATION_LIMIT'
                  break
        # 5. Results 
        chosen_sets_list = [list(sorted(list(all_k_subsets[idx]))) for idx in selected_k_subset_indices]
        result['sets'] = chosen_sets_list
        result['j_subsets_covered'] = num_j_subsets - len(needs_more_coverage_j_indices) # The number of satisfied j-subsets

        if not needs_more_coverage_j_indices: # If all j-subsets are satisfied by c times 
            result['status'] = 'SUCCESS'
            print(f"[Greedy-{run_idx}] Greedy algorithm successfully completed c={c} coverage, selecting a total of {len(chosen_sets_list)} sets.")
        else:
            # 如果循环退出但仍有未满足的
            if result['status'] == 'INIT': # 如果状态未被错误或迭代限制覆盖
                result['status'] = 'FAILED_INCOMPLETE_COVER' # 默认失败状态
            
            print(f"[Greedy-{run_idx}] 贪心算法结束，状态: {result['status']}。{result['j_subsets_covered']}/{num_j_subsets} 个 j-子集满足了 c={c} 覆盖。选择了 {len(chosen_sets_list)} 个集合。")

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

# --- CP-SAT (ILP) 求解器 (已修改以支持 c 次覆盖) ---

def cpsat_cover(args):
    """
    使用 OR-Tools CP-SAT 求解器解决覆盖问题，要求每个 j-子集被覆盖至少 c 次。
    参数: (q, univ, k, j, s, c, run_idx, timeout_solver)
         q: 结果队列
         univ: 元素全集
         k: 覆盖块大小
         j: 被覆盖子集大小
         s: 交集大小阈值 (>= s)
         c: **必须覆盖的次数**
         run_idx: 运行索引 (由调用者提供，通过数据库管理)
         timeout_solver: CP-SAT 求解器的内部超时时间（秒）
    """
    if not HAS_ORTOOLS:
        
        q.put({'alg': 'ILP', 'status': 'ERROR_MISSING_ORTOOLS', 'sets': [], 'time': 0, 'coverage_target': args[5], 'run_index': args[6]}) # 包含 run_idx
        return

    
    q, univ, k, j, s, c, run_idx, timeout_solver = args # 解包参数
    n = len(univ)
    start_time = time.time()
    # !!! 修改: coverage_target 使用 c
    result = {'alg': 'ILP', 'status': 'INIT', 'sets': [], 'time': 0, 'coverage_target': c, 'run_index': run_idx}

    print(f"[ILP-{run_idx}] 开始执行 CP-SAT 求解器 (覆盖目标 c={c})... Timeout={timeout_solver}s") 

    # 基本参数检查
    if c <= 0: 
        print(f"[ILP-{run_idx}] 错误：覆盖次数 c ({c}) 必须大于 0。") 
        result['status'] = 'ERROR_INVALID_C' 
        result['error_message'] = f"覆盖次数 c ({c}) 必须大于 0。" 
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
        print(f"[ILP-{run_idx}] 需要覆盖 {num_j_subsets} 个 j-子集，每个至少 {c} 次。") 

        if num_j_subsets == 0:
            print(f"[ILP-{run_idx}] 没有需要覆盖的 j-子集，直接完成。")
            result['status'] = 'OPTIMAL' # 无约束问题，目标为0，解为空集
            result['time'] = time.time() - start_time
            q.put(result)
            return

        # 3. Create the CP-SAT model
        model = cp_model.CpModel()

        # 4. Define variables: Whether each k-subset is selected (0 or 1)
        x = [model.NewBoolVar(f'x_{i}') for i in range(num_k_subsets)]

        # 5. Define constraints: Each j-subset must be covered by at least c satisfying k-subsets
        print(f"[ILP-{run_idx}] Starting to add constraints (each j-subset >= {c} times coverage)...")
        constraints_added = 0
        feasible = True # Mark whether the model is potentially feasible
        j_subset_potential_covers = defaultdict(list) # Precompute which k-blocks can cover each j-block
        for idx_k, k_subset in enumerate(all_k_subsets):
             for idx_j, j_subset in enumerate(target_j_subsets):
                 if len(j_subset.intersection(k_subset)) >= s:
                     j_subset_potential_covers[idx_j].append(idx_k)

        for idx_j, j_subset in enumerate(target_j_subsets):
            # Find the indices of all k_subsets whose intersection with the current j_subset is >= s
            covering_k_indices = j_subset_potential_covers[idx_j]

            # Check if there are enough k-subsets to satisfy c coverage
            if len(covering_k_indices) < c: 
                print(f"[ILP-{run_idx}] 错误：j-子集 {idx_j} ({set(j_subset)}) 最多只能被 {len(covering_k_indices)} 个 k-子集覆盖，无法满足 c={c} 的要求。问题不可行。") 
                result['status'] = 'INFEASIBLE' # 直接标记为不可行
                feasible = False
                break # 不再添加后续约束，因为已经确定不可行

            # Add constraint: The number of these candidate k-subsets selected must be >= c
            if covering_k_indices: # Only add constraints when there are potential covering blocks
                model.Add(sum(x[i] for i in covering_k_indices) >= c) 
                constraints_added += 1
            else: # If a j-subset has no k-subset that can cover it (and c>=1), then it is infeasible
                if c >= 1: 
                    print(f"[ILP-{run_idx}] Error: j-subset {idx_j} ({set(j_subset)}) has no k-subset that satisfies intersection >= {s}. Problem is infeasible.")
                    result['status'] = 'INFEASIBLE'
                    feasible = False
                    break

        if not feasible: 
             result['time'] = time.time() - start_time
             q.put(result)
             return

        print(f"[ILP-{run_idx}] Added coverage constraints for {constraints_added}/{num_j_subsets} j-subsets.")

        # 6. Define the objective function: Minimize the number of selected k-subsets
        model.Minimize(sum(x))

        # 7. Create the solver and set parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(timeout_solver)
        # solver.parameters.log_search_progress = True // its a lot of output 
        # Try using more workers if the machine core count allows
        num_workers = mp.cpu_count()
        if num_workers > 1:
            solver.parameters.num_search_workers = num_workers
            print(f"[ILP-{run_idx}] Solving with {solver.parameters.num_search_workers} workers...")
        else:
            print(f"[ILP-{run_idx}] Solving with default number of workers...")

        # 8. Solve the model
        status = solver.Solve(model)
        result['time'] = time.time() - start_time # Total time taken for solving 

        # 9. Results 
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
             print(f"[ILP-{run_idx}] 求解器在 {timeout_solver} 秒内未能找到最优解或证明不可行。可能是超时或问题复杂。")
        elif status == cp_model.INFEASIBLE:
             print(f"[ILP-{run_idx}] 模型被证明不可行，不存在满足 c={c} 覆盖条件的解。") 
        else:
             print(f"[ILP-{run_idx}] 求解器返回未处理状态：{result['status']}")

    except FileNotFoundError as fnf_err: # 特别处理 ortools 可能的依赖文件错误
        print(f"[ILP-{run_idx}] CP-SAT 文件错误: {fnf_err}。确认 OR-Tools 安装完整。")
        result['status'] = 'ERROR_ORTOOLS_FILE'
        result['error_message'] = str(fnf_err)
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
class Sample:
    """
    代表一次计算实例，负责管理参数、运行算法、收集结果。
    """
    def __init__(self, m, n, k, j, s, c, run_idx, timeout=60, rand_instance=None):
        """
        初始化 Sample 实例。

        参数:
            m (int): 基础集合的大小 {1, ..., m}
            n (int): 选取的子集 (universe) 的大小
            k (int): 每个覆盖块 (Set) 的大小
            j (int): 需要被覆盖的子集的大小
            s (int): j-子集中必须与覆盖块 (Set) 共享的元素数量的最小值 (>=s)
            c (int): **覆盖次数要求** (每个 j-子集要被覆盖 >= c 次)
            run_idx (int): 运行唯一标识符 (从数据库获取，跨会话持久)
            timeout (int): 整个 run 方法的总超时时间 (秒)
            rand_instance (random.Random, 可选): 用于生成宇宙的随机数生成器实例
        """
        self.m = m
        self.n = n
        self.k = k
        self.j = j
        self.s = s
        self.c = c # !!! 修改: 存储实际使用的覆盖次数要求 (c 代替 y)
        self.run_idx = run_idx # !! 这个 run_idx 是从数据库获取的持久化索引
        self.timeout = timeout
        self.rand = rand_instance if rand_instance else random

        # --- 结果属性初始化 ---
        self.univ = []        # 将使用的 n 个元素的宇宙 (可能被手动输入覆盖)
        self.q = mp.Queue()   # 用于接收算法结果的队列
        self.result = {}      # 存储选定算法的结果字典
        self.sets = []        # 存储选定算法找到的覆盖集列表
        self.ans = None       # 最终格式化的结果字符串

        # !!! 修改: Target_Y -> Target_C, {y} -> {c}
        print(f"[Sample-{self.run_idx}] 初始化实例: M={m}, N={n}, K={k}, J={j}, S={s}, Target_C={c}, RunIndex={run_idx}, Timeout={timeout}s") # 显示 run_idx

        # 1. 生成宇宙 (Universe)
        if n > m:
            raise ValueError(f"错误：N ({n}) 不能大于 M ({m})")
        if n > 0 and m >= n :
             # 只有在调用者没有提供 univ 时才生成随机的
             # 在 main.py 中，通常会在调用 run 之前设置 self.univ
             # 这里保留生成逻辑，以防被直接使用或测试
             # self.univ = sorted(self.rand.sample(range(1, m + 1), n))
             # print(f"[Sample-{self.run_idx}] 内部生成了初始 Universe (大小 {n}): {self.univ} (可能被外部覆盖)")
             pass # 通常由 main.py 在调用 run 前设置
        else:
             print(f"[Sample-{self.run_idx}] M或N无效，未生成 Universe。")
             if n <= 0 or m <= 0:
                 raise ValueError("M 和 N 必须为正数。")

    def run(self):
        """
        启动 Greedy 和 CP-SAT (如果可用) 算法并行计算，并根据结果选择最佳方案。
        算法现在应该使用 self.c 作为覆盖次数目标。
        """
        print(f"[RUN-{self.run_idx}] 开始运行并行计算 (目标 c={self.c})...") 

        # 检查 Universe 是否有效
        if not self.univ or len(self.univ) != self.n:
             # 尝试生成默认的，如果外部没提供
             if self.n > 0 and self.m >= self.n:
                 self.univ = sorted(self.rand.sample(range(1, self.m + 1), self.n))
                 print(f"[RUN-{self.run_idx}]警告：外部未提供有效 Universe，内部随机生成: {self.univ}")
             else:
                 print(f"[RUN-{self.run_idx}] 错误：Universe 无效或大小与 N 不符，且无法生成默认。Univ: {self.univ}, N: {self.n}")
                 self.ans = f"Fail: Invalid Universe"
                 self.result = {'status': 'Error', 'alg': 'Preprocessing', 'sets': [], 'time': 0, 'error': 'Invalid Universe', 'run_index': self.run_idx}
                 self.sets = []
                 return

        processes = []

        # -- 准备参数元组 --
        # c 值现在是关键参数，传递给算法 (self.c 代替 self.y)
        common_args = (self.q, self.univ, self.k, self.j, self.s, self.c, self.run_idx) # run_idx 现在是持久化的

        # -- 启动 Greedy 进程 --
        args_greedy = common_args
        p_greedy = mp.Process(target=greedy_cover, args=(args_greedy,), daemon=True)
        processes.append(p_greedy)
        
        p_greedy.start()
        print(f"[RUN-{self.run_idx}] 启动了 Greedy 进程 (PID: {p_greedy.pid}, 目标 c={self.c})")

        # -- 启动 CP-SAT 进程 (如果可用) --
        p_ilp = None
        if HAS_ORTOOLS:
            # Timeout for solver slightly less than overall timeout
            # Ensure timeout is at least a small positive value
            timeout_solver = max(1.0, self.timeout * 0.95 if self.timeout > 1 else 1.0 )

            args_ilp = common_args + (timeout_solver,) # 添加 solver 超时
            p_ilp = mp.Process(target=cpsat_cover, args=(args_ilp,), daemon=True)
            processes.append(p_ilp)
            p_ilp.start()
            
            print(f"[RUN-{self.run_idx}] 启动了 CP-SAT (ILP) 进程 (PID: {p_ilp.pid}, 目标 c={self.c}), 内部超时 {timeout_solver:.1f}s")
        else:
            print(f"[RUN-{self.run_idx}] CP-SAT (ILP) 求解器不可用，仅运行 Greedy。")

        # -- 等待并处理结果 (选择逻辑保持不变，优先 ILP 的 OPTIMAL/FEASIBLE) --
        res1 = None
        res2 = None
        time_start_wait = time.time()
        selected_result = None

        try:
            print(f"[RUN-{self.run_idx}] 等待第一个结果到达 (总超时={self.timeout:.1f}s)...")
            # 设置一个合理的最小超时，防止 timeout 过小导致 get() 阻塞太短
            get_timeout = max(0.1, self.timeout)
            res1 = self.q.get(timeout=get_timeout)
            time_elapsed = time.time() - time_start_wait
            res1_alg = res1.get('alg', 'N/A')
            res1_status = res1.get('status', 'N/A')
            print(f"[RUN-{self.run_idx}] 在 {time_elapsed:.2f}s 收到第一个结果，来自 {res1_alg}，状态: {res1_status}")

            timeout_rem = self.timeout - time_elapsed
            if timeout_rem > 0.1 and len(processes) > 1 : # 只有在有多个进程且还有时间时才等待第二个
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
            else: # timeout_rem <= 0.1
                 print(f"[RUN-{self.run_idx}] 剩余时间不足 ({timeout_rem:.2f}s)，不再等待第二个结果。")

            # --- 核心决策逻辑：选择最优结果 ---
            print(f"[RUN-{self.run_idx}] --- 分析结果 ---")
            results_map = {}
            if res1 and 'alg' in res1: results_map[res1['alg']] = res1
            if res2 and 'alg' in res2: results_map[res2['alg']] = res2

            # 将 run_idx 添加回结果字典（如果算法内部忘记添加的话）
            for alg in results_map:
                if 'run_index' not in results_map[alg]:
                    results_map[alg]['run_index'] = self.run_idx

            print(f"[RUN-{self.run_idx}] 可用结果: { {alg: data.get('status', 'N/A') for alg, data in results_map.items()} }")

            ilp_res = results_map.get('ILP')
            greedy_res = results_map.get('Greedy')

            # 优先级 1: ILP 得到 OPTIMAL 或 FEASIBLE 解
            if ilp_res and ilp_res.get('status') in ('OPTIMAL', 'FEASIBLE'):
                # 确保 'sets' 存在且不为 None
                if 'sets' in ilp_res and ilp_res.get('sets') is not None:
                    print(f"[RUN-{self.run_idx}] **选择 ILP 结果** (状态: {ilp_res.get('status')}, 集合数: {len(ilp_res['sets'])}).")
                    selected_result = ilp_res
                else:
                    print(f"[RUN-{self.run_idx}] 警告：ILP 报告 {ilp_res.get('status')} 但缺少有效 'sets'。尝试回退。")

            # 优先级 2: 如果 ILP 未选中，且 Greedy 成功 ('SUCCESS')
            if selected_result is None and greedy_res and greedy_res.get('status') == 'SUCCESS':
                 if 'sets' in greedy_res and greedy_res.get('sets') is not None:
                    print(f"[RUN-{self.run_idx}] **选择 Greedy 结果** (状态: {greedy_res.get('status')}, 集合数: {len(greedy_res['sets'])}). (ILP 不适用或未成功)")
                    selected_result = greedy_res
                 else:
                     print(f"[RUN-{self.run_idx}] 警告：Greedy 报告 SUCCESS 但缺少有效 'sets'。尝试回退。")

            # 优先级 3: 回退逻辑
            if selected_result is None:
                print(f"[RUN-{self.run_idx}] 未能基于最优/成功状态选择结果，进入回退...")
                # 优先选择任何 ILP 结果 (即使是 INFEASIBLE 或 UNKNOWN/Timeout)
                if ilp_res:
                    print(f"[RUN-{self.run_idx}] **选择 ILP 结果作为回退** (状态: {ilp_res.get('status', 'N/A')})")
                    selected_result = ilp_res
                # 其次选择 Greedy 结果（即使它不是 'SUCCESS'）
                elif greedy_res:
                    print(f"[RUN-{self.run_idx}] **选择 Greedy 结果作为回退** (状态: {greedy_res.get('status', 'N/A')})")
                    selected_result = greedy_res
                # 如果连一个结果都没有收到
                else:
                    print(f"[RUN-{self.run_idx}] 错误：无法从队列中获取任何有效结果。")
                    total_wait_time = time.time() - time_start_wait
                    self.ans = f"Fail: No valid result obtained."
                    self.result = {'status': 'NoResult', 'alg': 'None', 'sets': [], 'time': total_wait_time, 'run_index': self.run_idx}
                    self.sets = []

        except (queue.Empty, TimeoutError) as toe:
             total_wait_time = time.time() - time_start_wait
             print(f"[RUN-{self.run_idx}] 错误：在 {total_wait_time:.2f} 秒内未收到预期的结果 ({type(toe).__name__})。总超时为 {self.timeout:.1f}s。")
             self.ans = f"Fail: Timeout or Queue Empty ({self.timeout:.1f}s)"
             # 检查是否至少有一个结果，即使是超时的
             results_map = {}
             if res1 and 'alg' in res1: results_map[res1['alg']] = res1
             if res2 and 'alg' in res2: results_map[res2['alg']] = res2
             if results_map:
                 # 如果有部分结果，优先选择 ILP 的（通常 ILP更容易超时）
                 if 'ILP' in results_map:
                     selected_result = results_map['ILP']
                     if 'status' not in selected_result or selected_result.get('status') in ('INIT', 'UNKNOWN'):
                         selected_result['status'] = 'TIMEOUT_PARTIAL' # 标记为部分结果（可能超时）
                     print(f"[RUN-{self.run_idx}] **选择部分 ILP 结果** (状态: {selected_result.get('status')})")
                 else: # 否则选择 Greedy 的
                     selected_result = results_map['Greedy']
                     if 'status' not in selected_result or selected_result.get('status') in ('INIT'):
                         selected_result['status'] = 'TIMEOUT_PARTIAL'
                     print(f"[RUN-{self.run_idx}] **选择部分 Greedy 结果** (状态: {selected_result.get('status')})")
                 self.result = selected_result # 使用这个部分结果
             else: # 如果真的一个结果都没有
                 self.result = {'status': 'Timeout', 'alg': 'None', 'sets': [], 'time': total_wait_time, 'run_index': self.run_idx}
             self.sets = self.result.get('sets', []) if self.result else []

        except Exception as e:
            total_time = time.time() - time_start_wait
            print(f"[RUN-{self.run_idx}] 处理结果时发生意外错误: {e}")
            import traceback
            traceback.print_exc()
            self.ans = f"Error: Exception in result processing"
            # 尝试从部分结果中恢复信息
            part_res = res1 or res2
            err_alg = part_res.get('alg', 'Error') if part_res else 'Error'
            self.result = {'status': 'RuntimeError', 'alg': err_alg, 'sets': [], 'time': total_time, 'error': str(e), 'run_index': self.run_idx}
            self.sets = []

        finally:
            # 确保所有进程都已终止
            print(f"[RUN-{self.run_idx}] 尝试终止子进程...")
            for p in processes:
                try:
                    if p.is_alive():
                        print(f"  - 正在终止进程 PID {p.pid}...")
                        p.terminate() # 发送 SIGTERM
                        p.join(timeout=1.0) # 等待1秒
                        if p.is_alive(): # 如果还在运行
                            print(f"  - 进程 PID {p.pid} 未能正常终止，强制终止...")
                            p.kill() # 发送 SIGKILL
                            p.join(timeout=0.5) # 短暂等待
                        print(f"  - 进程 PID {p.pid} 已终止。")
                    else:
                        print(f"  - 进程 PID {p.pid} 已结束。")
                except Exception as term_err:
                    print(f"  - 终止进程 PID {p.pid} 时出错: {term_err}")
            # 清理队列
            while not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: break
            print(f"[RUN-{self.run_idx}] Run 方法结果处理和清理完成。")

        # --- 设置最终结果属性 ---
        if selected_result: # 检查 selected_result 是否被成功赋值
            self.result = selected_result
            self.sets = self.result.get('sets', [])
            num_results = len(self.sets) if isinstance(self.sets, list) else 0

            # 构建标准答案字符串: m-n-k-j-s-run_idx-num_sets
            self.ans = f"{self.m}-{self.n}-{self.k}-{self.j}-{self.s}-{self.run_idx}-{num_results}"

            final_alg = self.result.get('alg', 'N/A')
            final_status = self.result.get('status', 'N/A')
            final_time = self.result.get('time', 0)
            # !!! 修改: 默认值使用 self.c, Key 使用 'coverage_target'
            cov_target = self.result.get('coverage_target', self.c)
            final_run_idx = self.result.get('run_index', self.run_idx) # 确认 run_index

            print(f"[RUN-{final_run_idx}] ---- Final Result Summary ----")
            print(f"[RUN-{final_run_idx}] Selected Algorithm: {final_alg}")
            print(f"[RUN-{final_run_idx}] Status: {final_status}")
            
            print(f"[RUN-{final_run_idx}] Coverage Target (c): {cov_target}")
            print(f"[RUN-{final_run_idx}] Time (internal): {final_time:.2f}s")
            print(f"[RUN-{final_run_idx}] Sets Found: {num_results}")
            print(f"[RUN-{final_run_idx}] Result ID (ans): {self.ans}")
            print(f"[RUN-{final_run_idx}] -----------------------------")

        elif self.ans is None: # 如果未在异常中设置 ans，且 selected_result 为 None
             fail_status = self.result.get('status', 'UnknownFailure') if hasattr(self, 'result') and self.result else 'SetupFailure'
             num_results = 0 # 失败时集合数为 0
             self.ans = f"Fail({fail_status}):{self.m}-{self.n}-{self.k}-{self.j}-{self.s}-{self.run_idx}-{num_results}"
             if not hasattr(self, 'result') or not self.result: # 确保 self.result 存在
                 self.result = {'status': fail_status, 'alg': 'None', 'sets': [], 'time': 0, 'run_index': self.run_idx}
             if not hasattr(self, 'sets'): # 确保 self.sets 存在
                self.sets = []
             print(f"[RUN-{self.run_idx}] 最终未能选择有效结果。Ans 设置为: {self.ans}")

        # 在 run 方法结束前返回，确保 main.py 可以继续执行
        return

# --- 主程序入口 (用于直接测试 backend.py) ---
if __name__ == '__main__':
    
    print("backend.py 被直接运行。进行 c 次覆盖测试...")
    print("注意: 直接运行 backend.py 现在依赖 db.py 来获取 run_index。")
    print("将使用默认数据库文件 'k6_results.db' 来获取/更新索引。")

    # 导入 db 模块（如果尚未导入）
    try:
        import db as test_db
        test_db.setup_database() # 确保数据库和表存在 (列名 c_condition)
    except ImportError:
        print("错误：无法导入 db.py。请确保 db.py 在同一目录下或 Python 路径中。")
        exit()
    except Exception as db_setup_err:
        print(f"错误：设置数据库时出错: {db_setup_err}")
        exit()

    # 设置测试参数
    test_m = 10
    test_n = 7
    test_k = 4
    test_j = 3
    test_s = 2
    test_c = 2 # !!! 修改: test_y -> test_c
    test_timeout = 45 # 秒

    
    print(f"\n测试参数: M={test_m}, N={test_n}, K={test_k}, J={test_j}, S={test_s}, C={test_c}, Timeout={test_timeout}s")

    try:
        # 获取持久化的 run_index
        test_run_idx = test_db.get_and_increment_run_index(test_m, test_n, test_k, test_j, test_s)
        if test_run_idx is None:
             print("错误：无法从数据库获取运行索引。")
             exit()
        print(f"从数据库获取的本次运行索引: {test_run_idx}")

        test_random_instance = random.Random(0) # 固定种子
        # !!! 修改: test_y -> test_c in Sample instantiation
        sample_instance = Sample(test_m, test_n, test_k, test_j, test_s, test_c,
                                 test_run_idx, # 使用从数据库获取的索引
                                 test_timeout, test_random_instance)

        # 手动设置一个 Universe
        sample_instance.univ = sorted(test_random_instance.sample(range(1, test_m + 1), test_n))
        print(f"设置 Universe: {sample_instance.univ}")

        sample_instance.run()

        print("\n--- 测试运行结果 ---")
        print(f"最终结果标识 (ans): {sample_instance.ans}")
        # sample_instance.result['coverage_target'] 现在应该是 c 的值
        print(f"选择的算法结果 (result): {sample_instance.result}")
        print(f"找到的集合 (sets):")
        if sample_instance.sets:
            MAX_SETS_PRINT = 20
            for i, found_set in enumerate(sample_instance.sets[:MAX_SETS_PRINT]):
                print(f"  Set {i+1}: {found_set}")
            if len(sample_instance.sets) > MAX_SETS_PRINT:
                print(f"  ... (还有 {len(sample_instance.sets) - MAX_SETS_PRINT} 个集合未打印)")
        else:
            print("  未能找到任何集合。")

    except ValueError as ve:
         print(f"参数错误： {ve}")
    except Exception as e:
        print(f"测试过程中发生未预料的错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n后端测试结束。")