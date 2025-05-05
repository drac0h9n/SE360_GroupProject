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

            # Find the k-subset that covers the most currently *unsatisfied* j-subsets.
            # If multiple k-subsets cover the same maximum number, this simple loop picks the first one found.
            # More sophisticated tie-breaking (e.g., fewest total elements) could be added but isn't here.
            potential_candidates = []
            for idx_k in range(len(all_k_subsets)):
                # Optimization: If this k_subset was already selected, skip?
                # No, because a k_subset might be needed multiple times if c > 1,
                # although the current logic *doesn't re-select* the same block index.
                # This greedy approach selects a *new* block in each step.
                # If c > 1, it relies on *different* blocks contributing to the count.

                # Calculate which *still-to-be-covered* j-subsets the current k-subset can provide coverage for.
                relevant_j_indices = set(k_subset_covers_j_indices.get(idx_k, [])).intersection(needs_more_coverage_j_indices)
                current_coverage_increase_count = len(relevant_j_indices)

                # Greedy strategy: Select the k-subset that provides the most coverage
                # for the currently unsatisfied j-subsets.
                if current_coverage_increase_count > max_coverage_increase_count:
                    max_coverage_increase_count = current_coverage_increase_count
                    best_k_subset_idx = idx_k
                # Simple tie-breaking: if counts are equal, prefer lower index (implicitly done)

            if best_k_subset_idx == -1:
                # This means no available k-subset can cover any more *unsatisfied* j-subsets.
                # Check if this is expected (all covered) or an issue.
                if needs_more_coverage_j_indices:
                    # If there are still j-subsets needing coverage, but no k-subset helps,
                    # it implies either the problem is infeasible from the start (should have been caught earlier)
                    # or the greedy choice got stuck.
                    print(f"[Greedy-{run_idx}] 错误：在迭代 {iteration} 中，尽管仍有 {len(needs_more_coverage_j_indices)} 个 j-子集未满足 c={c} 覆盖，但找不到任何 k-块能增加它们的覆盖。问题可能不可行或贪心策略失败。")
                    result['status'] = 'FAILED_INCOMPLETE_COVER' # Indicate failure to cover all
                    result['error_message'] = "Greedy search could not find a k-subset to cover remaining j-subsets."
                else:
                     # This case shouldn't be reached if the while loop condition is correct.
                     # If needs_more_coverage_j_indices is empty, the loop should have terminated.
                     print(f"[Greedy-{run_idx}] 逻辑警告：在迭代 {iteration} 中 best_k_subset_idx 为 -1，但 needs_more_coverage_j_indices 已空。")
                break # Exit the loop

            # Pick the best k-subset found in this iteration.
            # Note: This basic greedy doesn't prevent selecting the same k-subset index multiple times
            # if it remains the best option in subsequent iterations. This is valid for c>1.
            # However, the *current implementation* uses list.append, implicitly building a list of *distinct steps*,
            # not necessarily distinct blocks if the same block is best multiple times.
            # For c>1, a better greedy might track remaining potential contribution per block.
            # Sticking to the current simple approach:
            selected_k_subset_indices.append(best_k_subset_idx)
            chosen_k_subset = all_k_subsets[best_k_subset_idx] # Get the set itself for logging
            # Logging the chosen block's index and content
            print(f"[Greedy-{run_idx}] Iteration {iteration}: Selected block index {best_k_subset_idx} {list(sorted(chosen_k_subset))}, "
                  f"potentially increasing coverage for {max_coverage_increase_count} unsatisfied j-subsets.")

            # Update the coverage count of the affected j-subsets
            newly_satisfied_count = 0
            # Find which j_indices this selected k_subset can cover (precomputed)
            j_indices_affected_this_round = k_subset_covers_j_indices.get(best_k_subset_idx, [])

            for idx_j in j_indices_affected_this_round:
                # Only update counts for j-subsets that *still need* more coverage
                if idx_j in needs_more_coverage_j_indices:
                    j_subset_coverage_count[idx_j] += 1
                    # Check if this j-subset has now reached the target coverage 'c'
                    if j_subset_coverage_count[idx_j] >= c:
                        needs_more_coverage_j_indices.remove(idx_j) # Remove from the set of unsatisfied j-subsets
                        newly_satisfied_count += 1

            print(f"[Greedy-{run_idx}]   -> After this round, {newly_satisfied_count} j-subsets newly reached the c={c} target.")
            print(f"[Greedy-{run_idx}]   -> {len(needs_more_coverage_j_indices)} j-subsets still need more coverage.")

            # Add an iteration limit to prevent potential infinite loops in edge cases
            # A reasonable upper bound might be num_j_subsets * c, but len(all_k_subsets) * c is safer if blocks are limited.
            max_iterations = len(all_k_subsets) * c if len(all_k_subsets) > 0 else num_j_subsets * c
            max_iterations = max(max_iterations, num_j_subsets) # Ensure at least num_j_subsets iterations if c=1
            if iteration > max_iterations and max_iterations > 0: # Check only if max_iterations is positive
                  print(f"[Greedy-{run_idx}] 警告：迭代次数过多 ({iteration} > {max_iterations})，可能陷入循环或收敛缓慢。提前终止。")
                  result['status'] = 'FAILED_ITERATION_LIMIT'
                  break # Exit loop if iteration limit is reached

        # 5. Results
        # The selected_k_subset_indices contains the *indices* of the blocks chosen in each step.
        # Convert these indices back to the actual sets (sorted lists).
        chosen_sets_list = [list(sorted(list(all_k_subsets[idx]))) for idx in selected_k_subset_indices]
        result['sets'] = chosen_sets_list
        result['j_subsets_covered'] = num_j_subsets - len(needs_more_coverage_j_indices) # The number of j-subsets that met the c requirement

        # Determine final status based on whether all j-subsets were covered
        if not needs_more_coverage_j_indices: # If the set of unsatisfied j-subsets is empty
            result['status'] = 'SUCCESS'
            print(f"[Greedy-{run_idx}] Greedy algorithm successfully completed c={c} coverage for all {num_j_subsets} j-subsets, selecting a total of {len(chosen_sets_list)} sets.")
        else:
            # If loop exited but needs_more_coverage_j_indices is not empty, coverage is incomplete.
            # Check if status was already set to an error/limit state inside the loop.
            if result['status'] == 'INIT': # If status hasn't been set by an earlier error
                result['status'] = 'FAILED_INCOMPLETE_COVER' # Default failure status

            print(f"[Greedy-{run_idx}] 贪心算法结束，状态: {result['status']}。{result['j_subsets_covered']}/{num_j_subsets} 个 j-子集满足了 c={c} 覆盖。选择了 {len(chosen_sets_list)} 个集合。")

    except MemoryError:
        print(f"[Greedy-{run_idx}] 内存错误：计算过程中内存不足。")
        result['status'] = 'ERROR_MEMORY'
        result['error_message'] = 'Memory error during execution.'
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
        k_subset_indices = {subset: i for i, subset in enumerate(all_k_subsets)} # Map subset to index
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
        # x[i] = 1 if the i-th k-subset (all_k_subsets[i]) is selected, 0 otherwise.
        x = [model.NewBoolVar(f'x_{i}') for i in range(num_k_subsets)]

        # 5. Define constraints: Each j-subset must be covered by at least c satisfying k-subsets
        print(f"[ILP-{run_idx}] Starting to add constraints (each j-subset >= {c} times coverage)...")
        constraints_added = 0
        feasible = True # Mark whether the model is potentially feasible

        # Precompute which k-subsets can potentially cover each j-subset to speed up constraint building
        j_subset_potential_covers = defaultdict(list) # Map: j_subset_index -> [k_subset_index1, k_subset_index2, ...]
        for idx_k, k_subset in enumerate(all_k_subsets):
             for idx_j, j_subset in enumerate(target_j_subsets):
                 if len(j_subset.intersection(k_subset)) >= s:
                     j_subset_potential_covers[idx_j].append(idx_k)

        # Add constraints for each target j-subset
        for idx_j, j_subset in enumerate(target_j_subsets):
            # Find the indices of all k_subsets whose intersection with the current j_subset is >= s
            # These are the k-subsets that *can* contribute to covering this j-subset.
            covering_k_indices = j_subset_potential_covers.get(idx_j, [])

            # Check feasibility constraint: Can this j-subset *ever* be covered c times?
            if len(covering_k_indices) < c:
                print(f"[ILP-{run_idx}] 错误：j-子集 {idx_j} ({set(j_subset)}) 最多只能被 {len(covering_k_indices)} 个 k-子集覆盖 (交集>=s)，无法满足 c={c} 的要求。问题不可行。")
                result['status'] = 'INFEASIBLE' # Directly mark model as infeasible
                result['error_message'] = f"j-subset {idx_j} cannot be covered {c} times (max possible: {len(covering_k_indices)})."
                feasible = False
                break # No need to add more constraints, problem is already known to be infeasible

            # Add the core constraint: the sum of selected covering k-subsets must be >= c
            # Only add constraint if there are potential covering blocks and c >= 1
            if covering_k_indices and c >= 1:
                # Constraint: sum(x[i] for i in covering_k_indices) >= c
                model.Add(sum(x[i] for i in covering_k_indices) >= c)
                constraints_added += 1
            elif c < 1:
                 # If c is 0 or negative (already checked at start, but as defense), no constraint needed.
                 pass
            # elif not covering_k_indices and c >= 1: # Handled by the feasibility check above

        if not feasible:
             # If pre-check found infeasibility, record time and exit
             result['time'] = time.time() - start_time
             q.put(result)
             return

        print(f"[ILP-{run_idx}] Added coverage constraints for {constraints_added}/{num_j_subsets} j-subsets requiring coverage (c={c}).")

        # 6. Define the objective function: Minimize the total number of selected k-subsets
        model.Minimize(sum(x))

        # 7. Create the solver and set parameters
        solver = cp_model.CpSolver()
        # Set the time limit provided by the caller
        solver.parameters.max_time_in_seconds = float(timeout_solver)
        # Optional: Log search progress (can be very verbose)
        # solver.parameters.log_search_progress = True
        # Try using multiple workers if CPU allows, often speeds up search
        try:
            num_workers = mp.cpu_count()
            # Avoid using too many workers if CPU count is low or OS limits apply
            solver.parameters.num_search_workers = max(1, num_workers // 2 if num_workers > 1 else 1)
            if solver.parameters.num_search_workers > 1:
                 print(f"[ILP-{run_idx}] Solving with {solver.parameters.num_search_workers} workers...")
            else:
                 print(f"[ILP-{run_idx}] Solving with default number of workers (1)...")
        except NotImplementedError:
             print(f"[ILP-{run_idx}] Could not detect CPU count, solving with default workers.")
             solver.parameters.num_search_workers = 1 # Default fallback

        # 8. Solve the model
        print(f"[ILP-{run_idx}] Starting CP-SAT solver...")
        solve_start_time = time.time()
        status = solver.Solve(model)
        solve_end_time = time.time()
        solver_wall_time = solve_end_time - solve_start_time
        print(f"[ILP-{run_idx}] Solver finished in {solver_wall_time:.2f} seconds.")


        # 9. Process Results
        status_map = {
            cp_model.OPTIMAL: 'OPTIMAL',        # Found the optimal solution.
            cp_model.FEASIBLE: 'FEASIBLE',      # Found a feasible solution, but optimality not proven (often due to timeout).
            cp_model.INFEASIBLE: 'INFEASIBLE',    # Proven that no solution exists.
            cp_model.MODEL_INVALID: 'MODEL_INVALID',# The model formulation itself is invalid.
            cp_model.UNKNOWN: 'UNKNOWN'         # Solver stopped without a conclusive status (e.g., timeout before finding feasible solution).
        }
        result['status'] = status_map.get(status, f'UNMAPPED_STATUS_{status}') # Get mapped status or raw status code

        # Get objective value (number of sets) if a solution was found
        obj_value = float('inf')
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            try:
                 # solver.ObjectiveValue() gives the minimized sum(x)
                 obj_value = solver.ObjectiveValue()
            except OverflowError: # Handle potential overflow for very large objectives
                 print(f"[ILP-{run_idx}] Warning: Objective value calculation resulted in overflow.")
                 obj_value = float('inf') # Treat as infinite if overflow

        print(f"[ILP-{run_idx}] 求解完成。状态: {result['status']}, 目标值(集合数量): {int(obj_value) if obj_value != float('inf') else 'N/A'}")

        # Extract the solution (selected k-subsets) if optimal or feasible
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            selected_indices = [i for i, var in enumerate(x) if solver.Value(var) == 1]
            chosen_sets = [all_k_subsets[i] for i in selected_indices] # Get the frozensets
            # Convert to list of lists (sorted elements) for consistent output
            result['sets'] = [list(sorted(list(s))) for s in chosen_sets]
            print(f"[ILP-{run_idx}] 找到 {len(result['sets'])} 个集合构成解。")
            # Sanity check: verify the number of sets matches the objective value
            if len(result['sets']) != int(obj_value):
                print(f"[ILP-{run_idx}] 警告：找到的集合数量 ({len(result['sets'])}) 与报告的目标值 ({int(obj_value)}) 不符！")
                # This might indicate an issue with solution extraction or the solver's reporting.
        elif status == cp_model.UNKNOWN:
             print(f"[ILP-{run_idx}] 求解器在 {timeout_solver} 秒内未能找到可行解或证明不可行。状态 UNKNOWN，通常因超时或问题复杂。")
             result['error_message'] = f"Solver timed out ({timeout_solver}s) or stopped with UNKNOWN status."
        elif status == cp_model.INFEASIBLE:
             print(f"[ILP-{run_idx}] 模型被证明不可行，不存在满足 c={c} 覆盖条件的解。")
             # Error message might have been set during constraint check, or can be set here.
             if 'error_message' not in result:
                 result['error_message'] = "The problem is proven to be infeasible by the solver."
        else: # Handle MODEL_INVALID or other unmapped statuses
             print(f"[ILP-{run_idx}] 求解器返回了非预期状态：{result['status']}")
             result['error_message'] = f"Solver returned unexpected status: {result['status']}"

    except FileNotFoundError as fnf_err: # Specifically catch potential OR-Tools dependency issues
        print(f"[ILP-{run_idx}] CP-SAT 文件错误: {fnf_err}。确认 OR-Tools 安装完整且路径正确。")
        result['status'] = 'ERROR_ORTOOLS_FILE'
        result['error_message'] = str(fnf_err)
        # This might happen if OR-Tools installation is broken or has missing native libraries.
    except MemoryError:
        print(f"[ILP-{run_idx}] 内存错误：CP-SAT 计算过程中内存不足。")
        result['status'] = 'ERROR_MEMORY'
        result['error_message'] = 'Memory error during CP-SAT execution.'
    except Exception as e:
        print(f"[ILP-{run_idx}] CP-SAT 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        result['status'] = 'ERROR'
        result['error_message'] = str(e)

    finally:
        result['time'] = time.time() - start_time # 确保记录最终时间 (包括模型构建和求解)
        try:
            q.put(result)
            print(f"[ILP-{run_idx}] 结果已放入队列。总耗时: {result['time']:.2f} 秒。")
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
            timeout (int): **ILP 优先等待时间 (秒)** (参照代码 B 的 'wait' 逻辑).
                           *注意：这里 timeout 的语义已根据请求调整为优先等待 ILP 的时间*
            rand_instance (random.Random, 可选): 用于生成宇宙的随机数生成器实例
        """
        self.m = m
        self.n = n
        self.k = k
        self.j = j
        self.s = s
        self.c = c # 存储实际使用的覆盖次数要求 (c)
        self.run_idx = run_idx # 持久化索引
        self.timeout = timeout # timeout 现在代表 ILP 的优先等待时间
        self.rand = rand_instance if rand_instance else random

        # --- 结果属性初始化 ---
        self.univ = []        # 将使用的 n 个元素的宇宙 (可能被手动输入覆盖)
        self.q = mp.Queue()   # 用于接收算法结果的队列
        self.result = {}      # 存储选定算法的结果字典
        self.sets = []        # 存储选定算法找到的覆盖集列表
        self.ans = None       # 最终格式化的结果字符串

        # 使用 c 打印
        print(f"[Sample-{self.run_idx}] 初始化实例: M={m}, N={n}, K={k}, J={j}, S={s}, Target_C={c}, RunIndex={run_idx}, ILP_WaitTimeout={timeout}s") # 显示 run_idx 和调整后的 timeout 含义

        # 1. 生成宇宙 (Universe) - 检查逻辑保持不变
        if n > m:
            raise ValueError(f"错误：N ({n}) 不能大于 M ({m})")
        if not (n > 0 and m >= n):
            print(f"[Sample-{self.run_idx}] M ({m}) 或 N ({n}) 无效，Universe 必须由外部提供或在 run() 中生成。")
            if n <= 0 or m <= 0:
                 raise ValueError("M 和 N 必须为正数。")
        # (Universe 的实际生成/设置通常在调用 run 之前由 main.py 完成)


    def run(self):
        """
        启动 Greedy 和 CP-SAT (如果可用) 算法并行计算。
        修改后的结果选择逻辑：优先等待 ILP 在指定的 `self.timeout` 秒内返回 OPTIMAL/FEASIBLE 解。
        如果 ILP 未能在此时间内成功返回，则等待并使用 Greedy 的 SUCCESS 解。
        如果两者都未成功，则使用回退逻辑（优先 ILP 失败状态，然后 Greedy 失败状态）。
        """
        print(f"[RUN-{self.run_idx}] 开始运行并行计算 (目标 c={self.c})... ILP 优先等待: {self.timeout}s")

        # 检查 Universe 是否有效 - 逻辑保持不变
        if not self.univ or len(self.univ) != self.n:
             if self.n > 0 and self.m >= self.n:
                 self.univ = sorted(self.rand.sample(range(1, self.m + 1), self.n))
                 print(f"[RUN-{self.run_idx}] 警告：外部未提供有效 Universe，内部随机生成: {self.univ}")
             else:
                 print(f"[RUN-{self.run_idx}] 错误：Universe 无效或大小与 N 不符，且无法生成默认。Univ: {self.univ}, N: {self.n}")
                 self.ans = f"Fail: Invalid Universe"
                 self.result = {'status': 'Error', 'alg': 'Preprocessing', 'sets': [], 'time': 0, 'error': 'Invalid Universe', 'run_index': self.run_idx}
                 self.sets = []
                 return

        processes = []
        results_map = {} # 用于存储收到的结果
        selected_result = None
        time_start_run = time.time() # 记录 run 方法开始时间

        # -- 准备参数元组 --
        common_args = (self.q, self.univ, self.k, self.j, self.s, self.c, self.run_idx)

        # -- 启动 Greedy 进程 --
        args_greedy = common_args
        p_greedy = mp.Process(target=greedy_cover, args=(args_greedy,), daemon=True)
        processes.append(p_greedy)
        p_greedy.start()
        print(f"[RUN-{self.run_idx}] 启动了 Greedy 进程 (PID: {p_greedy.pid}, 目标 c={self.c})")

        # -- 启动 CP-SAT 进程 (如果可用) --
        p_ilp = None
        if HAS_ORTOOLS:
            # CP-SAT 求解器的内部超时时间可以设置得比优先等待时间稍长或相等，
            # 因为我们主要关心的是它是否在优先时间内返回 *成功* 结果。
            # 这里设置为 self.timeout，与优先等待时间一致。
            timeout_solver_internal = max(1.0, float(self.timeout)) # 保证至少1秒

            args_ilp = common_args + (timeout_solver_internal,) # 添加 solver 内部超时
            p_ilp = mp.Process(target=cpsat_cover, args=(args_ilp,), daemon=True)
            processes.append(p_ilp)
            p_ilp.start()
            print(f"[RUN-{self.run_idx}] 启动了 CP-SAT (ILP) 进程 (PID: {p_ilp.pid}, 目标 c={self.c}), 内部超时 {timeout_solver_internal:.1f}s")
        else:
            print(f"[RUN-{self.run_idx}] CP-SAT (ILP) 求解器不可用，仅运行 Greedy。")

        # ==============================================================
        # ========== 修改后的结果选择逻辑 (模仿 Code B) 开始 ==========
        # ==============================================================
        ilp_completed_successfully_within_timeout = False
        greedy_completed = False
        greedy_result_data = None
        ilp_result_data = None

        # 优先等待 ILP 在 self.timeout 时间内完成并返回结果
        print(f"[RUN-{self.run_idx}] 优先等待 ILP 结果，最多 {self.timeout:.1f} 秒...")
        wait_start_time = time.time()
        time_waited = 0
        max_wait = self.timeout

        try:
            while time_waited < max_wait:
                remaining_time = max_wait - time_waited
                if remaining_time <= 0: break # 超时

                try:
                    # 尝试非阻塞或短时阻塞地获取结果
                    res = self.q.get(timeout=min(0.1, remaining_time)) # 短暂等待，避免长时间阻塞
                    res_alg = res.get('alg')
                    res_status = res.get('status')
                    res_run_idx = res.get('run_index') # 检查run_index匹配

                    if res_run_idx != self.run_idx:
                        print(f"[RUN-{self.run_idx}] 警告：收到来自不同运行 ({res_run_idx}) 的结果，已忽略。")
                        continue # 忽略不匹配的结果

                    if res_alg == 'ILP':
                        ilp_result_data = res
                        print(f"[RUN-{self.run_idx}] 在 {time.time() - wait_start_time:.2f}s 收到 ILP 结果，状态: {res_status}")
                        # 检查是否是成功的 ILP 结果
                        if res_status in ('OPTIMAL', 'FEASIBLE') and res.get('sets') is not None:
                            ilp_completed_successfully_within_timeout = True
                            selected_result = ilp_result_data # **立即选择 ILP 结果**
                            print(f"[RUN-{self.run_idx}] **选择 ILP 结果** (状态: {res_status}, 集合数: {len(selected_result['sets'])})，因为它在优先时间内成功返回。")
                            break # 找到满意的 ILP 结果，停止等待
                        else:
                             # ILP 完成但未成功 (INFEASIBLE, UNKNOWN, ERROR等)
                             # 继续等待看 Greedy 是否完成，或者等待超时
                             pass
                    elif res_alg == 'Greedy':
                        greedy_result_data = res
                        greedy_completed = True
                        print(f"[RUN-{self.run_idx}] 在 {time.time() - wait_start_time:.2f}s 收到 Greedy 结果，状态: {res_status}")
                        # 不立即选择 Greedy，继续等待 ILP 或超时
                    else:
                         print(f"[RUN-{self.run_idx}] 警告：收到未知算法 ({res_alg}) 的结果，已忽略。")

                    # 如果两个进程都已完成（且我们尚未选择 ILP），可以提前退出等待
                    if ilp_result_data is not None and greedy_result_data is not None and not ilp_completed_successfully_within_timeout:
                        break

                except queue.Empty:
                    # 队列为空，继续等待
                    pass
                except Exception as q_err:
                     print(f"[RUN-{self.run_idx}] 从队列获取结果时出错: {q_err}")
                     break # 出现错误，停止等待

                # 更新等待时间
                time_waited = time.time() - wait_start_time

            # --- 优先等待结束后的决策 ---
            if selected_result: # 如果 ILP 已被选中
                print(f"[RUN-{self.run_idx}] ILP 已在优先时间内成功返回并被选中。")
            else:
                # ILP 未在优先时间内成功返回
                print(f"[RUN-{self.run_idx}] ILP 未在 {self.timeout:.1f}s 优先等待时间内成功返回。现在检查/等待 Greedy 结果...")

                # 检查 Greedy 是否已经完成
                if greedy_completed and greedy_result_data and greedy_result_data.get('status') == 'SUCCESS' and greedy_result_data.get('sets') is not None:
                    selected_result = greedy_result_data # 选择已完成的 Greedy 成功结果
                    print(f"[RUN-{self.run_idx}] **选择 Greedy 结果** (状态: SUCCESS, 集合数: {len(selected_result['sets'])})，因为 ILP 未优先成功。")
                else:
                    # Greedy 尚未完成，或者已完成但未成功
                    # 需要继续等待 Greedy (可能超过 self.timeout)
                    if not greedy_completed and p_greedy.is_alive():
                        print(f"[RUN-{self.run_idx}] Greedy 进程仍在运行，继续等待其完成...")
                        remaining_overall_timeout = 3600 # 设置一个较长的总超时，避免无限等待
                        try:
                             # 循环获取，直到拿到 Greedy 的结果或超时
                             while greedy_result_data is None:
                                 res = self.q.get(timeout=remaining_overall_timeout) # 等待较长时间
                                 if res.get('alg') == 'Greedy' and res.get('run_index') == self.run_idx:
                                     greedy_result_data = res
                                     greedy_completed = True
                                     print(f"[RUN-{self.run_idx}] Greedy 结果终于收到，状态: {res.get('status')}")
                                     break
                                 elif res.get('alg') == 'ILP' and ilp_result_data is None and res.get('run_index') == self.run_idx:
                                     # 如果 ILP 结果现在才到，也记录下来
                                     ilp_result_data = res
                                     print(f"[RUN-{self.run_idx}] 晚到的 ILP 结果收到，状态: {res.get('status')}")
                                 # 忽略其他不相关结果
                        except queue.Empty:
                             print(f"[RUN-{self.run_idx}] 错误：在额外等待时间内仍未收到 Greedy 结果。")
                             # greedy_result_data 仍然是 None
                        except Exception as q_err_long:
                             print(f"[RUN-{self.run_idx}] 在额外等待 Greedy 时从队列获取结果出错: {q_err_long}")

                    # 再次检查 Greedy 结果是否可用且成功
                    if greedy_result_data and greedy_result_data.get('status') == 'SUCCESS' and greedy_result_data.get('sets') is not None:
                         selected_result = greedy_result_data
                         print(f"[RUN-{self.run_idx}] **选择 Greedy 结果** (状态: SUCCESS, 集合数: {len(selected_result['sets'])})，在额外等待后收到。")
                    else:
                         # Greedy 未成功或未收到结果，进入回退逻辑
                         print(f"[RUN-{self.run_idx}] Greedy 未成功或无法获取结果。进入回退选择逻辑...")
                         # 回退：优先选择任何 ILP 结果（即使失败），其次选择任何 Greedy 结果
                         if ilp_result_data: # 优先 ILP (即使是 INFEASIBLE, UNKNOWN, ERROR)
                             selected_result = ilp_result_data
                             print(f"[RUN-{self.run_idx}] **选择 ILP 结果作为回退** (状态: {selected_result.get('status', 'N/A')})")
                         elif greedy_result_data: # 其次 Greedy (即使是 FAILED, ERROR)
                             selected_result = greedy_result_data
                             print(f"[RUN-{self.run_idx}] **选择 Greedy 结果作为回退** (状态: {selected_result.get('status', 'N/A')})")
                         else:
                              # 连一个结果都没有收到
                              print(f"[RUN-{self.run_idx}] 错误：无法从队列中获取任何有效结果。")
                              total_wait_time = time.time() - wait_start_time
                              self.ans = f"Fail: No valid result obtained."
                              self.result = {'status': 'NoResult', 'alg': 'None', 'sets': [], 'time': total_wait_time, 'run_index': self.run_idx}
                              self.sets = []
                              # (后续的 finally 会处理进程终止)

        except (queue.Empty) as toe:
             # 这个 Empty 异常主要应该在初始等待时触发 (虽然内部循环也可能触发)
             total_wait_time = time.time() - wait_start_time
             print(f"[RUN-{self.run_idx}] 错误：在等待结果时队列为空或超时 ({type(toe).__name__})。总运行时间: {time.time() - time_start_run:.2f}s。")
             self.ans = f"Fail: Timeout or Queue Empty ({self.timeout:.1f}s priority wait)"
             # 检查是否有部分结果已记录
             if ilp_result_data:
                 selected_result = ilp_result_data
                 if 'status' not in selected_result or selected_result.get('status') in ('INIT', 'UNKNOWN'): selected_result['status'] = 'TIMEOUT_PARTIAL'
                 print(f"[RUN-{self.run_idx}] **选择部分 ILP 结果 (超时后)** (状态: {selected_result.get('status')})")
             elif greedy_result_data:
                 selected_result = greedy_result_data
                 if 'status' not in selected_result or selected_result.get('status') in ('INIT'): selected_result['status'] = 'TIMEOUT_PARTIAL'
                 print(f"[RUN-{self.run_idx}] **选择部分 Greedy 结果 (超时后)** (状态: {selected_result.get('status')})")
             else: # 如果真的一个结果都没有
                 self.result = {'status': 'Timeout', 'alg': 'None', 'sets': [], 'time': total_wait_time, 'run_index': self.run_idx}
             if selected_result: self.result = selected_result # 使用部分结果
             self.sets = self.result.get('sets', []) if self.result else []


        except Exception as e:
            total_time = time.time() - time_start_run
            print(f"[RUN-{self.run_idx}] 处理结果时发生意外错误: {e}")
            import traceback
            traceback.print_exc()
            self.ans = f"Error: Exception in result processing"
            # 尝试从部分结果中恢复信息
            part_res = ilp_result_data or greedy_result_data
            err_alg = part_res.get('alg', 'Error') if part_res else 'Error'
            self.result = {'status': 'RuntimeError', 'alg': err_alg, 'sets': [], 'time': total_time, 'error': str(e), 'run_index': self.run_idx}
            self.sets = []
        # ============================================================
        # ========== 修改后的结果选择逻辑 (模仿 Code B) 结束 ==========
        # ============================================================

        finally:
            # 确保所有进程都已终止 - 逻辑保持不变
            print(f"[RUN-{self.run_idx}] 尝试终止子进程...")
            for p in processes:
                pid_str = f"PID {p.pid}" if p.pid else "进程"
                try:
                    if p.is_alive():
                        print(f"  - 正在终止 {pid_str}...")
                        p.terminate() # 发送 SIGTERM
                        p.join(timeout=1.0) # 等待1秒
                        if p.is_alive(): # 如果还在运行
                            print(f"  - {pid_str} 未能正常终止，强制终止...")
                            p.kill() # 发送 SIGKILL
                            p.join(timeout=0.5) # 短暂等待
                        print(f"  - {pid_str} 已终止。")
                    else:
                        print(f"  - {pid_str} 已结束。")
                except Exception as term_err:
                    print(f"  - 终止 {pid_str} 时出错: {term_err}")
            # 清理队列 - 逻辑保持不变
            print(f"[RUN-{self.run_idx}] 清理结果队列...")
            while not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: break
                except Exception as q_clean_err:
                    print(f"  - 清理队列时出错: {q_clean_err}")
                    break # 避免无限循环
            print(f"[RUN-{self.run_idx}] Run 方法结果处理和清理完成。")

        # --- 设置最终结果属性 --- (逻辑保持不变, 但基于 selected_result)
        if selected_result: # 检查 selected_result 是否被成功赋值
            self.result = selected_result
            # 确保 run_index 在最终结果中
            if 'run_index' not in self.result or self.result['run_index'] != self.run_idx:
                 print(f"[RUN-{self.run_idx}] 警告：最终选择的结果缺少正确的 run_index，将强制设置为 {self.run_idx}。 Result: {self.result.get('run_index')}")
                 self.result['run_index'] = self.run_idx

            self.sets = self.result.get('sets', []) # 获取集合列表，默认为空
            # 确保 self.sets 是列表
            if not isinstance(self.sets, list):
                 print(f"[RUN-{self.run_idx}] 警告：最终结果中的 'sets' 不是列表 (类型: {type(self.sets)})，将重置为空列表。")
                 self.sets = []

            num_results = len(self.sets)

            # 构建标准答案字符串: m-n-k-j-s-run_idx-num_sets
            self.ans = f"{self.m}-{self.n}-{self.k}-{self.j}-{self.s}-{self.run_idx}-{num_results}"

            final_alg = self.result.get('alg', 'N/A')
            final_status = self.result.get('status', 'N/A')
            final_time = self.result.get('time', 0) # 算法内部报告的时间
            cov_target = self.result.get('coverage_target', self.c) # 确认 c 值
            final_run_idx = self.result.get('run_index', self.run_idx) # 再次确认

            print(f"[RUN-{final_run_idx}] ---- Final Result Summary ----")
            print(f"[RUN-{final_run_idx}] Selected Algorithm: {final_alg}")
            print(f"[RUN-{final_run_idx}] Status: {final_status}")
            print(f"[RUN-{final_run_idx}] Coverage Target (c): {cov_target}")
            print(f"[RUN-{final_run_idx}] Algorithm Time: {final_time:.2f}s") # 算法内部时间
            print(f"[RUN-{final_run_idx}] Total Run Method Time: {time.time() - time_start_run:.2f}s") # run方法的总时间
            print(f"[RUN-{final_run_idx}] Sets Found: {num_results}")
            print(f"[RUN-{final_run_idx}] Result ID (ans): {self.ans}")
            print(f"[RUN-{final_run_idx}] -----------------------------")

        elif self.ans is None: # 如果 selected_result 为 None 且未在异常中设置 ans
             # self.result 可能在异常处理中被设置，或者保持为空
             fail_status = self.result.get('status', 'UnknownFailure') if hasattr(self, 'result') and self.result else 'SetupFailure'
             num_results = 0 # 失败时集合数为 0
             self.ans = f"Fail({fail_status}):{self.m}-{self.n}-{self.k}-{self.j}-{self.s}-{self.run_idx}-{num_results}"
             # 确保 self.result 存在且包含基本信息
             if not hasattr(self, 'result') or not self.result:
                 self.result = {'status': fail_status, 'alg': 'None', 'sets': [], 'time': time.time() - time_start_run, 'run_index': self.run_idx}
             elif 'run_index' not in self.result:
                  self.result['run_index'] = self.run_idx # 添加run_index
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
    test_c = 2 # 使用 c
    test_ilp_wait_timeout = 15 # ** 修改: timeout 现在代表 ILP 优先等待时间 **


    print(f"\n测试参数: M={test_m}, N={test_n}, K={test_k}, J={test_j}, S={test_s}, C={test_c}, ILP_WaitTimeout={test_ilp_wait_timeout}s")

    try:
        # 获取持久化的 run_index
        test_run_idx = test_db.get_and_increment_run_index(test_m, test_n, test_k, test_j, test_s)
        if test_run_idx is None:
             print("错误：无法从数据库获取运行索引。")
             exit()
        print(f"从数据库获取的本次运行索引: {test_run_idx}")

        test_random_instance = random.Random(0) # 固定种子
        # 使用 c 和调整后的 timeout 实例化 Sample
        sample_instance = Sample(test_m, test_n, test_k, test_j, test_s, test_c,
                                 test_run_idx,
                                 test_ilp_wait_timeout, # 传递 ILP 优先等待时间
                                 test_random_instance)

        # 手动设置一个 Universe
        sample_instance.univ = sorted(test_random_instance.sample(range(1, test_m + 1), test_n))
        print(f"设置 Universe: {sample_instance.univ}")

        sample_instance.run()

        print("\n--- 测试运行结果 ---")
        print(f"最终结果标识 (ans): {sample_instance.ans}")
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