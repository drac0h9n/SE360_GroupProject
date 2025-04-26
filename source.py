# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import itertools
# import math
# import random
# import threading
# import time
# import sys
# from itertools import combinations
# from math import comb
#
# try:
#     from ortools.sat.python import cp_model
# except ImportError:
#     cp_model = None
#
#
# # --------------- 精确 CP-SAT ----------------
# def exact_cover(univ, k, j, s, y):
#     if cp_model is None:
#         return None
#     n = len(univ)
#     model = cp_model.CpModel()
#     blocks = list(combinations(range(n), k))
#     x = [model.NewBoolVar(f'x{i}') for i in range(len(blocks))]
#     ssubs = list(combinations(range(n), s))
#     cov = [model.NewBoolVar(f'c{i}') for i in range(len(ssubs))]
#     for si, S in enumerate(ssubs):
#         lst = [x[bi] for bi, b in enumerate(blocks) if set(S) <= set(b)]
#         if lst:
#             model.Add(sum(lst) >= 1).OnlyEnforceIf(cov[si])
#             model.Add(sum(lst) == 0).OnlyEnforceIf(cov[si].Not())
#         else:
#             model.Add(cov[si] == 0)
#     for J in combinations(range(n), j):
#         idx = [cov[ssubs.index(S)] for S in combinations(J, s)]
#         model.Add(sum(idx) >= y)
#     model.Minimize(sum(x))
#     solver = cp_model.CpSolver()
#     solver.parameters.max_time_in_seconds = 1e9
#     solver.parameters.num_search_workers = 8
#     res = solver.Solve(model)
#     if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#         return [[univ[i] for i in blk] for blk, var in zip(blocks, x) if solver.Value(var)]
#     return None
#
#
# # --------------- 贪心交集计数 -----------------
# def greedy_cover(univ, k, j, s, y):
#     n = len(univ)
#     blocks = [tuple(b) for b in combinations(range(n), k)]
#     Bset = [set(b) for b in blocks]
#     Jlist = [tuple(J) for J in combinations(range(n), j)]
#     need = {J: y for J in Jlist}
#     unsat = set(Jlist)
#     sol = []
#     while unsat:
#         idx, maxg = None, -1
#         for i, B in enumerate(Bset):
#             g = sum(1 for J in unsat if len(B & set(J)) >= s)
#             if g > maxg: maxg, idx = g, i
#         if maxg <= 0: break
#         sol.append([univ[x] for x in blocks[idx]])
#         B = Bset[idx]
#         rm = []
#         for J in unsat:
#             if len(B & set(J)) >= s:
#                 need[J] -= 1
#                 if need[J] == 0: rm.append(J)
#         for J in rm: unsat.remove(J)
#     return sol
#
#
# # --------------- 主循环 ----------------------
# run_cnt = {}
#
#
# def get_elems(m, n):
#     if input("自己输入 n 元素? (y/n): ").strip().lower() == 'y':
#         try:
#             lst = list(map(int, input(f"输入 {n} 个数字: ").split()))
#             if len(set(lst)) == n and all(1 <= x <= m for x in lst):
#                 return sorted(lst)
#         except:
#             pass
#         print("非法,随机生成")
#     return sorted(random.sample(range(1, m + 1), n))
#
#
# def main_loop():
#     while True:
#         m = int(input("m: "));
#         n = int(input("n: "))
#         k = int(input("k: "));
#         s = int(input("s: "))
#         j = int(input(f"j({s}-{k}): "))
#         ytxt = input("y/all: ").strip().lower()
#         y = comb(j, s) if ytxt == "all" else int(ytxt)
#         if not (45 <= m <= 54 and 7 <= n <= 25 and 4 <= k <= 7 and 3 <= s <= 7 and s <= j <= k and 1 <= y <= comb(j,
#                                                                                                                   s)):
#             print("参数非法\n");
#             continue
#         elems = get_elems(m, n);
#         print("元素集:", elems)
#         wait = float(input("等待精确秒数: "))
#
#         key = (m, n, k, j, s);
#         run_cnt[key] = run_cnt.get(key, 0) + 1;
#         run_idx = run_cnt[key]
#         start = time.time()
#
#         result = {'sol': None, 'algo': None}
#         done = threading.Event();
#         lock = threading.Lock()
#
#         def exact_thread():
#             sol = exact_cover(elems, k, j, s, y)
#             if sol:
#                 with lock:
#                     if result['sol'] is None:
#                         result['sol'] = sol;
#                         result['algo'] = "Exact";
#                         done.set()
#
#         def greedy_thread():
#             sol = greedy_cover(elems, k, j, s, y)
#             # 等待剩余时间，再决定是否发布
#             rem = max(0, wait - (time.time() - start))
#             time.sleep(rem)
#             with lock:
#                 if result['sol'] is None:  # ILP 仍未完成
#                     result['sol'] = sol;
#                     result['algo'] = "Greedy";
#                     done.set()
#
#         threading.Thread(target=exact_thread, daemon=True).start()
#         threading.Thread(target=greedy_thread, daemon=True).start()
#
#         done.wait()  # 一定会等到某线程 set
#         sol = result['sol'];
#         algo = result['algo']
#         blk_num = len(sol) if sol else 0
#         for i, b in enumerate(sol, 1): print(f"Set {i}: {b}")
#         print(f"{m}-{n}-{k}-{j}-{s}-{run_idx}-{blk_num}")
#         print("解来自:", algo)
#         print(f"运行耗时: {time.time() - start:.2f} 秒\n")
#         if input("继续? (y/n): ").strip().lower() != 'y':
#             sys.exit(0)
#
#
# if __name__ == "__main__":
#     main_loop()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools, random, time, math, threading, sys
from collections import defaultdict
from ortools.sat.python import cp_model   # pip install ortools

run_counter = defaultdict(int)

# ---------------- 预计算 ----------------
def precompute(n, k, j, s):
    s_subs = list(itertools.combinations(range(n), s))
    s_id   = {sub:i for i,sub in enumerate(s_subs)}

    blocks, block_mask = [], []
    for blk in itertools.combinations(range(n), k):
        m = 0
        for sub in itertools.combinations(blk, s):
            m |= 1 << s_id[sub]
        blocks.append(blk)
        block_mask.append(m)

    j_masks = []
    for J in itertools.combinations(range(n), j):
        m = 0
        for sub in itertools.combinations(J, s):
            m |= 1 << s_id[sub]
        j_masks.append(m)
    return blocks, block_mask, j_masks

# ---------------- 贪心（回滚版） ----------------
def greedy_cover(block_mask, j_masks_init, y):
    # 复制 j_masks，因为函数中会修改
    j_masks = j_masks_init[:]
    J = len(j_masks)
    need   = [y]*J
    remain = set(range(J))
    chosen = []

    while remain:
        best_idx, best_gain = -1, -1
        for i, bm in enumerate(block_mask):
            gain = 0
            for jdx in remain:
                hit = (bm & j_masks[jdx]).bit_count()
                gain += min(hit, need[jdx])
            if gain > best_gain:
                best_gain, best_idx = gain, i
        if best_gain <= 0:
            break
        chosen.append(best_idx)
        bm_sel = block_mask[best_idx]
        done = []
        for jdx in remain:
            hit = (bm_sel & j_masks[jdx]).bit_count()
            if hit:
                need[jdx] -= hit
                j_masks[jdx] &= ~bm_sel
                if need[jdx] <= 0:
                    done.append(jdx)
        for jdx in done:
            remain.remove(jdx)
    return chosen

# ---------------- CP-SAT 精确 ----------------
def cpsat_cover(block_mask, j_masks, y, time_limit):
    B, J = len(block_mask), len(j_masks)
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(B)]

    # 为每条 j-组合单独建局部 cov 变量
    for jdx, jm in enumerate(j_masks):
        bits = jm
        cov_vars = []
        while bits:
            lb = bits & -bits
            v  = model.NewBoolVar(f"c_{jdx}_{lb}")
            cover = [x[i] for i,bm in enumerate(block_mask) if bm & lb]
            model.Add(sum(cover) >= 1).OnlyEnforceIf(v)
            model.Add(sum(cover) == 0).OnlyEnforceIf(v.Not())
            cov_vars.append(v)
            bits ^= lb
        model.Add(sum(cov_vars) >= y)

    model.Minimize(sum(x))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers  = 8
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sel = [i for i in range(B) if solver.BooleanValue(x[i])]
        return sel, solver.WallTime()
    return None, solver.WallTime()

# ---------------- 打印 ----------------
def show_sets(univ, blocks, idxs):
    for i, idx in enumerate(idxs, 1):
        print(f"Set {i}: {sorted(univ[e] for e in blocks[idx])}")

# ---------------- 主循环 ----------------
def main():
    random.seed(0)
    while True:
        try:
            m = int(input("m: "))
            n = int(input("n: "))
            k = int(input("k: "))
            j = int(input("j: "))
            s = int(input("s: "))
            y_raw = input("y (整数 or all): ").strip().lower()
            wait = float(input("ILP 优先等待秒数: "))
        except Exception:
            print("输入错误，重试\n"); continue

        y = math.comb(j, s) if y_raw == 'all' else int(y_raw)
        if y < 1:
            print("y 必须 ≥1\n"); continue

        # Universe
        if input("手动输入元素? (y/n): ").lower() == 'y':
            try:
                arr = list(map(int, input(f"输入 {n} 个 1~{m}: ").split()))
                assert len(arr) == n and all(1 <= x <= m for x in arr)
                univ = sorted(arr)
            except:
                print("非法，随机生成")
                univ = sorted(random.sample(range(1, m+1), n))
        else:
            univ = sorted(random.sample(range(1, m+1), n))
        print("Universe:", univ)

        run_counter[(m,n,k,j,s)] += 1
        run_idx = run_counter[(m,n,k,j,s)]

        blocks, block_mask, j_masks = precompute(n, k, j, s)

        result, evt = {}, threading.Event()

        def cp_thread():
            sol, t = cpsat_cover(block_mask, j_masks.copy(), y, wait)
            if sol:
                result.update(idx=sol, alg='ILP', time=t)
                evt.set()

        def greedy_thread():
            t0 = time.time()
            sol = greedy_cover(block_mask, j_masks.copy(), y)
            evt.wait(max(0, wait - (time.time() - t0)))
            if not evt.is_set():
                result.update(idx=sol, alg='Greedy', time=time.time() - t0)
                evt.set()

        threading.Thread(target=cp_thread, daemon=True).start()
        threading.Thread(target=greedy_thread, daemon=True).start()
        evt.wait()

        idxs = result['idx']
        show_sets(univ, blocks, idxs)
        print(f"{m}-{n}-{k}-{j}-{s}-{run_idx}-{len(idxs)}")
        print("来源:", result['alg'])
        print(f"耗时: {result['time']:.2f} 秒\n")

        if input("继续? (y/n): ").lower() != 'y':
            sys.exit(0)

if __name__ == "__main__":
    main()
