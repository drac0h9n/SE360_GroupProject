# source.py

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
