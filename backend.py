import itertools, random, time, math, threading, sys
from collections import defaultdict
from ortools.sat.python import cp_model  # pip install ortools

run_counter = defaultdict(int)


class RunCounter:
    def __init__(self):
        self.counter = defaultdict(int)

    def increment(self, params_tuple):
        self.counter[params_tuple] += 1
        return self.counter[params_tuple]

    def get_count(self, params_tuple):
        return self.counter.get(params_tuple, 0)

    def reset(self, params_tuple=None):
        if params_tuple:
            self.counter[params_tuple] = 0
        else:
            self.counter.clear()


class Sample:
    def __init__(self, m, n, k, j, s, y, univ, timeout):
        self.m = m
        self.n = n
        self.k = k
        self.j = j
        self.s = s
        self.y = y
        self.univ = univ
        self.timeout = timeout
        self.ans = None
        self.sets = []
        self.result = None

    # ---------------- 预计算 ----------------
    def precompute(self):
        s_subs = list(itertools.combinations(range(self.n), self.s))
        s_id = {sub: i for i, sub in enumerate(s_subs)}

        blocks, block_mask = [], []
        for blk in itertools.combinations(range(self.n), self.k):
            m = 0
            for sub in itertools.combinations(blk, self.s):
                m |= 1 << s_id[sub]
            blocks.append(blk)
            block_mask.append(m)

        j_masks = []
        for J in itertools.combinations(range(self.n), self.j):
            m = 0
            for sub in itertools.combinations(J, self.s):
                m |= 1 << s_id[sub]
            j_masks.append(m)
        return blocks, block_mask, j_masks

    # ---------------- 贪心（回滚版） ----------------
    def greedy_cover(self, block_mask, j_masks_init):
        # 复制 j_masks，因为函数中会修改
        j_masks = j_masks_init[:]
        J = len(j_masks)
        need = [self.y] * J
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
    def cpsat_cover(self, block_mask, j_masks):
        B, J = len(block_mask), len(j_masks)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(B)]

        # 为每条 j-组合单独建局部 cov 变量
        for jdx, jm in enumerate(j_masks):
            bits = jm
            cov_vars = []
            while bits:
                lb = bits & -bits
                v = model.NewBoolVar(f"c_{jdx}_{lb}")
                cover = [x[i] for i, bm in enumerate(block_mask) if bm & lb]
                model.Add(sum(cover) >= 1).OnlyEnforceIf(v)
                model.Add(sum(cover) == 0).OnlyEnforceIf(v.Not())
                cov_vars.append(v)
                bits ^= lb
            model.Add(sum(cov_vars) >= self.y)

        model.Minimize(sum(x))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sel = [i for i in range(B) if solver.BooleanValue(x[i])]
            return sel, solver.WallTime()
        return None, solver.WallTime()

    # ---------------- 打印 ----------------
    def get_sets(self, blocks):
        for i, idx in enumerate(self.result['idx'], 1):
            set_str = f"Set {i}: {sorted(self.univ[e] for e in blocks[idx])}\n"
            self.sets.append(set_str)

    def run(self, run_idx):
        blocks, block_mask, j_masks = self.precompute()
        result, evt = {}, threading.Event()

        def cp_thread():
            sol, t = self.cpsat_cover(block_mask, j_masks.copy())
            if sol:
                result.update(idx=sol, alg='ILP', time=t)
                evt.set()

        def greedy_thread():
            t0 = time.time()
            sol = self.greedy_cover(block_mask, j_masks.copy())
            evt.wait(max(0, self.timeout - (time.time() - t0)))
            if not evt.is_set():
                result.update(idx=sol, alg='Greedy', time=time.time() - t0)
                evt.set()

        threading.Thread(target=cp_thread, daemon=True).start()
        threading.Thread(target=greedy_thread, daemon=True).start()
        evt.wait()

        self.result = result
        self.get_sets(blocks)
        self.ans = f"{self.m}-{self.n}-{self.k}-{self.j}-{self.s}-{run_idx}-{len(self.result['idx'])}"
