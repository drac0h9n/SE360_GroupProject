import math, sys, time, multiprocessing, pulp
from itertools import combinations
from math import comb


class Sample:
    def __init__(self, m, n, k, j, s, y):
        self.m = m
        self.n = n
        self.k = k
        self.j = j
        self.s = s
        self.y = y
        self.ans = None

    # ---------------- ILP 求解函数 ----------------
    def ilp_worker(self, queue, mode):
        """
        mode='A' -> 全覆盖； mode='B' -> 阈值覆盖
        结果通过 queue.put(val) 返回；无解或非 Optimal 返回 None
        """
        try:
            if mode == 'A':
                B = list(combinations(range(self.n), self.k))
                m = pulp.LpProblem('A', pulp.LpMinimize)
                x = pulp.LpVariable.dicts('x', range(len(B)), 0, 1, pulp.LpBinary)
                m += pulp.lpSum(x[i] for i in range(len(B)))
                for sub in combinations(range(self.n), self.s):
                    idx = [i for i, b in enumerate(B) if set(sub) <= set(b)]
                    m += pulp.lpSum(x[i] for i in idx) >= 1
            else:
                B = list(combinations(range(self.n), self.k))
                J = list(combinations(range(self.n), self.j))
                m = pulp.LpProblem('B', pulp.LpMinimize)
                x = pulp.LpVariable.dicts('x', range(len(B)), 0, 1, pulp.LpBinary)
                m += pulp.lpSum(x[i] for i in range(len(B)))
                for Jset in J:
                    idx = [i for i, b in enumerate(B) if len(set(b) & set(Jset)) >= self.s]
                    m += pulp.lpSum(x[i] for i in idx) >= self.y
            m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=600))
            val = None if pulp.LpStatus[m.status] != 'Optimal' else int(pulp.value(m.objective) + 0.5)
        except Exception:
            val = None
        queue.put(val)

    # ---------------- 近似快速算法 ----------------
    def greedy_A(self):
        if self.s > self.k: return None
        B = [tuple(b) for b in combinations(range(self.n), self.k)]
        subs = set(combinations(range(self.n), self.s))
        cov = [set(combinations(b, self.s)) for b in B]
        chosen = 0
        while subs:
            idx = max(range(len(B)), key=lambda i: len(cov[i] & subs))
            gain = cov[idx] & subs
            if not gain: return None
            subs -= gain;
            chosen += 1
        return chosen

    def greedy_B(self):
        if self.j < self.s or self.y < 1 or self.y > comb(self.j, self.s): return None
        B = [set(b) for b in combinations(range(self.n), self.k)]
        J = [set(t) for t in combinations(range(self.n), self.j)]
        need = [self.y] * len(J)
        chosen = 0
        while any(need):
            idx = max(range(len(B)),
                      key=lambda i: sum(need[p] and len(B[i] & J[p]) >= self.s for p in range(len(J))))
            gain = False
            for p, Jset in enumerate(J):
                if need[p] and len(B[idx] & Jset) >= self.s:
                    need[p] -= 1;
                    gain = True
            if not gain: return None
            chosen += 1
        return chosen

    # ---------------- 统一包装 ----------------
    def solve(self, timeout=10):
        t0 = time.time()
        # fast in main
        fast = self.greedy_A() if self.y == 'all' else self.greedy_B()
        # spawn ILP process
        q = multiprocessing.Queue()
        mode = 'A' if self.y == 'all' else 'B'
        p = multiprocessing.Process(target=self.ilp_worker,
                                    args=(q, mode))
        p.start()
        p.join(max(0, timeout - (time.time() - t0)))
        ilp_val = None
        if p.is_alive():
            p.terminate();
            p.join()
        else:
            ilp_val = q.get()
        self.ans = ilp_val if ilp_val is not None else fast

# # ---------------- CLI ----------------
# def main():
#     print("输入 n k j s y/all:")
#     n = int(input("n: "))
#     k = int(input("k: "))
#     j = int(input("j: "))
#     s = int(input("s: "))
#     y = int(input("y (整数或 all): ").strip().lower())
#     sample = Sample(45, n, k, j, s, y)
#     # ans = sample.solve()
#     # print(ans if ans is not None else '无可行解')
#     sample.solve()
#     print(sample.ans)
#     # sys.exit(0)
#
#
# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     main()
