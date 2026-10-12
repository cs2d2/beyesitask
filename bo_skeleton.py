"""
贝叶斯优化完整骨架
- 多目标（EHVI）
- 可行性加权（FeasibilityWeightedAcq）
- 批量候选点（q）
- 维度追踪打印

运行: python bo_skeleton.py
依赖: pip install torch botorch gpytorch
"""

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from gpytorch.mlls import ExactMarginalLogLikelihood

# ============================================================
# 0. 全局设置
# ============================================================
tkwargs = {"device": "cpu", "dtype": torch.double}


# ============================================================
# 1. 数据层
# ============================================================
class DataStore:
    """存储所有试验数据。目标数据只存可行的，约束数据存全部。"""

    def __init__(self, n_params, n_objectives):
        self.n_params = n_params          # = d
        self.n_objectives = n_objectives  # = m

        # 目标数据（只存可行）
        self.X_obj = torch.empty((0, n_params), **tkwargs)
        self.Y_obj = torch.empty((0, n_objectives), **tkwargs)

        # 约束数据（全部试验）
        self.X_con = torch.empty((0, n_params), **tkwargs)
        self.Y_con = torch.empty((0, 1), **tkwargs)

        # 完整记录
        self.trials = []

    def add(self, x, y, feasible):
        """
        x: (d,)    一个候选点，d 维参数
        y: (m,)    一个目标向量，m 维目标
        feasible:  bool
        """
        x = x.unsqueeze(0)                       # (d,) -> (1, d)
        feas_tensor = torch.tensor([[float(feasible)]], **tkwargs)  # (1, 1)

        # 约束数据：全部加入
        self.X_con = torch.cat([self.X_con, x])           # (n_con+1, d)
        self.Y_con = torch.cat([self.Y_con, feas_tensor]) # (n_con+1, 1)

        # 目标数据：只有可行才加入
        if feasible:
            self.X_obj = torch.cat([self.X_obj, x])                 # (n_obj+1, d)
            self.Y_obj = torch.cat([self.Y_obj, y.unsqueeze(0)])    # (n_obj+1, m)

        # 完整记录
        self.trials.append({"x": x.squeeze(0), "y": y, "feasible": feasible})

    def n_feasible(self):
        return self.X_obj.shape[0]

    def n_total(self):
        return self.X_con.shape[0]


# ============================================================
# 2. 模型层
# ============================================================
def fit_gp(X, Y):
    """
    X: (n, d)
    Y: (n, m)
    返回: GP 或 None（失败）
    """
    try:
        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp
    except Exception as e:
        print(f"[fit_gp] 失败: {e}")
        return None


# ============================================================
# 3. 采集层
# ============================================================
class FeasibilityWeightedAcq(AcquisitionFunction):
    """
    EHVI × 可行性概率。

    ⚠️ 注意：qLogEHVI 返回的是 **log 空间** 的值（log(EHVI) < 0）。
    log 值是负数，如果直接 `log_ehvi * p`（p ∈ [0,1]），负数会变大 ——
    也就是说"越不可行分数越高"，优化器会一路奔向不可行区域。
    正确做法是在 log 空间做加法：

        log(EHVI × P) = log(EHVI) + log(P)

    这与线性空间的 `EHVI × P` 完全等价，同时保留了 logEI 的数值稳定性。
    """

    def __init__(self, base_acq, con_model):
        super().__init__(base_acq.model)
        self.base_acq = base_acq
        self.con_model = con_model

    def forward(self, X):
        log_ehvi = self.base_acq(X)                        # (b,) log(EHVI)，整批 q 共享一个值
        post = self.con_model.posterior(X)                 # 后验
        prob_feas = torch.sigmoid(post.mean.squeeze(-1))   # (..., q) 每个候选点的可行概率

        log_prob = torch.log(prob_feas.clamp_min(1e-12))   # (..., q)
        if log_prob.dim() > log_ehvi.dim():
            # EHVI 是"整批 q 个点"的联合值（没有 q 维），所以把 q 维求和加回去：
            # 假设各点独立 → P(整批可行) = ∏_i p_i → log = Σ_i log p_i
            log_prob = log_prob.sum(dim=-1)                # (..., q) -> (b,)
        return log_ehvi + log_prob


def select_candidates(obj_model, con_model, ref_point, X_baseline,
                      bounds, batch_size=1, n_restarts=10, n_raw_samples=256,
                      Y_baseline=None):
    """
    obj_model: 目标 GP
    con_model: 约束 GP（可为 None）
    ref_point: (m,)
    X_baseline: (n_obj, d)
    Y_baseline: (n_obj, m) 原始观测（新 BoTorch 接口需要，见下方注释）
    bounds: (2, d)
    batch_size: q
    返回: candidates (q, d)
    """
    # 1. 从 GP 后验采样的采样器
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

    # 2. 标准 EHVI
    # 🔧 兼容两代 BoTorch 的 qLogEHVI 构造函数：
    #    · 旧接口（<= 0.9.x）：用 X_baseline 观测点 + prune_baseline 自动推前沿
    #    · 新接口（>= 0.10 ）：改为显式传入 NondominatedPartitioning（同一个数学对象，
    #      只是把"当前帕累托前沿"从 X 空间挪到 Y 空间手动构造）
    #    语义完全一致：都是"相对当前前沿的超体积增量"。
    try:
        acq = qLogExpectedHypervolumeImprovement(
            model=obj_model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            prune_baseline=True,
        )
    except TypeError:
        if Y_baseline is None:
            # 兜底：从模型训练目标里取（注意：单一输出 GP 的 train_targets
            # 已被标准化，正常路径下请显式传原始 Y_baseline）
            Y_baseline = obj_model.train_targets
            if Y_baseline.shape[-1] != ref_point.shape[-1]:
                Y_baseline = Y_baseline.transpose(-1, -2)   # (m, n) -> (n, m)
        pareto_Y = Y_baseline[is_non_dominated(Y_baseline)]          # (n_pareto, m)
        partitioning = NondominatedPartitioning(
            ref_point=ref_point,
            Y=pareto_Y,
        )
        acq = qLogExpectedHypervolumeImprovement(
            model=obj_model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )

    # 3. 可行性加权
    if con_model is not None:
        acq = FeasibilityWeightedAcq(acq, con_model)

    # 4. 优化采集函数
    try:
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,        # (2, d)
            q=batch_size,         # q
            num_restarts=n_restarts,
            raw_samples=n_raw_samples,
        )
        return candidates.detach()   # (q, d)
    except Exception as e:
        print(f"[select_candidates] 失败，回退随机采样: {e}")
        return draw_sobol_samples(bounds=bounds, n=batch_size, q=1).squeeze(1)


# ============================================================
# 4. 评估层
# ============================================================
def objective_function(x_natural):
    """
    x_natural: (d,) 物理参数
    返回: (y, feasible)，y 是 (m,)
    ⚠️ 这里用假函数代替，替换成你自己的 NMPC 仿真。
    """
    # 假目标：最优点在 0.3 和 0.7
    obj1 = -((x_natural - 0.3) ** 2).sum()
    obj2 = -((x_natural - 0.7) ** 2).sum()
    y = torch.stack([obj1, obj2])          # (m,) = (2,)

    # 假可行性：参数和 > 0.5 则可行
    feasible = bool(x_natural.sum() > 0.5)
    return y, feasible


def evaluate(x_normalized, bounds):
    """
    x_normalized: (d,)
    bounds: (2, d)
    返回: (y, feasible)
    """
    x_nat = unnormalize(x_normalized, bounds)  # (d,)
    return objective_function(x_nat)


# ============================================================
# 5. 更新层
# ============================================================
def compute_hypervolume(Y, ref_point):
    if Y.shape[0] == 0:
        return 0.0
    bd = DominatedPartitioning(ref_point=ref_point, Y=Y)
    return bd.compute_hypervolume().item()


def compute_pareto(Y):
    if Y.shape[0] == 0:
        return Y, Y
    mask = is_non_dominated(Y)
    return Y[mask], Y[~mask]


# ============================================================
# 6. 主循环
# ============================================================
class BayesianOptimizer:
    def __init__(self, bounds, ref_point, n_params, n_objectives):
        self.bounds = bounds                  # (2, d)
        self.ref_point = ref_point            # (m,)
        self.n_params = n_params              # = d
        self.n_objectives = n_objectives      # = m

        self.normalized_bounds = torch.stack([
            torch.zeros(n_params, **tkwargs),  # (d,)
            torch.ones(n_params, **tkwargs),   # (d,)
        ])                                     # (2, d)

        self.data = DataStore(n_params, n_objectives)
        self.hv_trace = []

    def generate_initial_data(self, n):
        """生成 n 个随机初始点。"""
        print(f"\n[init] 生成 {n} 个初始点")
        print(f"  normalized_bounds 形状: {self.normalized_bounds.shape}")

        X = draw_sobol_samples(
            bounds=self.normalized_bounds,  # (2, d)
            n=n,                            # n
            q=1,                            # q
        )                                   # (n, 1, d)
        print(f"  draw_sobol_samples 返回: {X.shape}")

        X = X.squeeze(1)                    # (n, d)
        print(f"  squeeze 后: {X.shape}")

        for i, x in enumerate(X):
            y, feasible = evaluate(x, self.bounds)
            self.data.add(x, y, feasible)

        print(f"[init] 完成，可行 {self.data.n_feasible()} / 总 {self.data.n_total()}")

    def step(self, batch_size=1):
        """一次 BO 迭代。"""
        print(f"\n=== 一次 BO 迭代，batch_size(q)={batch_size} ===")

        # 1. 数据不足 → 随机采样
        if self.data.n_feasible() < 2:
            print("[step] 可行数据太少，随机采样。")
            candidates = draw_sobol_samples(
                bounds=self.normalized_bounds,
                n=batch_size,
                q=1,
            ).squeeze(1)                        # (q, d)
            print(f"  随机候选点: {candidates.shape}")
        else:
            # 2. 拟合模型
            print(f"  拟合目标 GP，X: {self.data.X_obj.shape}, Y: {self.data.Y_obj.shape}")
            obj_model = fit_gp(self.data.X_obj, self.data.Y_obj)

            print(f"  拟合约束 GP，X: {self.data.X_con.shape}, Y: {self.data.Y_con.shape}")
            con_model = fit_gp(self.data.X_con, self.data.Y_con)

            # 3. 拟合失败 → 随机采样
            if obj_model is None:
                print("[step] 目标模型拟合失败，随机采样。")
                candidates = draw_sobol_samples(
                    bounds=self.normalized_bounds, n=batch_size, q=1
                ).squeeze(1)
            else:
                # 4. 选候选点
                print(f"  选候选点，baseline: {self.data.X_obj.shape}")
                candidates = select_candidates(
                    obj_model=obj_model,
                    con_model=con_model,
                    ref_point=self.ref_point,
                    X_baseline=self.data.X_obj,
                    Y_baseline=self.data.Y_obj,
                    bounds=self.normalized_bounds,
                    batch_size=batch_size,
                )
                print(f"  候选点: {candidates.shape}")

        # 5. 评估
        for i, x in enumerate(candidates):
            y, feasible = evaluate(x, self.bounds)
            self.data.add(x, y, feasible)

        # 6. 更新指标
        hv = compute_hypervolume(self.data.Y_obj, self.ref_point)
        self.hv_trace.append(hv)
        pareto, _ = compute_pareto(self.data.Y_obj)
        print(f"[step] HV={hv:.4f}, 帕累托点数={pareto.shape[0]}, "
              f"总试验={self.data.n_total()}")

    def run(self, n_init=5, n_iter=10, batch_size=1):
        self.generate_initial_data(n_init)
        for i in range(n_iter):
            print(f"\n=== 迭代 {i+1}/{n_iter} ===")
            self.step(batch_size=batch_size)
        return self.data, self.hv_trace


# ============================================================
# 7. 入口
# ============================================================
if __name__ == "__main__":
    # 参数空间：d = 3 个参数
    d = 3
    m = 2

    bounds = torch.tensor([
        [0.0, 0.0, 0.0],   # 下界
        [1.0, 1.0, 1.0],   # 上界
    ], **tkwargs)          # (2, d)

    ref_point = torch.tensor([-1.0, -1.0], **tkwargs)  # (m,)

    print(f"参数维度 d = {d}")
    print(f"目标维度 m = {m}")
    print(f"bounds 形状: {bounds.shape}")
    print(f"ref_point 形状: {ref_point.shape}")

    optimizer = BayesianOptimizer(
        bounds=bounds,
        ref_point=ref_point,
        n_params=d,
        n_objectives=m,
    )

    data, hv_trace = optimizer.run(n_init=5, n_iter=5, batch_size=2)
    print("\n最终 HV 轨迹:", [round(h, 3) for h in hv_trace])
