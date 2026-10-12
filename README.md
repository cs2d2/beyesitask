# 贝叶斯优化（多目标 EHVI + 可行性加权）

基于 BoTorch 的多目标贝叶斯优化骨架，用于给 NMPC / 车辆运动控制这类
「黑箱 + 昂贵仿真 + 有约束」的调参问题做寻优。

## 功能

- 多目标采集函数：`qLogExpectedHypervolumeImprovement`（EHVI）
- 可行性加权：`FeasibilityWeightedAcq` = EHVI × σ(约束 GP 预测均值)
- 批量候选点：`batch_size`(= q) 一次推荐多个点
- 维度追踪打印：每次迭代打印张量形状，方便排查 d / m / q 维度错误
- 只把**可行**样本喂给目标 GP，约束 GP 用**全部**样本，避免不可行点污染帕累托前沿

## 目录

```
贝叶斯优化/
├── bo_skeleton.py     # 主程序（数据层 / 模型层 / 采集层 / 评估层 / 更新层 / 主循环）
├── requirements.txt   # 依赖
└── README.md
```

## 环境

已创建专用 conda 环境 `bayesopt`（Python 3.10，torch 2.4.1+cpu / botorch 0.11.3 / gpytorch 1.12）。

```bash
conda activate bayesopt
```

如果要在别的机器上重建：

```bash
conda create -y -n bayesopt python=3.10
conda run -n bayesopt python -m pip install -r requirements.txt
```

## 运行

```bash
cd /home/chen/mytask1/贝叶斯优化
conda run -n bayesopt python bo_skeleton.py     # 或先 conda activate bayesopt
```

## 与原骨架的差异（3 处，都是为了跑通 + 修一个真 bug）

原骨架是按 BoTorch ≤ 0.9 的接口写的，装的是 0.11.3，有三处必须动：

1. **`qLogEHVI` 构造函数变了**。旧接口用 `X_baseline=观测点 + prune_baseline=True`
   自动推前沿；新接口要求显式传 `partitioning=NondominatedPartitioning(...)`。
   `select_candidates` 里用 try/except 同时兼容两代接口（同一个数学对象，只是把当前
   帕累托前沿从 X 空间挪到 Y 空间手动构造），并多了一个 `Y_baseline` 参数。

2. **`FeasibilityWeightedAcq` 的加权方式（真 bug）**。`qLogEHVI` 返回的是
   **log 空间**的值 `log(EHVI) < 0`。原代码写 `log_ehvi * prob`，负数乘一个
   `[0,1]` 的数会**变大** —— 等于给不可行点加分。实测后果很直观：优化器一路奔向
   不可行角点，HV 一直卡在 0.67 不动。改成 log 空间加法
   `log(EHVI) + Σ log(p_i)`，等价于线性空间的 `EHVI × ∏p_i`，同时保留 logEI 的
   数值稳定性。改完 HV 单调上升（0.57 → 0.91），推荐点基本全可行。

   （顺带修了广播：EHVI 是整批 q 个点的联合值，形状 `(b,)`，而可行概率是 `(b, q)`，
   原代码直接相乘在 `b=256` 时会报 `size of tensor a (256) must match ... (2)`
   —— 虽然被 try/except 兜成随机采样了，但那等于没在用 BO。）

3. **`gpytorch==1.11` → `1.12`**。botorch 0.11.3 硬绑定 `gpytorch==1.12`，按原依赖装会
   直接 `ResolutionImpossible`。

测试过的行为：3 个随机种子各跑 5 轮，HV 全部单调上升、无模型拟合失败、无随机采样回退。

## 换成你自己的仿真

只需要改 [bo_skeleton.py](bo_skeleton.py) 里的 `objective_function`：

```python
def objective_function(x_natural):
    """x_natural: (d,) 物理参数 → 返回 (y, feasible)，y 是 (m,)"""
    y = ...            # 例如 (跟踪误差, 控制能耗) 两个目标
    feasible = ...     # 例如 约束满足且仿真未发散
    return y, feasible
```

其余各层不用动：`evaluate` 会自动做归一化 → 物理量的反归一化，
`DataStore` 会自动分流可行/全部样本。

## 关键接口速查

| 位置 | 含义 | 维度 |
| --- | --- | --- |
| `bounds` | 物理参数上下界 | `(2, d)` |
| `normalized_bounds` | 优化器内部归一化空间 `[0,1]^d` | `(2, d)` |
| `ref_point` | 超体积参考点，须被最差目标支配 | `(m,)` |
| `X_obj` / `Y_obj` | 可行样本的决策/目标 | `(n_feas, d)` / `(n_feas, m)` |
| `X_con` / `Y_con` | 全部样本的决策/可行性标签 | `(n_all, d)` / `(n_all, 1)` |
| `select_candidates` | 返回本批推荐点 | `(q, d)` |

## 注意

- `ref_point` 必须比任何观测到的目标都差（对最大化问题即数值更小），否则
  超体积恒为 0、EHVI 失去梯度。若目标全为负值，`[-1, -1]` 这类参考点可用；
  建议按实际量纲设置。
- 目标维度 `m >= 2` 才是多目标。若 `m == 1`，应换用 `qLogExpectedImprovement`。
- `fit_gp` / `select_candidates` 都做了 try/except，失败会回退随机采样，
  保证主循环不会因为数值问题崩掉。
