# P40 统计重采样修正（真实日期对齐 + 共同日期块抽取）预注册契约（工作包 5A）

- 状态：**APPROVED**，用户 2026-09-19 直接指令「工作包 5A：修正统计重采样。
  范围：eval_corrections.py 及测试。按真实日期对齐收益矩阵，共同抽取日期块；
  预先固定失败折和缺失日期规则。用相关噪声、自相关噪声、重复候选等模拟校准
  拒绝率，不能只用一个随机种子『不显著』验收」（p30–p39 先例；批准记录回写于
  本文件头）。
- 起草：执行模型，分支 `codex/b04-load-cache-performance`。
- 勘察基线：`25f9983`（p39 收口后）。`git status` 仅未跟踪 `.mimosa/`（工具
  状态，不属于本任务）。
- 时序边界（用户原文）：本包与 5B（p41）可在 R02 描述性研究期间开发，
  **必须在正式统计结论之前完成**。

## 1 问题陈述（已实证，`ashare_model/eval_corrections.py` / `eval_metrics.py`）

1. **无日期轴**：行（`evaluate_signal`/`benchmark_row` 产物）只带 `n_dates`
   计数；`stitch_oos_series` 按位置拼接 daily/benchmark/excess，试验之间没有
   可对齐的真实交易日索引。
2. **逐试验独立抽块**：`_max_t_from_pool` 每次置换对**每个试验分别**抽块
   （`for i, s in enumerate(centered)` 内各自 `rng.integers`），破坏了观察统计量
   实际包含的跨试验依赖——所有试验测量在同一些交易日上、共享市场状态。
   独立抽块构造的 max-t 零分布与被检验统计量的依赖结构不对应（共同因子
   载荷高时零分布过宽，拒绝率系统性低于名义水平）。
3. **块长退化反保守**：`block = min(block_size, min(s.size))` 在序列短于
   block_size 时退化为 block == n：循环块抽取变成整段旋转，每次 bootstrap
   均值恒等于中心化均值（0），任何正均值噪声池都得到 p = 1/(n_perms+1)
   ——纯噪声必然显著。
4. **验收薄弱**：现有测试用单随机种子单次噪声抽样断言「不显著」
   （`test_max_t_noise_rows_not_significant` seed=0），未测量任何噪声体制下的
   拒绝率。

## 2 冻结语义（实现必须逐条可对照）

**R1 行携带真实日期**：`evaluate_signal` 与 `benchmark_row` 在行内新增
`daily_dates`（引擎标记日期，`BacktestResult.dates[1:]`，ISO 字符串列表），
与 `daily_returns` 逐元素对齐；同一 fold 上两条路径的 `daily_dates` 必须
一致（同一引擎日期权威）。失败行不受此约束（从不进入拼接）。

**R2 拼接携带日期**：`stitch_oos_series` 按折序拼接每段 `daily_dates`，试验
记录 `dates`；`len(dates) == len(daily_returns) == len(benchmark_daily_returns)
== len(excess_returns)`（断言）。试验内日期重复为完整性错误；日期允许非连续
（折间隙）。成功行缺 `daily_dates` → `ValueError`（fail-closed；错误信息指明
p40 迁移：p40 之前的 artifact 行不得进入拼接/试错池）。

**R3 失败折与缺失日期规则（预先固定）**：
- 失败折：维持 v20 语义——失败折不进拼接、记录于 `failed_folds`；其日期
  从该试验日期索引缺席。
- 裁决窗口 W = **本次运行各试验日期索引的交集**（本次运行的试验按构造都
  覆盖 W；W 为空 → `ValueError`）。
- 联合矩阵 M = 池（本次运行 + 历史试验，拼接彼此独立，维持 v20）中日期索引
  **包含 W** 的试验，逐试验按真实日期对齐到 W（非按位置截断）。
- 不覆盖 W 的历史试验：记 `matrix_excluded`（原因 `window_mismatch`），
  **仍计入 DSR 的 n_trials**（多重检验范围），但不进入 max-t 联合矩阵
  （不得伪造对齐）。
- 行内缺失日期（有 `daily_returns` 而无 `daily_dates`）= R2 完整性错误，
  不是「缺失按零处理」。

**R4 共同块抽取（max-t 重采样核心）**：有效块长 b = max(1, min(block_size,
n//6))（**长度/比率规则**，n = |W|；比率规则的依据是实现期测量：n/b=2 时
拒绝率 0.13–0.24，n/b≥6 时回到 0.054–0.092），联合窗口长度 n ≥
**P40_MIN_WINDOW_DATES = 30**，低于底线 → `ValueError` fail-closed 拒绝
（n=6/b=3 实测拒绝率 ≈0.42，n=12/b=2 ≈0.125：小样本下任何重采样都无法
校准 max-t 95% 分位数，正确行为是不产生 p 值，而非旧实现的必然显著）。
（勘误 2，2026-09-19 实现期：起草时的「b = min(block_size, n//2) 且无
窗口底线」在 n=6 实测 5/12 个固定 seed p=1/(n_perms+1)——对合法实现不可
达；按 §7 停止条件回契约修订。）每次置换在 W 上抽⌈n/b⌉ 个起点**一次**，
同一条日期块索引施加于矩阵**所有试验**（共同抽取，保留跨试验依赖与序列内
自相关）；逐试验用其**W 上观察标准差**学生化（观察统计量与 bootstrap
统计量同在 W 上计算）。p 值 (count+1)/(n_perms+1)、`significant_95`
（p ≤ 0.05）口径不变。DSR 数学（psr/expected_max_sr/deflated_sharpe）
不变（解析式，无最小窗口约束）；n_trials = 池大小（含 matrix_excluded）。

**R5 校准验收（拒绝率，非单种子）**：新增
`tests/test_eval_resampling_calibration.py`（同 commit 注册分片），全部固定
seed；名义 α=0.05、n_perms=200：
- **C1 共同因子相关噪声**（载荷 0.95，K=8，n=250，R=240）：经验拒绝率 ∈
  [0.010, 0.115]。
  （勘误 2026-09-19，实现前测量：旧实现（独立抽块）在该体制下尺寸扭曲温和——
  R=240 诊断测量 0.021–0.063（K=8/16/40、载荷 0.95/0.99），仍落在带内，
  并非起草时预期的「坍缩到带下界之外」；区分旧实现的判据是 N2 的复制
  不变量（精确不变式，旧实现必违）与 C4 的短序列退化（实测 ≈0.93 ≫ 0.18）。
  C1/C2/C3 的角色据此修正为：**新实现（共同抽取）尺寸控制的回归防线**，
  带宽与体制不变。）
- **C2 自相关噪声**（逐试验 AR(1) φ=0.4，K=8，n=250，R=240）：拒绝率 ∈
  [0.010, 0.115]（块抽取对序列内自相关稳健的回归防线）。
- **C3 重复候选**（3 个独立噪声 ×3 份精确复制，K=9，n=250，R=240）：
  拒绝率 ∈ [0.010, 0.115]；另加结构不变量：**精确复制的试验列不得改变
  p 值**（联合抽取下复制列的 bootstrap 统计量逐点相同，max 不变；旧实现
  独立抽块必违）。
- **C4 最小合法窗口反保守防线**（n=30 = P40_MIN_WINDOW_DATES，K=4，b=5，
  R=120）：拒绝率 ≤ 0.18（实现期实测 0.075–0.096；旧实现无底线概念，
  其 n=6 行为 ≈0.93，已由 fail-closed 拒绝结构性消灭——低于底线没有
  p 值可言）。小样本块抽取粗糙，下界不设（披露）。（勘误 2：C4 体制从
  n=6 改为 n=30，n=6 转为 N4 的结构性拒绝断言。）
- **C5 功效**（1 个真信号试验混于 7 个噪声，R=120）：拒绝率 ≥ 0.90
  （防止修正后永不拒绝的保守破坏）。
- 带宽依据：binomial(240, 0.05) 的 99% 双侧区间 ≈ [0.014, 0.086]，向两侧
  放宽至 [0.010, 0.115] 以吸收块 bootstrap 有限样本尺寸畸变（披露的容差，
  远离 0.00 / 1.00 两个退化方向）。校准直接驱动重采样器内部接口（合成矩阵），
  另保留一条经 `max_t_from_rows` 的公共路径测试钉住端到端语义。

## 3 范围与非目标

范围：`ashare_model/eval_corrections.py`、`eval_metrics.py` 的行/拼接日期
携带（R1/R2 是 R3/R4 的必要数据底座，单点修改覆盖全部行产地：trained/
baseline/selfcheck/stress/r01/r02 均经 `evaluate_signal`/`benchmark_row`）、
`evaluation.load_trial_rows` 的 p40 前行拒绝、测试与分片注册。

非目标：DSR 数学与阈值、候选-证据身份绑定（5B/p41：adjudicated 候选、统计
结果身份块、promotion G2 绑定、基准补零改硬失败）、promotion 任何门禁、
R02 面板语义、对任何已存在 artifact 的重写。

## 4 版本影响表

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `PROTOCOL_VERSION` | 27 | **28** | 统计校正语义（max-t 零分布构造）+ 行/拼接 schema（`daily_dates`/`dates`）变更（§3.2 统计检验类） |
| `ARTIFACT_SCHEMA_VERSION` | 3 | 不变 | spine 字段集合不变（dsr/max_t 仍 dict，行为新增键） |
| `PROMOTION_RULE_VERSION` | 2 | 不变 | 门禁消费零变化（p41 处理） |
| 旧 artifact | — | 保留原版本不重写 | v≤27 产物按既有版本分类权威拒绝混接；作为 extras 载入时因无 `daily_dates` 被显式拒绝（错误信息指明 p40） |

## 5 预期 RED（独立 oracle——禁止从实现反推）

`tests/test_p40_statistical_resampling.py`（同 commit 注册分片）：

- **N1** 合成带日期行经 `stitch_oos_series` → 试验 `dates` == 逐折标记日期
  按序拼接；成功行缺 `daily_dates` → `ValueError`。
- **N2** 复制不变量：两个精确复制的噪声试验（同日期轴）的 max-t p 值 ==
  单试验 p 值（同 seed）；旧实现独立抽块必不相等（RED）。
- **N3** 窗口交集与 matrix_excluded：本次运行两试验其一缺中段日期 → W 为
  交集，max-t 在 |W| 上计算（结果记录 n_dates）；日期错开一天的历史试验
  → `matrix_excluded`（`window_mismatch`），且 DSR n_trials 仍计它。
- **N4** 双断言（勘误 2 结构）：n=6 噪声池 → `ValueError`（低于
  P40_MIN_WINDOW_DATES，无 p 值）；n=30 噪声池、60 个固定 seed 的
  不显著率 ≥ 0.75（实测尺寸 ≈0.08–0.13），且有效块长 ≤ max(1, n//6)
  （钉住比率规则与旧退化）。旧实现：n=6 全部 p = 1/(n_perms+1)（RED）。
- **N5** `evaluate_signal`/`benchmark_row` 在同一 fold 上产生一致的
  `daily_dates`，且与 `daily_returns` 等长（合成数据 + populated_db）。
- 校准文件 C1–C5（§2 R5）同属 RED（rev-0 缺 `max_t_from_matrix` 能力，
  逐测试 ImportError）：其中 C4 在旧实现上以实测 ≈0.93 的拒绝率失败；
  C1/C3 为新实现的尺寸防线（勘误见 §2 R5）。

既有测试修改白名单（§10.1 规则 2，本契约即依据）：手工构造行的夹具补
`daily_dates`（语义新增，断言强度不降）；RED commit 只加新文件，夹具更新
随 GREEN commit。

## 6 验证矩阵

1. 聚焦：`python -m pytest -q tests/test_p40_statistical_resampling.py
   tests/test_eval_resampling_calibration.py`
2. 相邻：`tests/test_evaluation.py`、`tests/test_stitched_oos.py`、
   `tests/test_eval_module_split.py`、`tests/test_promotion.py`（版本 bump
   后按 §4 重验）、r01/r02 相关批次。
3. `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio
   ashare_trading scripts webapi`、`python scripts/check_test_shards.py
   --check`、`git diff --check`、全量 `python -m pytest -q tests -n auto`。

## 7 停止条件

- 校准带（C1/C2/C3）在联合抽取实现下不可达 → 停止回契约重推带宽/块长，
  禁止静默放宽。
- `daily_dates` 贯通迫使 §3 非目标内的语义变化（如 promotion 门禁、DSR 数学）
  → 停止修订契约，不夹带。
- 全量回归出现无法归因于本包的新红/新 warning → 停止诊断。
