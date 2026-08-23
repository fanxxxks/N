# AlphaGPT 工程评估报告（2026-08-23）

> 评估对象：`D:\minequant\AlphaGPT` @ main `da13c02`（merge: fix CUDA VM parity test）
> 评估方式：通读全部核心源码（42 个源文件 / 13,085 行，含 ashare_data / ashare_model /
> ashare_trading / webapi / dashboard / scripts），跑通本地全量测试（523 passed），
> 生产门禁 6/6 PASS，并执行一次约 3 小时的正式训练 + 回测（见 §4）。

---

## 1. 项目是什么

AlphaGPT 是一个**纯 A 股横截面多因子发现 + 模拟盘**的研究系统：

- **数据层** `ashare_data/`：AkShare 拉取 → Parquet 缓存 → DuckDB（日线、交易日历、
  股票列表、PIT 成分区间、PIT 财报、两融、申万行业），`universe.py` 构建
  `[stock × date]` 点级（PIT）eligibility 掩码，`check_production_gates.py` 六道
  生产门禁把关。
- **模型层** `ashare_model/`：词表化的因子公式（62 特征 / 40 算子），LoopedTransformer
  policy 按合法后缀式采样公式，`StackVM` 向量化解释执行，奖励 = rank-ICIR − 换手成本
  （REWARD_VERSION=7），REINFORCE + value baseline + 熵正则训练；walk-forward 协议
  （5 折 × 3 种子 + 随机搜索基线 + Deflated Sharpe / max-t 多重检验）做与奖励无关的
  OOS 裁决。
- **交易层** `ashare_trading/`：T+1、整手、涨跌停/停牌、单一口径费用的模拟撮合 +
  断点续跑（水位线 + 原子快照），供回测与模拟盘共用。
- **展示层** `webapi/`（FastAPI）+ `webui/`（React）与 `dashboard/`（Streamlit）。

一句话：**基础设施（数据 PIT 纪律、回测引擎、模拟盘、实验归档、门禁、测试）质量明显
高于研究产出（RL 搜索尚未显著跑赢经典因子）**——详见 §2/§5。

---

## 2. 工程化总体评价（强项）

| 维度 | 评价 | 证据 |
| --- | --- | --- |
| 测试 | **优秀**。33 个测试文件 / 10,194 行，测试:源码 ≈ 0.78，523 用例 | 全模块有专属测试；含"未来成员哨兵"全链路契约测试、resume==全量跑黄金测试、CPU/GPU 一致性、确定性种子 |
| PIT 纪律 | **优秀**，贯穿全链路 | 所有正式路径强制传 `universe_mask`，无 None 旁路；IC/收益/基准/截面算子全部只在 signal-date eligible 单元上统计 |
| 版本与溯源 | **优秀** | REWARD_VERSION / PROTOCOL_VERSION / feature_version；按名重映射词表；实验归档 manifest 含 git commit + 脏工作区 + SHA-256 |
| 可复现性 | **优秀** | `(init_seed, seed)` 固定权重与采样；policy 恒在 CPU 保证跨机器 RNG 一致（有测试固化） |
| 分支/提交纪律 | 好 | 99 commits，feat/fix/chore/experiment/merge 规范前缀，长期 feature-branch + merge 工作流 |
| CI | 基本可用 | `.github/workflows/ci.yml` 跑 pytest（Python 3.12，无 GPU）；但本地 CUDA 机器曾跑挂（已修，见 §3-1） |
| 文档 | 良好 | README 297 行非常详细；但存在与代码漂移的细节（见 §3-3） |
| 单一事实源 | 良好 | 费用模型、基准收益、IC 实现、排序/打分均单一口径复用 |

## 3. 发现的问题（按严重度）

### 3.1 高严重度（正确性）

- **H1 — `JUMP` 算子非因果（向前看），且污染 walk-forward 协议。**
  `ops.py:116-120` 用整条时间轴 `x.mean(dim=1)/x.std(dim=1)` 标准化，t 时刻的取值
  依赖未来数据；更严重的是 `evaluation.py:344-350` 先在**全量 2823 日**张量上执行
  VM 再切片测试窗，含 JUMP 的公式在测试期信号会看到测试期之后的数据。
  已用最小实验复现（未来列扰动改变 t=0 的 z 值）。与 README "无未来泄漏" 声明矛盾。
- **H2 — `_rolling_capm` 的市场序列未按个股停牌有效性对齐。**
  `factors.py:200-236`：`pm/pm2` 前缀和覆盖窗口内**所有**交易日，而 `n` 只统计个股
  自身有效日。对存在停牌缺口的股票，`BETA_60/IVOL_60/RSQ_60` 的 `cov/var_m` 把
  市场收益的窗口长度与个股分母混用，产生偏误。现有 CAPM 测试只用连续序列，未覆盖。
- **H3 — 日线同步股票池存在幸存者偏差（数据缺口而非代码 bug）。**
  `sync.py:185-218` 用**当前**成分快照决定同步哪些股票，退市/历史成员无日线，
  `build_universe_mask` 将其标记为 MISSING_BAR 静默排除——所有历史回测偏乐观。
  README 已承认"快照非 PIT"，但没有补 bar 的机制。
- **H4 — `sync_all` 会抹掉 PIT 上市日期回填。**
  `db.py:173-185` `upsert_stocks` 是 DELETE-all + INSERT 当前快照，`sync.py:181-183`
  每次全量同步无条件执行，会删除 `import_pit_universe.py` 回填的退市股票，破坏
  生产契约，直到重跑 import 脚本。常规操作间存在静默数据丢失。
- **H5 — 回测引擎对"亏光"账户崩溃（本次会话已修复）。**
  `backtest.py:360` 对 total_return < -1（成本叠加在 -100% 毛收益上，复合净值转负）
  计算 `(1+total_return)**(252/n)` 产生复数，`float()` 抛 TypeError，整个回测入口
  崩溃——本次训练出的 LIMIT_BREAK 反向策略当场触发。已修：clamp total_return ≥ -1、
  drawdown ∈ [0,1]，4 个回归测试（`fix/backtest-bust-metrics`）。

### 3.2 中严重度

- **M1 — `vm.execute` 裸吞一切异常（`except Exception: return None`，`vm.py:142`）。**
  设备不匹配、OOM、编程错误都退化成"公式非法"，无法区分；本次本地测试失败
  （CUDA 掩码未随张量移动）正是被它掩盖成 None。
- **M2 — 验证尾参与策略梯度。** `train.py:490` 用 `full_window_reward`（含验证尾，
  `reward.py:557-560`）做 REINFORCE 奖励；验证尾只从**公式选择**中隔离，未从
  **策略学习信号**中隔离。最终 OOS 裁决靠协议测试折兜底，但训练内乐观偏差存在。
- **M3 — IC 项不屏蔽不可交易股票。** `reward.py:560-567` 的 `rank_ic_series` 只吃
  `universe_mask`，`blocked_buy/sell` 只约束篮子模拟；涨跌停/停牌个股仍能贡献 IC。
- **M4 — 历史 ST 股按错误价幅回放。** 历史回放/回测只按板块价幅（10%/20%），
  历史 ST 5% 价幅数据缺失（README 已声明）；`processor.py:91-108`。
- **M5 — 行业成分为当前快照外推到历史。** `factors.py:96-98`：`IND_REL_*` 与
  `CS_NEUTRALIZE` 用的申万成分无 PIT（README 已声明）。
- **M6 — `webapi` 控制端点无鉴权。** `/api/sim/start|stop|reset`、`PUT /api/sim/config`
  无任何认证，且 `app.py:8-10` 生产文档写 `--host 0.0.0.0`；局域网内可停盘/重置/
  改费用配置。目前唯一缓解是默认绑 127.0.0.1。
- **M7 — `PS_TTM` 是 PE×净利率近似而非真实市销率**（`fundamentals.py:345-362`），
  名称有误导性（README 已注明口径）。
- **M8 — `import_pit_universe.py` O(n²) 重建**（`:301-323` 逐行 `pd.concat` +
  循环内集合查询）；该脚本**零测试覆盖**。
- **M9 — Dashboard 读取坏产物会崩**（`dashboard/app.py:32-40`、`data_service.py:12-22`
  裸 `json.loads`），而 webapi 层对同一读取做了防御性包装——两套标准。
- **M10 — 默认训练配置在本机不可用。** `model.batch_size: 4096` 在 torch 2.11.0+cu128 /
  py3.13 / Windows 下 step 0 两次以 `c10.dll` 0xC0000005 崩溃（事件日志同偏移），
  batch 256 同代码路径稳定通过；且每步成本随 batch 近似线性（256 → 64 s/步），
  1000 步 × 4096 的默认值既跑不动也不现实。默认值或文档需与实际可运行规模对齐。
- **M11 — 打分分块的内存预算只算了 1×（本次会话已修复）。**
  `candidates.score_many` 瞬时同时持有原始信号栈 + 批量拷贝 + 双向 interleave 拷贝
  ≈ 4× 单信号；训练侧 `_reward_chunk_size` 与协议随机搜索（固定 `chunk=16`）都按
  1× 预算。本次协议首次运行在 fold 5 随机搜索分配 846 MiB 时 MemoryError 崩掉
  （机器 commit 上限吃紧）。已修：`candidates.score_chunk_size` 统一按 4× 预算
  （`fix/reward-chunk-memory-budget`，含预算性质测试）。

### 3.3 低严重度 / 代码卫生

- 死代码：`data_loader.date_index`（`:36-50`，docstring 声称"单一代码路径"但无生产
  调用）、`train._train_end_index`、`train.best_reward` 别名、`schemas.DailyBar/
  StockMeta/FactorFrame/PortfolioPosition`、`processor.long_factor_frame`、
  `dashboard.visualizer.factor_bar_figure`；`train.py --offline` 参数声明后从未使用。
- 重复实现：原子写 JSON 4 处（portfolio/manager/run_sim/service）、STOP 信号 3 处、
  `_validate_eligible` 2 处（ops/diagnostics）、日期归一化 2 套（universe/akshare）、
  SQL IN-list 手拼 4 处；`rank_ic_stats` 重造 ICIR；两个方向判定器并存。
- 手造轮子：Acklam `norm_ppf`+Halley、`_rankdata`、`_pearson`（scipy 非依赖的理由成立，
  但属维护负担）。
- `assert` 参与控制流（`candidates.py:300`，`python -O` 下消失）；`_op_jump` 用
  样本标准差 vs 其余总体标准差；`sortino` 两分支 eps 不一致；`icir` 单日 NaN 写入 JSON
  （`diagnostics.py:134`）。
- 文档漂移：`ashare_config.yaml` 注释仍写 "v4 semantics"（实际 REWARD_VERSION=7）；
  README 写 complexity_penalty 0.2 / validation_splits 3，实际 0.02 / 4；
  README "margin 次日可用" 与实现同日的措辞出入。
- 性能小项：`_cs_neutralize` 逐行业 Python 循环；`train._update_stack_state` 每算子
  重建 arity 张量；`factors.py:133` DataFrame 碎片化 PerformanceWarning；
  采样循环 `torch.cat` 逐步拼接 O(t²)。
- `ashare_logging` 内存日志无上限；`service.load_stock_names` 永久缓存无失效。
- 网络层：`_call_with_timeout` 超时线程泄漏（已注释承认）；`load_config` 缺文件时
  静默返回 `{}` 而非报错。

## 4. 本次 3 小时训练 + 回测

- 环境：RTX 2060 6GB，torch 2.11.0+cu128，Python 3.13，16GB RAM；
  因子张量 [62, 1630, 2823] float32 ≈ 1.1GB，训练切片（至 2023-12-31）≈ 0.9GB。
- **实测节奏**：数据装载 ≈ 5–6 min；batch 256 下每步 ≈ 64 s（其中 VM 逐公式
  串行执行 + 分块奖励打分占绝对大头，成本随 batch 内唯一公式数近似线性增长），
  因此配置默认 1000 步 × 4096 在本机**不现实**。
- **新发现（运行期）**：batch 4096 的训练在 step 0 两次以 `c10.dll` 0xC0000005
  崩溃（Windows 事件日志证实，同一偏移），batch 256 同代码路径稳定通过；
  torch 2.11.0+cu128 / Windows / py3.13 组合疑似存在与批量相关的间歇性崩溃，
  而 `ashare_config.yaml` 的默认 batch_size 正是 4096 —— 默认配置在本机不可用。
  本次按协议 screening 档规模（150 步 × 256）执行 ≈ 2.5–3 h，与配置中
  `protocol.screening` 完全一致。
- 产物：`data/best_ashare_strategy.json` + `ashare_model.pt`，随后
  `python -m ashare_model.backtest` 全区间回测（2015-01-01 → 数据末端），
  `scripts/archive_run.py --mode backtest --commit` 归档。

<!-- TRAINING_RESULT -->

**单次训练结果（screening 档 150 步 × 256，seed=42，train_end 2023-12-31）**

- 步进耗时：前 2 步 ~64 s/步（CUDA 冷启动 + 首次全量打分），随后 ~1.3 s/步，
  全程 5.4 min；策略在 ~50 步后明显收敛（unique_frac 0.18 → 0.012，熵 0.004），
  说明 150 步之后基本是缓存命中，训练时长再长收益递减。
- 选出公式：裸因子 `LIMIT_BREAK`（token 58，无算子），direction=-1，
  val_reward 0.369 / val_icir 0.408（验证尾中位数）——但 full_window_icir = **-0.013**：
  验证尾选出的"好"公式在全窗口上是负质量，暴露 M2（验证尾参与梯度）与选择
  过度适配尾部的组合风险。
- **全区间回测（2015-01 → 2026-08）**：该公式直接亏光（total_return -100%，
  sharpe -1.96，max_drawdown 100%）——并当场触发 `_metrics` 的复数年化崩溃
  （已修复：`fix/backtest-bust-metrics`，clamp total_return ≥ -1 + drawdown ∈ [0,1]，
  4 个新回归测试）。这同时是一个方法论警示：**验证尾单窗口不足以代表全窗口**。

**协议级运行（screening，5 折 × 3 种子 + 4096 随机搜索 + 7 基线 + DS/max-t）**

- 首次运行 1.27h 后在 fold 5 随机搜索因 M11 内存预算问题崩掉（846 MiB 分配失败）；
  修复后重跑 1.8h 完整通过——fold 5 随机搜索正是首次崩溃点，修复得到端到端验证。
- 结果（60 试错）：top trial = `benchmark:equal_weight`（2025 测试年 sharpe 1.717）；
  **DSR = 0.043**（n=60，最佳非基准候选 `baseline:RSQ_60` sr=0.044）、
  **max-t p = 1.0000**——没有任何候选显著。
- trained 聚合（15 次训练）：sharpe 中位数 0.032（q25 -0.03 / q75 0.35）、
  excess_return 中位数 **-8.7%**（系统性跑输等权基准）、ic_mean 中位数 0.026。
  结论：**RL 训练公式的 OOS 表现 ≈ 随机水平且跑输基准；最佳"发现"仍是已知基线
  因子（RSQ_60），与既往 6 次协议筛选的结论一致。**
- 全程观察：每折 3 个种子的训练都在 ~50 步内策略坍缩（unique_frac < 0.1，
  熵 < 0.01），选出的公式几乎全是裸因子（ROE / LIMIT_UP_CNT_20 / TURNOVER_STD20 /
  LIMIT_BREAK）——RL 搜索实际在"退化到抄底基线因子"，并未合成新结构。

## 5. 改进方案（按优先级，与 §3 对应）

> 本次会话已按"分支 → 测试 → 验证 → 合并"流程落地 3 个小修复：
> `fix/vm-cuda-parity-test`（CUDA 测试掩码设备）、`fix/backtest-bust-metrics`
> （H5）、`fix/reward-chunk-memory-budget`（M11）。P0 中的语义性修复（H1/H2/H3/H4/M1）
> 需要各自分支 + 回归测试 + bump 版本号 + 协议重跑，未在本会话实施。

**P0 — 先堵正确性漏洞（改动语义，需各自分支 + 测试 + bump 版本号）**

1. **H1 JUMP 因果化**：改为仅用当前及历史数据的展开窗口（如 t≤窗口期用 expanding、
   之后用 trailing 窗口标准化），或直接退役该算子；为 `evaluate_formula` 增加
   "测试窗执行与全量执行结果一致"的因果回归测试；若语义变更，bump REWARD_VERSION
   并重跑 `--selfcheck` 与协议确认档。
2. **H2 CAPM 对齐**：市场前缀和按个股 `valid` 掩码逐窗口对齐（`pm_valid =
   prefix(m0*valid)`），新增含停牌缺口的回归测试（已知 beta 恢复）。
3. **H3/H4 数据管线**：把历史成分 + 退市股票的日线纳入同步清单（BaoStock 免费历史
   bar 可作为 PIT 区间股票的回填源）；`upsert_stocks` 改为 UPSERT（保留退市行），
  让 `sync_all` 幂等且不再破坏契约；新增"sync 后契约仍成立"的集成测试。
4. **M1 异常可见性**：`vm.execute` 只把"结构性非法"（token/arity/形状）返回 None，
   其余异常上抛（或记录），并补设备不匹配即报错的测试（本次修复的测试就是此事的
   回归锚点）。

**P1 — 研究有效性**

5. **M2 验证尾隔离**：策略梯度奖励改用训练段（不含验证尾）的 ICIR，或对尾段权重
   降权；同时保持选择逻辑不变。
6. **M3 IC 可交易性**：`rank_ic_series` 增加 blocked 掩码参数，训练 IC 与篮子口径
   对齐（需评估对现有结果的影响后 bump 版本）。
7. **M4/M5 数据补充**：历史 ST 状态表（退市表 + 公告解析）与申万 PIT 成分；在数据
   就绪前维持现有声明即可。
8. **模拟盘 H1 类遗漏**：`run_sim.py` 持仓不在 `ts_codes` 时的处置路径（当前永久
   滞留）+ 补 `webapi/app.py` 的 HTTP 路由层测试。
9. **M10 训练可运行性**：定位/规避 batch 4096 的 c10.dll 崩溃（最小复现脚本 +
   逐步二分；必要时把默认 batch 降到 256/512，或 pin torch 版本），并加一个
   "大 batch 冒烟" CI/门禁用例。

**P2 — 工程质量（不改变语义，低风险，适合日常迭代）**

10. 清理死代码清单（§3.3 全部条目），并为每个删除配套测试更新。
11. 收敛重复实现：统一 `atomic_write_json` / STOP 协议 / 日期归一化 / SQL 参数绑定。
12. `webapi` 加 token 鉴权（或至少默认拒绝非 127.0.0.1 来源）；README 生产段去掉
    0.0.0.0 引导。
13. Dashboard 读取复用 webapi 的防御性读取；日志环形缓冲；`load_config` 缺文件报错。
14. 补 `import_pit_universe.py` 的单元测试（当前零覆盖）+ 修复其 O(n²)。
15. 修复文档漂移（yaml v4 注释、README 默认值），建立"改配置必同步 README"的
    约定或生成式文档。
16. 可选：scipy 加入依赖以删掉手写 `norm_ppf/_rankdata/_pearson`（若体积可接受）。

**P3 — 研究路线**

17. 本次完整协议筛选（60 试错）再次确认：DSR 0.043 / max-t p 1.0，top trial 是
    基准本身，最佳非基准候选是经典基线 `RSQ_60`——**RL 搜索尚未产生可辩护的
    alpha**。建议下一步不是加大训练时长（策略 ~50 步就坍缩，训练再长无收益），
    而是：a) 修完 P0 因果问题（H1/H2）后重跑协议基线，排除"被污染因子拖累"的
    可能；b) 用 `ablate_families.py` 找出真正出力的因子族，收缩搜索空间；
    c) 把随机搜索基线当作下界，先让 RL 稳定 ≥ 随机搜索再谈升档；
    d) 处理策略坍缩（熵系数/温度/重启），否则 RL 只是在退化抄底基线因子。
