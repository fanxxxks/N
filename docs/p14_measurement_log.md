# P14 正式测量日志（E4/t84 — 证据记录，随测量推进追加）

**预注册契约**: `docs/p14_search_digest_preregistration.md` — **APPROVED**（用户追认批准 2026-09-04，经 captain 转达；批准记录 commit `6653cf7` @ codex/g0-tushare-pit-contract，t35/t36 评审 pass；该批准记录已 cherry-pick 导入测量分支 `d851adb`）。测量类型 = **promotion/正式**。仲裁顺序：契约 → 本日志；测量结果只执行 §9 预注册裁决。

**被测实现**: 分支 `codex/t23-p14-measurement`，HEAD `91eebbe`（谱系：d851adb [p14 APPROVED 导入 + p12/p13/p14/p17 状态回填 + g0 审计文档] ← 4a3cbab [E3 执行器 cherry-pick] ← df445b4 [t46 run_store 竞态修复] ← 8411811 [E2 因子缓存] ← 4a88d84 [D7 资格门] ← db76700 [D4/D5/D6 全链] ← ac60e74 [t60 非季末修复 cherry-pick] ← e8dcde1 [收窄配置] ← 9a42a87 [t74 as-of 帧优化 cherry-pick] ← 91eebbe [CLI 拆分旗标]）。

**dataset_id 演进链**: `43ef1e4c…`（D3 初版）→ `157e0cf5`（t68 adj_factor 缺口补齐）→ `af825c2e`（t80 v2 市场表增量至 0907）→ **`b7b4dd4b03fef19755814530dfbead040d6fb88137a3fc5f3dc4992526b64377`**（当前，campaign identity 记录）。

## 1. 环境与数据

- OS Windows Server（16GB），Python 3.13.12（D:\minequant\.venv），CPU-only。
- 生产库: `D:/minequant/AlphaGPT/data/ashare.duckdb`（日历/成分；t83 后 daily_bar 至 0907，G1-G8 **8/8 ALL PASS**，data-engineer 执行）。
- v2 库: `D:/minequant/AlphaGPT-d2/data/ashare_v2.duckdb`（57,156,485 行，t80 增量至 0907）。
- 装载窗: start_date 2022-01-01（收窄裁决，见 §5）；window_cap 300×400（fold 0 train_end 20240628）。

## 2. 前置验证

1. **loader 修复（t60）**: 非季末 end_date 行不参与季度累计 TTM；`test_financial_pit + test_fundamentals` 43/43 绿（ac60e74 验证）。
2. **as-of 优化（t74）**: 44×（24.70s→0.56s 单表基准）cherry-pick 9a42a87；**native crash (0xC0000005) 消失**——优化前同路径双现场崩溃/卡顿（我方 25+ CPU-min、factor-engineer 70+ CPU-min），优化后无崩溃。
3. **G8 新鲜度**: 首次冒烟 fail-closed 拒绝（生产库 daily 停 0901 vs 参考 0907，lag=4>3）→ t83 生产库同步（daily_bar 0902-0907 四会话 +22,193 行）→ **G1-G8 8/8 PASS** → 复跑通过（门禁正确行为的完整留证）。

## 3. §9 工程冒烟（budget 64 × 24 行）

- **命令**: `python -m ashare_model.searcher_bench --config configs/p14_measurement.yaml --run-dir experiments/p14_smoke2 --seeds 42,7,2024 --searchers gp,tpe,random,rl --budget 64 --research-budget 40 --promotion-budget 24 --fold 0`
- **结果**: `campaign.json` = **status=completed, rows=24, not_run=0**；ledger 48 trials（24 running-start + 24 succeeded，append-only 完整链）；run_id `c4a3313d667f4f4eaf31707250a25fa8`；产物 `experiments/p14_smoke2/`。
- **行级摘要**（evals/requested/wall_s/per_1000/rss/done）: 全 24 行 done=True；research 轨 40/40 满耗，promotion_tier_a 轨 24/24 满耗（seed 2024 rl research 33/40 提前停滞——如实记录）。
- **冒烟专属偏差**: 拆分 40/24（非 60/40 比例）系 rl_steps=8 整除约束（38/26 触发 "rl row budget not divisible" fail-closed）；正式 campaign 用预注册 1200/800。

## 4. 正式 campaign（budget 2000 = 1200/800 预注册拆分）

- **命令**: `python -m ashare_model.searcher_bench --config configs/p14_measurement.yaml --run-dir experiments/p14_formal --seeds 42,7,2024 --searchers gp,tpe,random,rl --budget 2000 --fold 0 --wall-cap-hours 7`
- **状态**: 启动于 2026-09-08（后台）；产物 `experiments/p14_formal/`；campaign 整体单任务粒度（E3 内部 ledger 行级 resume，已裁定）。
- **结果与 §9 三判据**: 待 campaign 完成后追加（§9 判据只能执行预注册裁决，禁止预写）。

## 5. 全部偏差与裁定（跨四次派发累计）

1. **loader 投影偏差**: p14 §10 "装载 5.6 min" 基于 legacy 生产库规模假设；57M 行 v2 库实测装载+降级审计 ~4 min（收窄后）/ 全量 65+ min（收窄前）——投影口径已过时。
2. **py-spy 误归因修正**: 曾把 factor-engineer 的 factor_inventory_audit 卡顿误记为我方挂起——py-spy dump 归因修正（诊断纪律案例）。
3. **收窄裁决（captain 有条件批准）**: start_date 2015→2022-01-01；条件①p14 无 start_date 钉值（无冲突）②装载窗 ⊇ 张量窗（≈620 会话 ≥ 400+余量）③正式结果解读注明收窄装载窗——三项已入配置注释与本日志。
4. **family-⑤ 性能数据点**: as-of 帧构建在 57M 行库上的三现场证据（双 py-spy + native crash）；t74 优化后本机实测 ~40 min/全量帧构建（1630 码 × 1250 会话，含多表）——单表 0.56s 基准与全量口径差异如实记录。
5. **生产库同步治理**: G8 fail-closed → p18 §3.4 边界确认 → captain 裁决（用户概括授权 + 例行维护解读，透明记录入 t83）→ t83 执行 → G1-G8 8/8。
6. **数据事件**: t68 adj_factor 补齐（20140422，2,528 行）发生于测量运行期间、装载窗外——manifest 演进链已记录，对已装载张量零影响。
7. **688xxx 降级审计**: STAR 板 daily_price_limit 缺失（p19 §2.4 逐股 fail-closed，按设计）——与 t18 发现（1630/1630 degraded）互证，D3 已知 stk_limit 缺口。
8. **CLI 拆分旗标**: 91eebbe 补 --research-budget/--promotion-budget（API 有参数 CLI 未暴露的集成缺口）；正式 campaign 使用预注册默认 1200/800，旗标仅服务冒烟缩放。

## 6. §9 三判据（待 campaign 完成后执行）

- 判据 1: GP research 轨三 seed 中位 consumed/requested ≥ 0.50 → 待填
- 判据 2: tpe/random 全部行 content=11 占比 ≤ 50% → 待填
- 判据 3: promotion Tier-A 轨计费候选 max_tier==A 占比 == 100% → 待填
- 裁决: 待填（允许负结果；不达标即如实记录并停止，禁止重跑换样本）
