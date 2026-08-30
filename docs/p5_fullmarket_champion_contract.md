# P5 全市场扩池与 Champion 双验证契约

状态：预注册（实现前）
适用版本：`PROTOCOL_VERSION = 23`、`REWARD_VERSION = 14`、
`GRAMMAR_VERSION = 2`、`MODEL_VERSION = 3`、execution 2、
portfolio constructor 1、`DATA_TIER_VERSION = 1`、`SEARCH_CONTRACT_VERSION = 1`、
`ADMISSION_RULE_VERSION = 2`。

本文是 P5 测试断言与验收的来源。仲裁顺序为本文/需求、测试、实现；测量结果
不得反向改写本契约。实施完成后按惯例补 `docs/p5_measurement_log.md`，
测量日志只记录事实，不放宽本文任何裁决规则。

## 0. 目标、终审标准与停止条件

目标（用户已确认）：产出一个**真实可上实盘**的 A 股策略 champion，前提是
**回测与模拟盘双验证都跑通**。约束：只用免费数据（AkShare/BaoStock/交易所
公开接口）；算力上限为本机（6GB VRAM / 16GB RAM）。

终审标准（全部同时成立才算达成）：

1. 全市场池上运行 protocol v23 confirmation 档，`--trials` 合并同一
   dataset_id 下的全部历史 trial 后，**DSR > 0.95 且 max-t p ≤ 0.05**；
2. `ashare_model.promotion` 六道门禁（数据与公式 P0、统计显著性、超额收益
   与风险、成本/容量压力网格、data_tier、纸面观察窗）全部通过，data_tier
   默认 Tier A only；
3. champion 公式进入模拟盘，完成至少一个**预注册长度 ≥ 40 个交易日**的
   未来纸面观察窗（`data/paper_windows.json`），且模拟盘净值与回测权重经
   `ashare_portfolio/golden.py` 重放的差异在既有可解释残差口径内。

停止条件（任一成立即停，结论如实写入测量日志，不降级统计门槛硬上）：

- **S1**：两个独立配置臂（见 §6，daily 与 weekly 各算一轮）的 confirmation
  均 DSR ≤ 0.95 或 max-t p > 0.05 → 诚实接受"免费日频/周频数据 + 本机
  算力此刻未发现可实盘 alpha"，项目定位回到研究工具；
- **S2**：全市场区间构建后零 bar 区间占比 > 2% 且经回填仍无法清零 →
  回退到 800 池路线（扩池决策作废，A/B 证据照实记录）；
- **S3**：数据回填因接口限制连续 3 个后台运行日无实质推进 → 暂停并回报
  用户，不绕过速率限制、不换付费源。

Non-goals：券商实盘接口（最后一公里，本期不做）；付费数据；RL/PPO/
imitation 任何工作（P4 已预注册否决）；Streamlit/React UI 新功能；
多用户化；改动 reward v14 / protocol v23 语义。

## 1. 范围与不变量

P5 改数据层 universe 语义（新增全市场成员来源）、配套同步与内存守卫，
并执行大规模测量。下列不变量在改动前后必须成立：

1. 生产默认不变：`config/ashare_config.yaml` 的 `index_codes` 保持
   沪深300+中证500，`model.searcher: gp` 不变；全市场只通过**独立的
   专用配置文件**启用（§3.6），默认链路行为逐字节不受影响。
2. `AshareDataLoader` 仍是逐日资格掩码的唯一构建者；不新建第二套
   mask/eligibility 代码路径。全市场通过既有 `constituents` 区间机制
   表达（§3.1），G1–G7、reason codes、`UniverseMask` 语义全部复用。
3. 统计门禁只紧不松：DSR 0.95 / max-t 0.05 / 六道晋级门 / Tier A 准入
   全部维持；任何"为了出 champion 而调参"的行为违反本契约。
4. 旧产物维持 legacy 盖章；P5 新产物必须携带完整 provenance
   （dataset_id、universe policy、protocol/reward/execution/constructor
   版本、配置 hash）。
5. 试验诚实性：每一个 protocol trial（含失败、崩溃、两个频率臂）都进入
   `data/experiment_ledger.jsonl`；confirmation 的 `--trials` 只合并与
   本次 campaign **相同 dataset_id** 的 trial；其它 dataset 的历史 trial
   只作为历史证据引用，不并入校正。

## 2. 路线总览（用户已批准"三线并行"）

- **线 0（数据/工程，纯后台优先启动）**：全市场成员区间构建 + 全历史
  日线回填 + G1–G7 适配 + 内存守卫。
- **线 1（基线测量，先于线 0 的数据落库完成）**：当前代栈（v23/v14）
  在 800 池上跑 screening 基线并归档（同时是线 2 的 A/B 对照）；
  裸因子 daily/weekly 四象限预检验。
- **线 2（主战场）**：全市场 screening（daily + weekly 双臂）→ 与 800
  池基线做 A/B → GP 多 seed campaign → confirmation 裁决。
- **终点**：有 champion → 晋级六门 → 模拟盘重置并启动纸面观察窗；
  无 champion → 按 S1 诚实收尾。

时序约束：线 1 的 800 池基线必须在 FULL.CN 区间写入 constituents 表
**之前**完成并归档（dataset_id 随 constituents 内容变化，基线必须锚定
当前 dataset_id `b927074a455a…`；A/B 跨越两个 dataset_id 的事实写入
测量日志）。线 0 的代码（不落库的纯函数与测试）可与线 1 并行。

## 3. D1 全市场 universe 契约

### 3.1 表达方式：合成指数 `FULL.CN`

全市场成员语义 = "已上市且未退市"。实现为 `constituents` 表中
`index_code = 'FULL.CN'` 的合成成员区间，每只股票一个区间
`[in_date, out_date)`：

- `in_date` = `stocks.list_date`（真实上市日，交易所批量资料口径）；
- `out_date` = 交易所退市表的退市日；退市日缺失且股票当前已退市时，
  回退为 `最后 bar 日期 + 1 日`；当前在市股票为开区间（out_date NULL）。
  每个区间的 `out_date` 来源必须在构建报告中逐股记录
  （`out_date_source: exchange | last_bar | open`）。
- "当前已退市但无退市日"的判定宽限：最后 bar 日期距数据窗口末端
  > 45 个自然日才允许用 `last_bar` 回退；否则视为在市开区间（长期停牌
  股由回填的平盘零量 bar 覆盖，last bar 会贴近窗口末端，不会误截）。

区间合法性沿用既有规则：半开、同 `(index_code, ts_code)` 不重叠、
主键 `(index_code, ts_code, in_date)`；2015 年前上市的股票保留真实
`in_date`，G7/会话统计按既有"日线数据窗口封顶"规则处理。
2015 年前已退市、无任何窗口内 bar 的股票不产生区间（对研究无意义）。

### 3.2 超集不变量

在任一共同日期上，FULL.CN 资格掩码是 300/500 掩码的**超集**：同一只
股票在 800 池 eligible 的 `(stock, date)` 单元，在全市场配置下必须同样
eligible（同样的上市年龄、bar presence、reason code 规则）。预注册测试
固化该不变量（§8）。

### 3.3 零 bar 政策

构建时逐区间审计窗口内 bar 数：窗口内零 bar 的区间**不入库**，计入构建
报告（`dropped_zero_bar_intervals`，含逐股清单）；回填后复审计，正式运行
前 G7 必须显示 0 个零 bar 区间。丢弃占比 > 2% 触发停止条件 S2。

### 3.4 无未来泄漏

上市日/退市日只决定区间端点；`MISSING_BAR`/`NOT_YET_LISTED`/
`LISTING_AGE_INSUFFICIENT` 等 reason code 语义不变。上市满 60 个交易
会话仍按交易会话计算。未来上市哨兵股测试（既有 `test_universe.py` 的
未来成员契约）必须扩展到全市场配置：未来成员 F 在上市前的极端行情
对任何历史结果零影响。

### 3.5 数据层唯一新增代码路径

`ashare_data.sync` 目前对每个配置的 index_code 拉取"当前成分快照"做
并集与防伪校验；`FULL.CN` 没有外部成分快照源。允许的**唯一**适配：
sync/akshare_client 把 `FULL.CN` 识别为无外部快照源的合成代码，快照
步骤跳过该代码（日线并集已由 constituents 表中的 FULL.CN 历史区间
覆盖全部股票，完整性不受损）。除此处之外不得为全市场新增任何特例
分支；该跳过逻辑必须有测试（伪造一个未知代码必须仍报错，只有
`FULL.CN` 被豁免）。

### 3.6 专用配置

新增 `config/ashare_config_fullmarket.yaml`：完整复制基线 YAML，仅改
`index_codes: ["FULL.CN"]`、`index_names: ["全市场"]`。全市场的一切正式
运行（训练/协议/诊断/回测/模拟/归档）显式传 `--config
config/ashare_config_fullmarket.yaml`。协议/回测产物经既有 universe
policy 字段（`index_codes`）自然记录 `FULL.CN`，**不需要** bump
`PROTOCOL_VERSION` 或新增 artifact 版本；该决策即本契约条款。

### 3.7 基本面/资金/行业数据

`fundamental_pit` 范围过滤、两融、申万行业快照的同步并集均派生自
"stocks ∪ PIT 成分 ∪ 日线缓存"，FULL.CN 入库后自动覆盖全市场，无需
新代码；小盘股基本面覆盖率由既有 coverage 门禁（≥0.2）和诊断报告
如实反映，不为覆盖率修改任何门禁。

## 4. D2 数据回填契约

1. 范围：`stocks` 表中全部股票在 `[2015-01-01, 窗口末端]` 的日线，
   对本地无缓存的代码（估计约 3000 只）做一次性全历史回填；退市股
   为静态数据，回填一次后不再重复请求。
2. 复用 `scripts/backfill_member_bars.py` 的兜底链（东财→新浪→腾讯→
   BaoStock）与"停牌行=平盘+零成交量"口径；不得引入新的数据源。
3. 回填必须断点续跑安全（per-code Parquet 缓存即水位线），进度写日志；
   因接口限速，允许跨多个后台运行日完成。
4. 回填完成后跑一次全量 `ashare_data.sync` 刷新 dataset manifest；
   新 dataset_id 写入测量日志。
5. 回填期间**禁止**在任何正式入口使用全市场配置（G7 不可能通过，
   门禁本身会拒绝；不得用 `--dev` 绕过）。

## 5. D3 内存与窗口契约

1. **全截面不变量**：全市场正式运行的每个日期截面必须包含当日全部
   eligible 股票；禁止任何会截断股票轴的 window cap（如 300 只）进入
   正式全市场结果。截面算子（CS_RANK/CS_ZSCORE/CS_NEUTRALIZE）与
   top-N 语义以全截面为准。
2. **日期轴预算**：全市场因子张量（62 × ~5000 × 日期 × float32）必须
   按 fold/训练窗裁剪日期轴加载；GPU 上张量驻留目标 ≤ 1GB（约
   4500 股 × 400 日），CPU 运行亦须显式日期窗，禁止全历史无裁剪加载。
3. **loader 守卫**：`AshareDataLoader` 在全市场配置 + 未设置日期窗/
   window cap 时必须 fail-fast，报错信息指明该契约条款；守卫有测试。
4. 若现有 per-fold 加载已天然满足日期窗语义，守卫按现状校准并在测量
   日志记录结论；不得借守卫之名改 fold 语义。

## 6. D4 测量与裁决契约

### 6.1 线 1：800 池基线（当前 dataset_id）

- 命令：`python -m ashare_model.evaluation --tier screening`
  （默认配置，seeds 42/7/2024，GP+TPE+Random+7 裸因子 baseline）。
- 产物归档：`python scripts/archive_run.py --mode protocol --commit`。
- 记录：DSR、max-t p、trained/TPE/Random/裸因子的拼接 OOS 中位
  active IR、top-trial 身份。这是 v23/v14 代栈首个完整基线，同时是
  线 2 的 A/B 对照。
- 预检验：`python -m ashare_model.bare_factor_backtest`（既有四象限
  daily/weekly × equal_weight/optimizer），逐因子对比 daily 与 weekly
  的净超额与换手。预检验结果只作为方向性证据写入测量日志，
  **不构成任何门禁**；weekly 臂无论如何都在线 2 运行（§6.3）。

### 6.2 线 2 A/B：全市场 vs 800 池

全市场 screening（daily 臂，与 §6.1 同 seeds、同 tier）完成后，预注册
对比指标（跨 dataset_id 的事实已记录，仅作程序内对照）：

- (a) trained 候选拼接 OOS 的中位 active IR；
- (b) top-trial DSR。

若全市场在 (a)(b) **同时严格更差** → 暂停 campaign，携带证据回报用户
再决定是否继续烧 campaign 预算；其余情形按本契约继续。

### 6.3 GP campaign（全市场，daily 与 weekly 双频率臂）

- 频率臂：`daily/horizon=1` 与 `weekly/horizon=1`（合法非重叠组合，
  P3 调仓日历机制已支持；weekly 臂用专用配置复制件改
  `protocol.frequency: weekly`）。
- seed 集：默认 3 个（42/7/2024）+ campaign 扩展 5 个
  （1337/999/5150/31415/271828）= 8 个；每个 (searcher, seed) 的拼接
  OOS 序列是一个 trial。screening 档 150×256。
- 基线随行：TPE、Random 与 7 裸因子在同 folds/seeds 同预算跑齐
  （"什么水平算好"的标尺），预算估计约 3–4 个机夜；不足时先砍 TPE
  臂并在测量日志说明，GP 与 Random 不可砍。
- 全部 trial（含两臂、失败与崩溃）入 ledger。

### 6.4 Confirmation 裁决

- 对两臂中 screening 拼接 OOS 中位 active IR 更高的一臂先跑
  confirmation（200×512），`--trials` 合并本 dataset_id 全部 screening
  trial；另一臂随后同样确认（两轮独立配置对应停止条件 S1）。
- 裁决：`DSR > 0.95 且 max-t p ≤ 0.05` 才存在 champion 候选；候选公式
  = 该臂内拼接 OOS 最优且通过候选质量/复杂度/容量既有门禁者。
- 裁决只消费完整回测引擎指标与 rank IC/ICIR；`best_reward`/
  `val_reward` 只归档不参与排序（既有规则，重申）。

## 7. D5 Champion 与模拟盘纸面观察窗契约

1. 晋级：`python -m ashare_model.promotion --artifact <confirmation 产物>
   --config config/ashare_config_fullmarket.yaml`，六道门全部通过，
   data_tier 默认 A-only；Tier B 对照如需另跑，必须显式
   `--allow-tier-b` 且单独记录，不与 A 档结果混排。
2. 模拟盘切换（champion 存在时才执行，且执行前必须经用户确认，
   因为涉及旧状态处置）：先经 manager 的 reset 流程自动归档旧 legacy
   状态（`LIMIT_BREAK` 时期），再以全市场配置 + champion 公式
   `--reset` 启动，日常经 `--resume` 续跑。
3. 纸面观察窗：在 `data/paper_windows.json` 注册窗口（起点=champion
   模拟盘首个执行日，长度 ≥ 40 个交易日）；窗口注册前必须先读
   `promotion.py` 对该文件的解析契约，字段不一致以代码为准并在测量
   日志记录。
4. 一致性：窗口内每周用 `ashare_portfolio/golden.py` 口径核对模拟盘
   成交与回测权重残差；`scripts/analyze_sim.py` 的费用拖累/毛盈亏/
   现金核对作为常驻体检。
5. 日常运营手册（写入测量日志附录）：每交易日收盘后
   `ashare_data.sync`（全市场配置）→ `run_sim --resume`；信号日
   t → t+1 开盘执行的既有因果不变。

## 8. 预注册失败测试清单（首个代码提交即写入，先 RED 后 GREEN）

代码任务遵循项目惯例：测试先行。以下测试名与断言语义是本契约的一
部分（实现细节以代码为准，断言语义不得弱化）：

1. `tests/test_fullmarket_universe.py::test_intervals_half_open_and_sourced`
   — 构建区间半开、不重叠；`out_date_source ∈ {exchange,last_bar,open}`
   且 `last_bar` 回退只在距窗口末端 >45 天时触发。
2. `tests/test_fullmarket_universe.py::test_fullmarket_is_superset_of_index_mask`
   — §3.2 超集不变量（共用 fixtures，同一 `(stock,date)` 单元比对）。
3. `tests/test_fullmarket_universe.py::test_future_member_sentinel_fullmarket`
   — §3.4 未来成员哨兵：F 上市前一切结果与删除 F 完全一致。
4. `tests/test_fullmarket_universe.py::test_zero_bar_intervals_dropped_and_reported`
   — §3.3 丢弃与构建报告字段。
5. `tests/test_fullmarket_universe.py::test_delist_fallback_uses_last_bar_only_when_stale`
   — §3.1 的 45 天宽限与开区间判定。
6. `tests/test_sync.py::test_fullmarket_snapshot_skipped_but_unknown_index_rejected`
   — §3.5 唯一豁免路径；未知代码仍报错。
7. `tests/test_data_loader.py::test_fullmarket_requires_date_window_guard`
   — §5.3 loader 守卫 fail-fast 与报错文案。
8. `tests/test_gates.py::test_gates_pass_on_fullmarket_fixture`
   — G1–G7 在全市场 fixtures 上通过（含 G7 零 bar=0）。
9. `tests/test_artifact_versions.py` 现有版本不变量回归 — 本契约
   不 bump 任何既有版本常量。

## 9. 版本与迁移策略

- 不 bump：`PROTOCOL_VERSION`、`REWARD_VERSION`、`GRAMMAR_VERSION`、
  `MODEL_VERSION`、execution、constructor、`DATA_TIER_VERSION`。
- `constituents` schema 不变（只新增 `FULL.CN` 行）；旧 300/500 行
  不动；两个 universe 共存于同一 DuckDB，由配置选择。
- dataset_id 将因 constituents/日线内容变化而更新；新旧 dataset_id
  及其对应研究阶段在测量日志中一一登记。
- 旧策略/回测/协议/模拟产物维持 legacy 盖章与消费端防护，不重训、
  不冒充、不删除。
- 若实施中发现本契约与现实冲突（如 `paper_windows.json` 字段不符），
  以实现前回报用户、修订契约并留痕为唯一合法路径；禁止静默偏离。

## 10. 验收 Definition of Done

1. §8 全部测试 GREEN 且全量 pytest 通过数 ≥ 改动前基线（逐阶段记录
   计数，不得新增 skip/xfail/弱断言）；
2. 全市场配置下 `scripts/check_production_gates.py` G1–G7 通过，
   G7 零 bar = 0；
3. §6 全部测量产物落盘并归档，A/B 与双臂结果如实记录（含阴性）；
4. 终审标准（§0）达成，或停止条件触发并如实收尾；
5. `docs/p5_measurement_log.md` 完成：命令、commit、dataset_id、
   墙钟、pytest 前后计数、决策点证据、运营手册附录。
