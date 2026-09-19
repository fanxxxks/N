# P19 市场价格语义切换预注册契约（D4：raw 成交 / 复权因子序列）

- 状态：**APPROVED — captain 代行批准（2026-09-04，依据用户授权扩展记录
  89dd121 附录 C + p18 §5.2；批准声明逐字见 §9 批准记录）**。批准后本文
  保持前瞻性，禁止用实现或测量输出反向修改。
- **修订指针（2026-09-19，P38）**：§2.4 由
  [p38_data_qualification_unblock_draft.md](p38_data_qualification_unblock_draft.md)
  （APPROVED，用户直接指令「全部授权」）指针式修订——增补「限额记录必需于
  (股, 会话) 有 bar 行（open 有限且 >0）之处；无 bar 会话不属必需格网；
  缺失 = 非有限或 ≤0，与 exact 可交易矩阵同判据」。逐日交易语义
  （§2.4 的 blocked 规则）与其余条款零变化；详见 p38 §2/§7。
- 编号说明（已裁决固定，2026-09-04 captain 代行裁决，随 t11 评审存档）：
  本契约为**独立契约**，编号 **p19**（当前索引最高 p18 之后的下一空闲号；
  p15 空缺不回填——编号只向
  后推进的既有惯例）。裁决为独立编号的理由：(a) D4 是研究/执行语义变更
  （AGENTS §3.2：标签、执行、组合、因子输入），不是数据层修正——p17 的
  仲裁范围是数据底座，混入执行语义会模糊两个变化原因（AGENTS §12）；
  (b) 仓库先例：语义变更独立成约（p9/p11/p12/p13）。若 captain/用户裁决
  改为 p17 修订节，正式化 = 改名 + 索引行更新的同 commit 处理，正文语义
  不变。
- 起草：governance-auditor（AgentTeams tushare-pit-migration t10，D4 工作包
  前置），2026-09-04。
- 证据基线：**main @ `8ff21f6`**（全部代码事实在该 commit 实测；本分支
  codex/g0-tushare-pit-contract 代码与其逐字一致——仅 docs/ 差异）。数据
  事实 = D3 回填产物 **v2 dataset `aa6f6d55`**（分支 codex/tushare-schema-v1
  @ 6026129，8 commits 本地未 push；19 表 57,109,667 行，日期范围
  20140102..20260904）。数据侧上游契约 = p17（APPROVED，含勘误 #2/#3）。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**研究语义变更**
  （AGENTS §3.2：标签、执行、组合、因子输入）+ **运行态/晋级**
  （§4.3 fail closed）。仲裁顺序：本文/批准 → 契约测试 → 实现 → 测量日志。

## 1. 问题陈述（勘察事实，main @ 8ff21f6 逐点实测）

1. **价格源 = 前复权（qfq）**：`ashare_data/config.py:41` `adjust: str =
   "qfq"`——AkShare 拉取即前复权；Baostock 兜底 `adjustflag="2"` 同为 qfq
   （`ashare_data/akshare_client.py:553,580`）。生产 `daily` 表存 qfq OHLCV。
   qfq 历史随每次除权被整段改写（非 append-only）——数据层根因已由 p17
   §1.1 预注册，本文是其研究/执行侧的兑现契约。
2. **执行/组合语义运行在 qfq 价格上**：回测与评估引擎以
   `loader.raw_data_cache["open"]` 成交——`ashare_model/backtest.py:562`、
   `bare_factor_backtest.py:101`、`diagnostics.py:255`、`eval_search.py:93/
   391`、`p3_measurement.py:107`、`train_windows.py:326`。名义交易量、整手、
   佣金下限（`cost_matrix.py:15` min_commission floor 逐单生效）、最小交易
   额、成交额全部计算在失真价格上——qfq 价可系统性偏离真实交易价（除权
   后 qfq 价低于真实价），名义约束（100 股整手、现金、费用下限）随之失真。
3. **因子与标签在 qfq 上**：`compute_factor_tensor` 直接消费 qfq OHLCV
   frame（`data_loader.py:280-289`）；`target_ret = open_to_open_returns
   (open)`（`data_loader.py:302-304`；`processor.py:302-319`）；P3 稀疏
   因果目标同源（`train_windows.py:326`）。标签比率在 qfq 下近似正确，
   但**因子值随每次除权整段改写**——同一公式在两次拉取间历史值不可比
   （可复现性破坏，AGENTS §1 原则 4）。
4. **可交易性为近似推断**：`_blocked_parts`（`processor.py:135-163`）——
   停牌 = (o≤0)|(v≤0)|(pc≤0)；一字板 = o≈h≈l 且 change 相对板级费率
   ±0.005 容差（processor.py:158-160）；费率来自 `limit_rate(code)` 板级
   规则。无精确涨跌停价、无真实停牌表——近限价/四舍五入边界可误判。
5. **adj_factor 已入 pipeline 但零消费**：daily 表 adj_factor 列
   （`db.py:73`）→ normalize 仅 ffill().bfill()（`processor.py:384`）→
   loader 缓存（`data_loader.py:261`）——除此外全仓无任何消费（grep 证实）。
   v2 数据侧已就位：daily_bar_raw（未复权 OHLCV + pre_close）、
   daily_adj_factor、daily_price_limit（up_limit/down_limit + pre_close
   typed 供给）、daily_suspension（trade_date/suspend_type/suspend_timing，
   勘误 t44 后定义）——dataset `aa6f6d55` 内四表完整回填（D3 t9 输出）。

## 2. 语义定义（预注册）

1. **AshareDataLoader 双价格路径**（单一实现，禁止第二套换算——§1 原则 3）：
   - **raw_execution_prices**：未复权 OHLCV（v2 `daily_bar_raw`，含
     pre_close），名义元——执行的唯一价格基座；
   - **adjusted_factor_prices**：raw × 累计 adj_factor 派生连续序列——
     adjusted_x(t) = raw_x(t) × (adj_factor(t) / adj_factor_base)，base =
     加载窗口首日锚（序列内部比率与后复权一致；OHLC 一致缩放）；
   - adj_factor 缺失/断档/≤0 → 该股序列 fail-closed（不可交易 + degraded
     记录），**禁止 ffill().bfill() 伪造连续性**（现状 processor.py:384 的
     行为在新路径预注册废除）。
2. **成交语义**：PortfolioConstructor、回测、golden、simulation 的成交一律
   raw 名义价格——成交额、整手、佣金下限（min_commission floor）、最小
   交易额、现金约束、单股上限全用真实价格。
3. **因子与标签**：因子与标签（target_ret、P3 稀疏因果目标）使用
   adjusted 连续序列；因子值在新基座上整体重算（FACTOR_COMPUTE bump 承载）。
4. **执行约束精确化**：精确 up_limit/down_limit（daily_price_limit）与
   suspension（daily_suspension，trade_date 当日会话可用，§6.1）替换近似
   推断；语义：buy blocked ⟺ open ≥ up_limit；sell blocked ⟺ open ≤
   down_limit；suspension 表记录当日停牌；**当日无涨跌停价记录 → 该股当日
   fail-closed 不可交易（degraded 记录），禁止回退近似推断**。
5. **legacy 退位**：qfq daily_bar 路径与近似 tradability 路径退为 legacy
   audit（只读保留、零删除），仅可作 parity 对比，禁止进入正式计算路径
   （grep 验证）。

## 3. 范围与非目标

- **范围**：AshareDataLoader 双价格路径；factors/labels 消费切换；
  backtest/bare_factor/diagnostics/eval_search/p3_measurement/train_windows
  消费点切换；精确涨跌停/停牌约束接入；版本 bump；RED 清单与回归。实施
  落点 = t12 inScope（ashare_data/、ashare_portfolio/、factors.py、
  data_tier.py、evaluation.py、ashare_trading/、测试 + 分片注册）。
- **非目标**：不改因子公式/词表/搜索器/晋级阈值（晋级门 G1–G7 不动）；
  不改组合构造算法与权重语义；不改费率表语义（FEE_MATRIX）；不动 legacy
  数据与 p17 数据层 DDL（v2 表已就绪）；不做 UI；不切搜索默认器。

## 4. 不变量（实施全程必须成立）

1. 单一价格换算路径（loader 内派生，禁止第二套）；
2. 成交价 = raw，因子/标签价 = adjusted，两路径同源派生且可互推；
3. adj_factor 异常 → fail-closed（degraded 记录），禁止伪造连续性；
4. 精确约束优先；约束数据缺失 → fail-closed 不可交易（degraded 记录）；
5. 旧 artifact 版本集合拒绝（fail-closed）；跨价格语义禁止 matched
   comparison；
6. golden parity 逐日比较（逐日 target、buy/sell weight、turnover、order、
   cost、net return）。

## 5. 版本影响表（`git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 全量
检索：35 命中 = 33 赋值所有者 + 2 docstring 文本，基线 main @ 8ff21f6）

**bump（6 项）**：

| 版本所有者 | 旧值 | 新值 | bump 理由 |
|---|---|---|---|
| ashare_portfolio/execution_spec.py:22 EXECUTION_SPEC_VERSION | 2 | **3** | 成交价格语义 raw 化 + 精确涨跌停/停牌约束接入（执行语义变更） |
| ashare_model/versions.py:17 PROTOCOL_VERSION | "25" | **"26"** | 评价协议输入变更（标签价格基座 + 可交易性来源）——本文变更协议语义本身，非搜索层变更（P10/P14 先例不适用） |
| ashare_model/factors.py:46 FACTOR_COMPUTE_VERSION | 1 | **2** | 因子输入价格基座切换（qfq → adjusted），因子值整体重算 |
| ashare_model/targets.py:12 TARGET_CONTRACT_VERSION | 1 | **2** | 标签价格基座切换 |
| ashare_model/bare_factor_backtest.py:56 BARE_FACTOR_BACKTEST_VERSION | 3 | **4** | 同引擎成交价切换 |
| ashare_model/p3_measurement.py:50 P3_MEASUREMENT_VERSION | 3 | **4** | 测量语义随价格基座与约束来源变更 |

**不变（其余 27 项赋值所有者，归类理由）**：PORTFOLIO_CONSTRUCTOR 1
（构造器纯权重语义，ashare_portfolio/ 无价格消费——grep 证实）；
REBALANCE_POLICY 2、FEE_MATRIX 1（费率表语义不变，金额输入变化由
EXECUTION_SPEC 3 承载）；MANIFEST 1（v2 manifest 新实例、dataset_id 重算
——p13/p17 先例）；TUSHARE_STORE_SCHEMA_VERSION 1（L1 零改动）；
GRAMMAR 6、FEATURE_REGISTRY 6、RESEARCH_DOMAIN 3、REWARD "15"、
PROMOTION_RULE "2"、SEARCH_CONTRACT 4、SEARCHER_BENCH 4、ADMISSION 2、
MODEL 3、IMITATION 1、RL_DIAGNOSTICS 1、SEMANTIC_CACHE 1、LEDGER 1、
ELITE_ARCHIVE 1、P10_ADJUDICATION 1、RUNSPEC_SCHEMA 1、INDEX_SCHEMA 1、
LIFECYCLE_CONTRACT 1、DATA_TIER 1、ARTIFACT_SCHEMA 2（versions dict 记录
新版本集合为 additive——p9/p13 先例）、LEGACY_SCHEMA_VERSIONS (1,)。
**新增 artifact-facing 版本：无**（变更由既有所有者承载，双价格路径无独立
schema——执行 commit 须附全量 33 项逐项核对表）。

## 6. 迁移/拒绝策略

1. 旧 artifact（PROTOCOL "25" / EXECUTION_SPEC 2 / FACTOR_COMPUTE 1 /
   TARGET_CONTRACT 1 / BARE_FACTOR_BACKTEST 3 / P3_MEASUREMENT 3 时代）只读
   legacy；与新运行禁止 matched comparison、禁止拼接晋级证据——版本集合
   fail-closed（既有 G1 pin + artifact 分类权威）。
2. qfq daily_bar 路径退为 legacy audit：只读保留零删除；loader 正式源切换
   v2 双表；config adjust="qfq" 语义仅保留于 legacy audit 路径。
3. 因子值整体重算（新基座），无历史迁移；旧因子值随旧 artifact 只读。
4. **数据前置（实施启动条件）**：v2 价格四表（daily_bar_raw /
   daily_adj_factor / daily_price_limit / daily_suspension）完整性审计
   （行数/日期范围 vs manifest `aa6f6d55`）+ G1–G8 数据资格门——D3 已识别
   gate runner 与 v2 表名的 Catalog Error 缺口，须先落地 v2 gate adapter
   （独立任务）或等效 scoped 资格审计；未过门禁不启动实施。

## 7. 必测清单与 RED 计划（t12 实施前先红；新测试同 commit 注册分片）

1. **RED-1 未来除权泄漏（现状即红）**：构造除权事件 E ∈ T+k；断言 T 及
   以前的因子值/标签在 E 加入前后逐位不变——现状 qfq 整段改写 → 红；
   实现后绿（adjusted 序列按已实现事件锚定，未来事件不可见）。
2. **RED-2 除权日连续性（golden）**：合成含除权股票——adjusted 序列在
   除权日无跳变、raw 序列呈现真实价格缺口；adjusted 收益率与手工复权公式
   逐位一致。
3. **RED-3 raw 成交语义**：整手（100 股整数倍）、现金约束、佣金下限
   （min_commission floor）、最小交易额在 raw 名义价格上断言；构造 qfq/raw
   价差 ≥10 倍场景（旧行为红）。
4. **RED-4 精确约束**：up_limit/down_limit/suspension 表驱动掩码 vs 旧近似
   掩码的误判差异用例 + 涨跌停价缺失行 fail-closed（不可交易 + degraded
   记录）。
5. **RED-5 逐日 parity**：reward/backtest/simulation 同种子同配置下逐日
   比较 target、buy/sell weight、turnover、order、cost、net return——
   golden parity 工具既有，新基座全字段逐日一致。
6. **RED-6 旧 artifact 版本拒绝**：PROTOCOL "25" 时代 artifact 在分类入口
   被拒（fail-closed）。
7. **相邻回归**：universe/PIT mask、成本矩阵、paper sim 路径；§8.2 全约束
   验证（T+1、整手、停牌、涨跌停、佣金下限、印花税、过户费、滑点、现金
   上限、单股上限、权重和 ≤1）。

## 8. 资源上限与停止条件

- 资源：纯代码 + 测试；无搜索预算消耗；全量回归一轮并行 + 一轮串行对账；
  golden parity 本地分钟级。
- 停止条件（任一命中 → 停止、留证、上报，禁止静默变通）：
  1. 逐日 parity 无法对齐且差异不可归因（归因 = 本文语义差异或已声明的
     浮点容差外差异）；
  2. 复权序列除权日跳变无法由 adj_factor 解释；
  3. v2 价格四表资格审计不过或 gate adapter 缺口未闭合；
  4. adj_factor 异常（≤0/断档/缺失）超过审计阈值且无法归因；
  5. RED 出现预期之外的无关失败（§2.2.3）；
  6. 与 D5/D6 并行线出现共享文件冲突（§12）。

## 9. 批准与实施边界

- **批准流程**：t11 评审 + 用户批准门通过后生效。
- **批准记录（p16 `2f5e18b` 先例补记，t54，2026-09-04）**：
  - 批准人 = captain（代行，依据用户授权扩展记录 `89dd121` 附录 C——
    「这次批准，接下来你自行把握即可不用再问我」——+ p18 §5.2）；
  - 批准对象 = 本契约全文 `26d2697` 版本 + 独立编号裁决（p19；p15 空缺
    不回填）；
  - 批准声明（逐字，来源 = captain → governance-auditor 邮箱 05fded07，
    ts 1788543107392）：「批准 p19 独立编号与 D4 预注册契约全文（26d2697
    版本），作为 D4 价格语义切换的实施依据；批准依据 = 用户路线图 D4
    条款 + p18 代行授权」；
  - 评审 = t11 verdict **pass**（语义完整性/版本影响表/必测清单/登记
    编号/批准门五节全过；评审记录随 t11 存档）；
  - 数据前置不变：§6.4（v2 价格四表完整性审计 + G1–G8 gate adapter，
    t53）完成方可启动 t12 实施。
- **t11 非阻断移交的处理记录（两条）**：
  1. §5 不变枚举未逐一命名 TIER_REPORT 与 PORTFOLIO_OPTIMIZER（另有
     TUSHARE_STORE_SCHEMA_VERSION 系 D2 新增所有者、超出 main 基线属
     前瞻覆盖）——已转 **t12 实施的强制要求**：全量 33 项逐项核对表必须
     纳入这两个所有者并给出无价格消费/no-bump 依据；本补记登记该移交，
     兑现由 t12 执行 commit 承担；
  2. §9 批准记录补记——**由本补记（t54）兑现**。
- **当前状态：APPROVED，未实施**；t12 实施待 t53（gate adapter）完成后
  启动。索引行"暂定编号"措辞按本裁决读作已固定（CONTRACTS.md 零改动）。
- 实施顺序：本契约先行 → t12 实施（RED → 最小实现 → GREEN/回归 → 原子
  提交）→ t13 评审；实施分支从最新 main 创建（集成 D1–D3 线合并结果后），
  数据前置（§6.4）完成方可启动。
- 文件落点：t12 inScope；热点文件（evaluation.py、run_sim.py）单点集成
  （AGENTS §12）。
- 本契约不授权 push/远程 PR；全程仅本地。
- 未运行/未验证项如实列出；本文不预写任何验证结果（AGENTS §11.1）。
