# WP15 冻结期望（FROZEN EXPECTATIONS）——可融资地接近目标组合

冻结日期：2026-09-22（实现存在之前；当前工作区无任何 WP15 代码，核实命令与结果见 §0.4）。

## §0 文档地位、真源与使用规则

### §0.1 地位
- 本文件是 WP15 的**唯一期望真源**：p51 测试（`tests/test_p51_wp15_*.py`）的一切数值断言只能引用本文件 §编号，不得照实现输出反写（W5 规则）。
- 对实现者/修复者**只读**（红线 3）：实现与本文件冲突视为实现缺陷，修实现；若确认本文件推导本身有误，经审核确认后**只可以"勘误附注"追加于文末，不得删改原文**。
- 本文件先于实现冻结；所有数字仅由 WP15 任务书（W1–W7 与场景 1/2/3 原文）与侦察文档 `reports/wp15/_recon/map.md` 及其引用的代码事实推导，未引用任何实现输出。

### §0.2 推导依据（全部本会话核实）
- 任务书：W1–W7 与场景 1/2/3 文字（§2/§3/§4 逐字保留）。
- `reports/wp15/_recon/map.md` §1–§8。
- `ashare_portfolio/constructor.py`：line 272（`initial_funding := prev.sum() <= 1e-12`）、299（`forced_exit = held & ~eligible & ~sell_blocked`）、317-321（legacy wind-down → mandatory）、410-413（`budgeted` 判定）、441-514（priority 段；456-463 卖序 `(-rank, key)` / 买序 `(rank, key)`；477-506 逐腿分配；508-514 诊断键）、519（现金再封顶）、29（`PORTFOLIO_CONSTRUCTOR_VERSION = 2`）。
- `ashare_execution.py`：103-135（`trade_cost`：commission=max(rate·n, min_commission) 每非零订单独立下限；stamp 仅卖出；transfer/slippage 双边）、137-141、196-225（`affordable_shares`：整手向下取整、现金含费用、line 221 容差 1e-9）。
- `ashare_model/backtest.py`：302-324（卖出段；310-318 目标≈0 时整仓卖出含零股、部分卖出整手 floor；322 卖款净费用当日入现金）、333-363（买入段：整手 floor、可负担循环、354 费用对账、355-359 fail-closed）、269（capital=当期权益）。
- `configs/r1_slow_ab.yaml`：58（initial_capital 100000.0）、63（min_trade_amount 5000.0）、66-70（费率）。
- `ashare_domain.py:271`（slow_fundamental `turnover_budget=0.05`）、`ashare_data/config.py:448`（`turnover_allocation` 默认 `"priority"`）、`ashare_data/config.py:939-940`（yaml 未写 turnover_budget 时由域注入）。

### §0.3 全局红线（实现必须遵守）
1. 不改 `AlphaGPT/data/**`；不动 `ashare_model/experiment_integrity/`（WP14）。
2. 默认行为不变：新分配方式是显式 opt-in（`turnover_allocation` 新枚举值 `"trade_groups"`；W4 新 kwarg 默认 `None`）。未显式开启时代码路径与现在逐字节一致；参考配置补键 `turnover_allocation: priority`（语义=现缺省，加法式）。
3. 费用唯一权威 `ExecutionCostModel`：禁止任何自有费用表/费率推导；一切费用数字必须出自 `buy_cost/sell_cost/trade_cost/affordable_shares` 调用。
4. 禁止另建回测系统：离散可行性放纯计划函数（建议 `ashare_portfolio/trade_groups.py`，map §5）或既有执行层。

### §0.4 冻结时点核实（本会话执行）
```
ls /d/minequant/AlphaGPT/ashare_portfolio/
→ __init__.py __pycache__ constructor.py execution_spec.py optimizer.py rebalance.py   # 无 trade_groups.py
grep -rn "trade_groups|initial_funding_used" ashare_portfolio/ ashare_data/config.py → 无匹配
ls tests/ | grep -i p51 → 无输出
```
即：`trade_groups` 模式、W4 kwarg、p51 测试在冻结时点均不存在。

---

## §1 冻结的确定性分配算法（`turnover_allocation = "trade_groups"` 模式）

**总则：算法无自由度——同样输入唯一输出。** 所有排序键为全序（rank 由构造器给出、0=最优；`stable_keys` 唯一，constructor.py:187-188 校验）；所有金额为 float64；预算判定用精确比较 `x + y <= B_r`，现金判定用绝对容差 `<= C + 1e-9`（对齐 `affordable_shares`，ashare_execution.py:221）。

### §1.1 模式与接线前提（规范性要求）
1. 新枚举值：`turnover_allocation = "trade_groups"`，进入 `validate_portfolio_config`（constructor.py:102-107 扩展）。复用既有键则 provenance 自动携带模式（map §5；`execution_spec._PORTFOLIO_CONFIG_FIELDS` 无需新增）。
2. 构造器在该模式下走**与 unbudgeted 分支完全相同**的代码路径（阈值过滤、min-trade 过滤、`_cap_increases_to_available_cash` 照旧），仅 `diagnostics["turnover_budget_mode"] = "trade_groups"`；构造器自身的 priority/proportional 预算分配**不执行**（预算交由 §1.4 规划器）。
3. 规划器是**纯函数**（建议 `ashare_portfolio/trade_groups.py`）：无盘 IO、无内部状态、无随机性；输入输出见 §1.2/§1.8。
4. 执行层契约（W2）：规划器只产出计划；实际执行失败买入时**保留真实卖出与现金，禁止伪造回滚**，记录 `execution_failed_buys`（tuple[str]，按 stable_key 排序），不做补偿交易。
5. 参考配置补键：`configs/r1_slow_ab.yaml` backtest 段显式加 `turnover_allocation: priority`（当前缺省即 priority，语义不变，加法式）。

### §1.2 规划器输入
- `prev_shares[i]`（≥0）、`prices[i]`（入场开盘价；`<=0` 或非有限 = 无价，禁止引用该名下单，engine 语义 backtest.py:293）、`cash >= 0`、会话权益 `E = cash + Σ prev_shares[i]·prices[i]`、目标权重 `w[i]`（构造器输出）、`rank[i]` 与 `stable_keys[i]`（构造器排序键）、配置（`turnover_budget`、`min_trade_amount = m`、`lot_size = L`）、`ExecutionCostModel` 实例、掩码（eligible/buy_blocked/sell_blocked）、`initial_funding_used`（W4，§6）。
- 需求定义（yuan，未取整）：卖出需求 `s_i = max(v_i − w_i, 0)·E`，买入需求 `b_i = max(w_i − v_i, 0)·E`，其中 `v_i = prev_shares[i]·prices[i]/E`。
- 预算：`B_r = turnover_budget × E`；`turnover_budget is None` 或豁免生效（§6）时 `B_r = ∞`。

### §1.3 腿分类与执行顺序（三类，序固定）
1. **M 类（强制风险退出，独立处理，不受预算挤压）**：`forced_exit = held & ~eligible & ~sell_blocked`（constructor.py:299 同定义）∪ legacy wind-down（constructor.py:317-321 同定义）。按卖序 `(-rank, stable_key ASC)` 逐名：有价且未 blocked → **整仓卖出**（含零股，engine 语义 backtest.py:310-318），预算消耗 0；`sell_blocked` 或无价 → 保留持仓并记录 `forced_exit_blocked`（(key, cause) 对，cause ∈ {"sell_blocked","missing_price"}，按 key 排序）。M 类豁免 min-trade 过滤（对齐构造器 mandatory 豁免）。
2. **D 类（目标降杠杆单边卖，预算豁免，量受目标净降幅封顶）**：M 类执行后，对剩余 discretionary 需求计算 `S = Σ s_i`、`B_dem = Σ b_i`、`R = max(S − B_dem, 0)`。按卖序逐名配 `fill_i = min(s_i, R_remaining)`（整手 floor；合并名义 ≥ m，否则跳过并记码）；R 为 0 时 D 类为空。D 类不消耗预算；**边界（冻结）**：D 类量只由 R 封顶，不受换手预算约束——若目标反复大幅升降仓位，D 类不受预算限制属本实验模式已知边界，必须由 `deleveraging_sell_notional` 诊断报告。
3. **R 类（普通换股，可融资交易组）**：§1.4。预算只约束 R 类；R 类买卖两腿名义均计入预算消耗。

### §1.4 交易组构造算法（R 类核心）
卖候选（剩余卖出需求 > 0 的 discretionary 名，M/D 已扣减）按 `(-rank, stable_key ASC)` 排序（与 priority 卖序一致，constructor.py:456-459）；买候选（剩余买入需求 > 0 且可选：eligible、~buy_blocked、有价）按 `(rank, stable_key ASC)` 排序（与 priority 买序一致，constructor.py:460-463）。

现金账本顺序：M 卖 → D 卖 → 各组按构造顺序；卖腿入账 `x − Fsell(x)`（净额，当日可用，engine 语义 backtest.py:322），买腿出账 `y + Fbuy(y)`；现金不允许为负。

**循环**：重复直至无可行组：
1. 取**买序中第一个能构成可行组的候选** j（曾在本会话被记跳码的候选不再参与）；逐个检查买序中未满足候选，全部不可行则会话结束。
2. **确定 y\***：y 取整手格点 `n·L·p_j`（n≥1 整数）。候选 y 可行 ⟺ 存在可组成的 x（按卖序连续附着，见步 3 的组成规则）满足：
   - (a) 预算 `x + y <= B_r`；
   - (b) 融资 `y + Fbuy(y) <= C_g + x − Fsell(x)`（C_g = 当前账本现金；费用为合并口径 §1.5 的增量）；
   - (c) `y <=` j 的剩余买入需求；
   - (d) 合并名义 ≥ m 且 ≥ 一手；
   可行性对 n 单调（n 减小时 x 可相应缩小，约束放宽），故 **y\* = 最大可行 n 的名义**（从 `n_max = floor(需求/(L·p_j))` 向下扫描，首个可行者）。若 j 的需求 < 一手 → 记码 `lot`；≥ 一手但最大可合并名义 < m → 记码 `min_amount`。
3. **附着卖出（最小融资原则）**：`need = y* + Fbuy(y*) − C_g`；若 `need <= 0` → **纯买组**（W1"现有现金足够时可以只买"），无卖腿。否则沿卖序逐候选 a：`u_a = min(a 的剩余卖出需求, B_r − y* − X_已附)`；取 **最小的整手格点 f ≤ u_a**（若 a 的卖出为清仓腿——即该名本会话目标权重为 0 且 u_a = 全部剩余持仓——允许整仓含零股）使 `X + f − Fsell(X + f) >= need`；不存在这样的 f 则 `f = u_a`（整手 floor）继续下一候选。附着结果唯一（`x − Fsell(x)` 对 x 严格单增）。
4. 成组：记录组 `{sells: ((key, shares, notional), ...) 按卖序, buy: (key, shares, notional), budget_consumed: x + y*}`；扣预算、更新账本与双侧剩余需求。**同一买候选可连续成组**（预算未尽且仍有需求时它仍是队首）。
5. 任何被记跳码的候选从本会话候选表移除（跳过不消耗预算，对齐 constructor.py:479-500 语义）；循环每步要么成组（预算至少减一个最小可行组）要么移除一个候选，必然终止。

**合规性说明（W1 禁止事项映射）**：本规则不是"固定买卖各半"——y\* 由需求/现金/预算/整手联合决定，卖出仅取融资所需最小值。反例（冻结示例）：C_g = 3,000、B_r = 10,000、零费、买需求 10,000、价格 10、L=100 → y\* = 6,500（650 股）、x = 3,500（350 股），非对半。禁止强制最低仓位：无买需求时留现金（§4 分支 C）。禁止 exits-first：卖腿永不由预算单独驱动（预算外的卖出只能来自 M/D 类）。

### §1.5 同日同股同方向合并计费（W3）
- 计费单元 = (会话, 名, 方向)。该单元全部成交数量合并为单一名义 n，**恰好一次**送费用权威：`sell_cost(n)` / `buy_cost(n)`；禁止按组、按腿、按手重复计费。理由：`trade_cost` 对每次调用独立收取 `max(rate·n, min_commission)`（ashare_execution.py:116-120），拆分调用会重复最低佣金（map §3）。
- 规划期增量费用：向某单元追加 Δ 时，费用增量 = `fee(n + Δ) − fee(n)`（两次都用合并名义过权威）。增量之和 = 最终合并费用（望远镜恒等式），规划期可负担性判定与执行期对账**逐字节一致**。
- 会话费用诊断 `session_fees = {"sell": …, "buy": …, "total": …}` 一律取自权威调用的 `total`。

### §1.6 整手取整方向与最低金额（冻结）
- 买入：向下取整至 L 的整数倍（engine backtest.py:339-344 与 `affordable_shares` ashare_execution.py:196-225 均 floor）。
- 卖出：部分卖出向下取整至整手；**清仓卖出（该名目标权重 0）卖全部剩余股数，含零股**（engine backtest.py:310-318）。
- 最低金额约束作用于 (会话, 名, 方向) 的**合并名义 ≥ m**；M 类豁免；D 类与 R 类不豁免。

### §1.7 预算用不完的原因码（枚举封闭，W2）
码集（字符串字面量冻结）：`"budget" | "lot" | "min_amount" | "cash" | "fees" | "no_viable_counterpart"`。
对**首个未满足的买候选**（买序）按以下**固定顺序**判定，首个命中者即会话级 `budget_unspent_reason`（预算恰好用尽或无需求时为 `None`）；每个未成交腿在 `leg_skip_reasons` 记 `(key, side, code)`（按 (key, side) 排序）：
1. `lot`：该候选剩余买入需求 < 一手名义 `L·p_j`。
2. `min_amount`：可成整手但最大可合并名义 < m。
3. `no_viable_counterpart`：不存在任何可接受卖腿（全部：剩余卖出需求为 0 / sell_blocked / 无价）且 `C_g < y_min + Fbuy(y_min)`（`y_min = max(m, L·p_j)`）。
4. `cash`：存在卖腿但**零费松弛**（费率全 0、预算视 ∞）下仍不可行（现金+全部可接受卖容量凑不足 `y_min` 的融资）。
5. `fees`：零费松弛可行，但**实费松弛**（真实费率、预算视 ∞）不可行。
6. `budget`：实费松弛可行而实际剩余预算不足最小可行组（纯买组为 `y_min`；配对组为 `y_min + x_min`）。
（结构缺席 → 经济不可负担 → 额度不足的判定次序为冻结设计，先小后大无循环。）

### §1.8 诊断字段契约（trade_groups 模式新增键；既有键语义不回归）
- `turnover_budget_mode = "trade_groups"`（既有键的新枚举值）。
- `forced_exit_executed`: tuple[str]（M 类实际卖出名，按执行序）。
- `forced_exit_blocked`: tuple[(str, str)]（按 key 排序；cause ∈ {"sell_blocked","missing_price"}）。
- `deleveraging_sell_notional`: float（D 类合计名义）；`deleveraging_skipped`: tuple[(str, str)]。
- `trade_groups`: tuple[组记录]，组记录 = `{"sells": tuple[(key, shares:int, notional:float)], "buy": (key, shares:int, notional:float) | None, "budget_consumed": float}`（纯买组 sells=()；只含 R 类组，M/D 不入）。
- `budget_unspent_reason`: §1.7 枚举或 None；`leg_skip_reasons`: tuple[(str, str, str)]。
- `session_fees`: {"sell": float, "buy": float, "total": float}（§1.5）。
- `turnover_budget_consumed`: 本模式下 = Σ 组 `budget_consumed`（**仅 R 类**；M/D 豁免于此键，模式内语义冻结）。
- `execution_failed_buys`: tuple[str]（§1.1.4）。
- 既有键 `forced_exit_count` / `legacy_position_wind_down` / `initial_funding`（W4 后语义见 §6）保持。

---

## §2 场景 1：零费多期换股——完整推导

> 场景原文（逐字）："零费多期换股：10 万元、十只旧股换十只新股，价格不变，预算 10%，最低订单 2500 元 → 每期可完成一组卖 5000、买 5000；权益与投入不因分配机制下降；目标差距持续缩小。"

### §2.1 规范实例化（场景未给价格向量，冻结如下）
- 20 名；所有价格 = 10 元且永不移动；L = 100 股；费率全 0（commission_rate=min_commission=stamp=transfer=slippage=0）；m = 2,500；`turnover_budget = 0.10` → 每会话 `B_r = 0.10 × 100,000 = 10,000`。
- 旧股 O1..O10 各持 1,000 股（各 10,000 元，总 100,000）；新股 N1..N10 各目标 10,000 元。
- rank（0 = 最优）：N1=0, …, N10=9, O1=10, …, O10=19。**卖序**（最差先）：O10, O9, …, O1；**买序**（最优先）：N1, …, N10。
- 分类：S = 50,000? ——否：S = Σ s_i = 100,000（十个各 10,000），B_dem = 100,000，R = max(S − B_dem, 0) = 0 → 无 D 类；无 M 类（全部 eligible）。

### §2.2 每期组推导（一般式）
C_g = 0，B_r = 10,000。y ∈ {1,000, 2,000, …}（100 股 × 10 元）。约束：(a) x + y ≤ 10,000；(b) 零费融资 y ≤ x；x ∈ 1,000 的整倍数。
- y = 5,000：需 x ≥ 5,000 且 x + y ≤ 10,000 → x = 5,000 ✓ 可行。
- y = 6,000：x ≥ 6,000 → 和 ≥ 12,000 > 10,000 ✗。
- 由 (b)(a)：y ≤ x 且 y ≤ 10,000 − x ⇒ y ≤ 5,000。**y\* = 5,000**；最小融资 x = 5,000。预算消耗 10,000 = B_r → **每会话恰好一组**，组 = (卖 500 股队首旧股 = 5,000 元，买 500 股队首新股 = 5,000 元)。每腿 5,000 ≥ m = 2,500 ✓，500 股 = 5 整手 ✓。
- 队首推进：卖腿把队首旧股卖至 0 需 2 期，随后下一旧股成为队首；买腿同理。第 2j−1 与 2j 期作用于 O(11−j) 与 N(j)，j = 1..10。

### §2.3 多期轨迹表（完整 20 期）
| 期 k | 卖出腿 | 买入腿 | 期终现金 | 权益 | 剩余目标差距 |
|---|---|---|---|---|---|
| 1 | O10 500股/5,000 | N1 500股/5,000 | 0 | 100,000 | 190,000 |
| 2 | O10 500股/5,000（→0） | N1 500股/5,000（→10,000） | 0 | 100,000 | 180,000 |
| 3 | O9 500/5,000 | N2 500/5,000 | 0 | 100,000 | 170,000 |
| 4 | O9（→0） | N2（满） | 0 | 100,000 | 160,000 |
| 5 | O8 | N3 | 0 | 100,000 | 150,000 |
| 6 | O8→0 | N3 满 | 0 | 100,000 | 140,000 |
| 7 | O7 | N4 | 0 | 100,000 | 130,000 |
| 8 | O7→0 | N4 满 | 0 | 100,000 | 120,000 |
| 9 | O6 | N5 | 0 | 100,000 | 110,000 |
| 10 | O6→0 | N5 满 | 0 | 100,000 | 100,000 |
| 11 | O5 | N6 | 0 | 100,000 | 90,000 |
| 12 | O5→0 | N6 满 | 0 | 100,000 | 80,000 |
| 13 | O4 | N7 | 0 | 100,000 | 70,000 |
| 14 | O4→0 | N7 满 | 0 | 100,000 | 60,000 |
| 15 | O3 | N8 | 0 | 100,000 | 50,000 |
| 16 | O3→0 | N8 满 | 0 | 100,000 | 40,000 |
| 17 | O2 | N9 | 0 | 100,000 | 30,000 |
| 18 | O2→0 | N9 满 | 0 | 100,000 | 20,000 |
| 19 | O1 | N10 | 0 | 100,000 | 10,000 |
| 20 | O1→0 | N10 满 | 0 | 100,000 | 0 |

（表中未标注的卖出/买入腿均为 500 股 / 5,000 元。）

### §2.4 不变量与论证
- **权益不变式**：每期 现金 0 → 卖 5,000 → 5,000 → 买 5,000 → 0；持仓名义恒为 100,000（价格不动）⇒ 权益 = 100,000 ∀k；**投入**（持仓名义）恒 100,000，从不清仓降投入——"权益与投入不因分配机制下降"成立（零费、无价格变动，分配机制本身 1:1 转移价值）。
- **目标差距单调缩小**：定义 G(k) = Σ 名义级剩余需求。每腿满足 0 ≤ fill ≤ 剩余需求 ⇒ 未触及名的需求不变、触及名严格减少其 fill；G(0) = 100,000(卖侧) + 100,000(买侧) = 200,000，每期恰减 x + y = 10,000 ⇒ G(k) = 200,000 − 10,000k 严格单调降。
- **完成期数**：G = 0 需 k = 200,000/10,000 = **20 期**（每期换出 5% 权益，全部换仓 = 100%/5% = 20 期）。
- **对照（缺陷显影，非期望）**：现行 priority 会以整 10,000 预算先卖（exits-first），10 期清仓、期间零买入、现金累积；清仓后 line 272 判 initial_funding=True → 下一期 unbudgeted 整笔买入（constructor.py:272+411，map §2 缺陷）。trade_groups 模式期望即本节表：全程满投、逐期双 Leg 推进。

---

## §3 场景 2：有费用的融资——逐步复算与第二期

> 场景原文（逐字）："有费用的融资：10 万元全部持有 A，价格 10 元，转向 B，整手 100 股，双边佣金 0.1%，预算 10% → 例如首期卖 500 股、买 400 股，现金 991 元、权益 99991 元；用实际成交与费用对账，不凭目标权重宣称满仓。"

### §3.1 规范实例化
- E₀ = 100,000；A：10,000 股 @10；目标 100% B（B 价格 = 10 元）；L = 100；佣金双边 0.1%（commission_rate = 0.001，min_commission = 0，stamp = transfer = slippage = 0）；`turnover_budget = 0.10`。
- S = 100,000，B_dem = 100,000 → R = 0（无 D 类）；无 M 类。

### §3.2 首期逐步复算（核到用户数字）
B_r = 0.10 × 100,000 = 10,000。C_g = 0。y、x ∈ 1,000 的整倍数（100 股 × 10 元）。融资不等式：`x − 0.001x ≥ y + 0.001y`。
- y = 5,000：需 0.999x ≥ 1.001×5,000 = 5,005 ⇒ x ≥ 5,010.01… ⇒ 最小整手格点 x = 6,000 ⇒ x + y = 11,000 > 10,000 ✗。
- y = 4,000：需 0.999x ≥ 1.001×4,000 = 4,004 ⇒ x ≥ 4,008.01… ⇒ 最小整手格点 **x = 5,000**；x + y = 9,000 ≤ 10,000 ✓。
- ⇒ **y\* = 4,000（400 股 B），x = 5,000（500 股 A）**。
- 费用（各一次送权威）：卖佣金 = 0.001×5,000 = 5.00；买佣金 = 0.001×4,000 = 4.00；合计 9.00。
- 现金 = 0 + (5,000 − 5) − (4,000 + 4) = 4,995 − 4,004 = **991.00** ✓（用户：991）。
- 权益 = 991 + 9,500×10 + 400×10 = 991 + 95,000 + 4,000 = **99,991.00** ✓（用户：99991；= 100,000 − 费用 9）。
- 预算消耗 9,000/10,000；剩余 1,000，第二组不可行（y = 1,000 需 x ≥ (1,001−991)/0.999 = 10.01 ⇒ x = 1,000，和 2,000 > 1,000 ✗；纯买 1,001 > 991 ✗）→ `budget_unspent_reason = "budget"`（零费松弛下 2,000 > 1,000 同样不可行，故非 cash/fees）。
- **对账口径（W5）**：以实际成交与费用记账——期末 B 权重 = 4,000/99,991 ≈ 4.00%，距目标仍远；**不得凭目标权重宣称满仓**。名义级两腿差距（以当期权益重定基）200,000 → 190,991（减 9,009 = x + y 9,000 + 费用 9：费用缩减权益同时缩减买侧目标）。

### §3.3 第二期走向推导
期初：现金 991；A 9,500 股；B 400 股；E₁ = 99,991。
- B_r = 0.10 × 99,991 = **9,999.10**。卖需求 A = 95,000；买需求 B = 99,991 − 4,000 = 95,991 ⇒ R = 0。C_g = 991。
- y = 5,000：需 0.999x ≥ 5,005 − 991 = 4,014 ⇒ x ≥ 4,018.02… ⇒ x = 5,000 ⇒ 和 10,000 > 9,999.10 ✗（权益缩水使预算差 0.90，恰好卡死）。
- y = 4,000：需 0.999x ≥ 4,004 − 991 = 3,013 ⇒ x ≥ 3,016.02… ⇒ **x = 4,000**；和 8,000 ≤ 9,999.10 ✓。
- ⇒ 第二期 = **卖 400 股 A、买 400 股 B**。费用：卖 4.00、买 4.00，合计 8.00。
- 现金 = 991 + (4,000 − 4) − (4,000 + 4) = 991 + 3,996 − 4,004 = **983.00**。
- 权益 = 983 + 9,100×10 + 800×10 = 983 + 91,000 + 8,000 = **99,983.00**（= 99,991 − 8）✓。
- 预算消耗 8,000；剩余 1,999.10；第三组：y = 1,000 需 x ≥ (1,001 − 983)/0.999 = 18.02 ⇒ x = 1,000 ⇒ 和 2,000 > 1,999.10 ✗；纯买 1,001 > 983 ✗ → 会话止，`budget_unspent_reason = "budget"`。
- 名义级两腿差距：190,991 → 182,983（减 8,008 = x + y 8,000 + 费用 8）。

---

## §4 场景 3：强制退出——三分支期望

> 场景原文（逐字）："强制退出：一万元持仓失去资格，普通预算仅 2000 元 → 可卖时仍按风险退出规则处理；不可卖时记录阻断；没有合格新目标时允许留现金。"

### §4.1 规范实例化
E = 10,000；单名 X 1,000 股 @10（全部权益）；X 失去资格（~eligible）→ M 类；普通预算 2,000 元/会话（`turnover_budget × E = 2,000`）；零费（场景未给费率，冻结为 0；含费变体只影响现金与权益的费用差）；m = 0（未给）；L = 100。Y = 合格新目标（价格 10，目标 10,000 元），仅在分支 1 存在。

### §4.2 分支 1：可卖（X 未 blocked、有价）
1. M 类整仓卖出 X：1,000 股，名义 10,000，**预算消耗 0**（豁免；10,000 > 2,000 正是"普通预算挤压不了风险退出"的验证点）。现金 = 10,000。
2. R 类：买序队首 Y，需求 10,000。纯买组判定：y + 0 ≤ 10,000 ✓ 但 y ≤ B_r = 2,000 ⇒ **y\* = 2,000（200 股）**。预算消耗 2,000（用尽）。剩余买需求 8,000 → 跳码 `budget`。
3. 期末：现金 8,000；持仓 Y 2,000；权益 10,000。
4. 期望诊断：`turnover_budget_mode = "trade_groups"`；`forced_exit_count = 1`；`forced_exit_executed = (X,)`；`forced_exit_blocked = ()`；`trade_groups = ({"sells": (), "buy": (Y, 200, 2000.0), "budget_consumed": 2000.0},)`；`turnover_budget_consumed = 2000.0`；`budget_unspent_reason = None`（用尽）；`leg_skip_reasons = ((Y, "buy", "budget"),)`；`session_fees = {sell: 0.0, buy: 0.0, total: 0.0}`。

### §4.3 分支 2：不可卖（X sell_blocked）
1. M 类：X 保留持仓，记录阻断。无任何成交。
2. R 类：Y 为买候选，C_g = 0，无可接受卖腿（X blocked 不可附）→ §1.7 序 3 命中。
3. 期末：持仓 X 10,000；现金 0；权益 10,000。
4. 期望诊断：`forced_exit_count = 1`；`forced_exit_executed = ()`；`forced_exit_blocked = ((X, "sell_blocked"),)`；`trade_groups = ()`；`turnover_budget_consumed = 0.0`；`budget_unspent_reason = "no_viable_counterpart"`；`leg_skip_reasons = ((Y, "buy", "no_viable_counterpart"),)`。（无价变体：cause = "missing_price"。）

### §4.4 分支 3：可卖但无合格新目标
1. M 类整仓卖出 X：名义 10,000，预算消耗 0。现金 = 10,000。
2. R 类：买候选集为空 → 无组；**现金保留**（禁止强制再投资/最低仓位）。
3. 期末：现金 10,000；权益 10,000。
4. 期望诊断：`forced_exit_executed = (X,)`；`forced_exit_blocked = ()`；`trade_groups = ()`；`turnover_budget_consumed = 0.0`；`budget_unspent_reason = "no_viable_counterpart"`；`leg_skip_reasons = ()`。

---

## §5 W6 配置可行性判据与措辞边界

### §5.1 判据公式（忽略整手与费用）
一次普通换股（一对买卖腿）在单会话可完成 ⟺
```
2 × min_trade_amount ≤ turnover_budget × E
```
（卖腿与买腿各自 ≥ 最低订单金额；两腿名义均计入预算。）单腿（纯买或 D/M 外的单卖）可行 ⟺ `min_trade_amount ≤ turnover_budget × E`。含费与整手只会更紧：精确判据需 `sell_min + buy_min + 双边费用(整手格点上取整) ≤ B_r`，W6 冻结为忽略整手与费用的上界必要条件版。

### §5.2 slow 参考配置代入判定
`configs/r1_slow_ab.yaml`：`initial_capital = 100000.0`（line 58）、`min_trade_amount = 5000.0`（line 63）、`turnover_budget = 0.05`（ashare_domain.py:271 经 config.py:939-940 注入）：
```
2 × 5000 = 10000 > 0.05 × 100000 = 5000
```
⇒ **该配置下单会话无法完成一对普通买卖**（判定：不可行）。边界注记：单笔最低腿 5,000 恰等于全部预算 5,000——每会话至多执行一支最低尺寸的单腿，**永远**凑不成一对；等号 `2m = B_r` 只在忽略整手与费用的理想化下恰好可行。

### §5.3 措辞边界（冻结原文，报告与 `reports/wp15/config_feasibility.json` 章节必须采用）
> 在该配置下测出的长期受限收益不能直接代表因子的信息质量：收益被执行可行性（换手预算与最低订单金额的结构性冲突）压制，而非由信号的信息含量决定。对此配置产出的任何收益结论，必须附带本可行性判定；要评估因子信息质量，需在满足 §5.1 判据的配置下重测，或明确声明收益为执行受限下界。

---

## §6 W4 首次建仓豁免状态机

### §6.1 缺陷（冻结描述）
现行 `initial_funding := bool(prev.sum() <= 1e-12)`（constructor.py:272）把"现在空仓"当"首次建仓"：固定目标探针下 priority 连续卖空后，下一期 `budgeted = False`（constructor.py:411）重新取得 unbudgeted 全额买入资格——清仓自动重获豁免，违反豁免语义（map §2）。

### §6.2 状态机定义
- **状态变量**：`initial_funding_used ∈ {False, True}`（每账户账本一个；即"已用过豁免或曾有过持仓"）。吸收态 True。
- **API**：`construct(..., initial_funding_used: bool | None = None)`。`None`（缺省）→ 与现行 line 272 **逐字节一致**（红线 2；p37 漏斗计数语义不动，tests/test_p37_r02_matrix.py:212-230 钉住的语义不回归）。传入 bool b → `initial_funding := (not b) and (prev.sum() <= 1e-12)`。
- **迁移规则（调用方在 run 级维护，冻结）**：
  - 初始化：账本书自 run 起始（含恢复的持久化书）**始终为空** → False；否则 True。
  - 每会话后：`used := used or (执行后账本书非零)`。即豁免仅当"从未持仓且当前空仓"时授予；授予并实际建仓（执行后书非零）即翻 True；首建执行完全失败（书仍空）不烧豁免（下次可重试）；**清仓不重置**（True 吸收）。
- **谁持有**（map §2 结论，冻结）：
  1. 回测引擎：run 级变量，与 weights/shares/cash 并列（backtest.py:238-241 初始化处）。
  2. 纸面模拟 run_sim：随组合状态持久化（state_path，config.py:469；恢复语义对齐 run_sim.py:472-486），重启不丢。
  3. reward_baskets：每 (t, b) 行一个标志（一实例并行服务 b 套账本，构造器自身**不得**持有状态——实例可被注入/复用，backtest.py:81-88、map §2）。
- **诊断**：传入 kwarg 时 `diagnostics["initial_funding_used"] = b`（回显）；缺省时不新增该键。

---

## §7 验收规则与实现边界
1. p51 测试断言只准引用本文件 §编号（docstring 注明契约 §，仓库惯例）；断言不得照实现输出重写（W5）。场景 1/2/3 的全部数字（20 期表、991/99,991、第二期 983/99,983、分支诊断）为本文件冻结值。
2. 实现顺序红线：先冻结（本文）→ RED 证据 `reports/p51_red_evidence.txt`（逐字捕获失败命令+退出码）→ 实现 → GREEN；全量回归 `cd D:/minequant/AlphaGPT && python -m pytest tests/ -q` 必跑（map §7 基线：57 passed 于构造器四件套）。
3. 默认行为不变的验收面：`turnover_allocation ∈ {proportional, priority}`（含缺省）与 `initial_funding_used is None` 的全部现有路径不变；`configs/r1_slow_ab.yaml` 仅加法式补 `turnover_allocation: priority`。
4. 产出去向：W6 的 `reports/wp15/config_feasibility.json` + 报告章节采用 §5 冻结公式、判定与措辞。
5. 若实现者选择另一种确定性分配规则：必须先另立冻结推导（新文档），本文件按红线 3 只可追加勘误附注，不得删改。

---

*冻结完毕。本文件之后的一切差异，先怀疑实现。*

---

## 勘误/裁决附注（2026-09-22，实现落地时追加；原文以上未改动）

**§1.1.5 落地方式裁决**：本文件要求给 `configs/r1_slow_ab.yaml` 加法式补
`turnover_allocation: priority`，与 WP14（p50）实验完整性机制冲突——p50 会从
磁盘 yaml 重新物化 plan 并与冻结真实 run 报告的 `provenance.config_sha256`
比对，任何编辑（即使语义空操作）触发 `IntegrityRefusal`（实测：加键后 10 个
`test_p50_wp14_*` 失败，report='3D20EFDC…' vs plan='704212e2…'；撤销后
70/70 全绿）。经裁决（2026-09-22）：**撤销补键、参考配置保持字节原样**——
WP14 的字节级溯源（P45 教训）优先，历史不可变性高于显式化要求；§1.1.5 的
防默认漂移意图由面向未来的规则承载：新实验计划/新配置必须显式写
allocation 模式键（`docs/p51_trade_groups_draft.md` §2），代码层 opt-in 枚举
与默认语义不变。证据：`reports/p51_red_evidence.txt` 追加条目（只追加）。

**§1.4 反例数值注记（待审核确认的候选勘误，未改动原文）**：§1.4 合规性说
明的冻结示例 "C_g=3,000、B_r=10,000、零费、买需求 10,000、价格 10、L=100
→ y*=6,500（650 股）、x=3,500（350 股）" 与本文件自身的规范性整手规则
（§1.4 步 2："y 取整手格点 n·L·p_j（n≥1 整数）"）不一致——650/350 股非整
手。按规范算法推导该例应为 y*=6,000（600 股）、x=3,000（300 股）；实现遵
循规范算法（§2/§3/§4 三个冻结场景的全部数字与整手规则一致，不受影响）。
