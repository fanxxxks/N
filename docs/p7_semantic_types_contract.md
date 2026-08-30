# P7-E GP/搜索器语义类型契约（研究语义变更，预注册）

状态：预注册（实现前）。本文是 Phase E 测试断言与验收的唯一来源；仲裁顺序
为本文 > 测试 > 实现。按 AGENTS.md §3.2，本文先于任何实现提交。

## 0. 问题陈述

现有"强类型 GP"实际上是单类型系统：`gp_search.Signal` 是唯一类型 token，
所有特征终端与所有算子共享之。因此语法合法但经济上无意义的公式大量存在：
事件信号参与除法、行业中性化施加于价格量、相关性作用于事件序列等。RL 策略
采样、随机基线与 TPE 共享的 `build_action_mask` 只做栈合法性（arity/可终
止性）检查，同样不区分钟类。这浪费唯一语义评价预算，并稀释搜索器的有效
搜索空间（AGENTS.md §6：优先增加语义类型、模板限制、去重和剪枝）。

## 1. 假设与类型格

类型格 = P7 D1 注册表的六类 `SemanticType`（price_like / return_like /
volume_like / fundamental_like / cross_sectional_signal /
boolean_event_signal）。特征终端的类型 = 该特征的 `semantic_type`（单一来
源：FEATURE_METADATA；禁建第二份）。算子的输入/输出语义 = P7 D2
`OPERATOR_REGISTRY`（单一来源）加上本文对 D2 留白（`output=None`）的裁决：

| 算子（族） | 输入约束 | 输出 |
|---|---|---|
| ADD/SUB/MUL/NEG/ABS/SIGN | 全部参数**同型** T，T ∈ 五个连续类（排除 boolean_event） | T（族保持） |
| DIV | 两参数 ∈ 五个连续类，**可异型**（Amihud 类跨族除法合法） | cross_sectional_signal（比值=相对量） |
| GATE | 条件位任意（事件作条件是其本职）；x/y 同型 T（五个连续类） | T |
| MA5/10/20/60、TS_RANK5/10/20/60、DELAY1、MAX3、DECAY | 单参数 ∈ 五个连续类 | T（族保持：TS_RANK 保族只重标度） |
| STD5/10/20/60、DOWNVOL5/10/20/60、DELTA5/10/20 | 单参数 ∈ 五个连续类 | return_like（波动/差分尺度） |
| CORR5/10/20/60 | 两参数 ∈ 五个连续类，可异型 | cross_sectional_signal（D2 留白裁决） |
| CS_RANK/CS_ZSCORE/CS_DEMEAN | 单参数 ∈ 五个连续类 | cross_sectional_signal |
| CS_NEUTRALIZE | 单参数 **= cross_sectional_signal**（行业中性化只对横截面信号有意义） | cross_sectional_signal |
| JUMP | 单参数 ∈ 五个连续类 | boolean_event_signal（稀疏事件检测器） |

## 2. 范围与非目标

范围：四个搜索采样路径的合法性收紧——(a) DEAP GP 的 pset 按签名注册；
(b) `build_action_mask` 类型感知（栈携带类型）；(c) 随机基线与 TPE 因共用
掩码自动收紧；(d) 共享的 token 序列类型校验器
（`ashare_model/semantic_sampling.py`）作为唯一判定实现。

非目标：VM 执行、指标、Reward、成本、canonicalization、semantic cache
指纹、公式长度上限均不变；不引入第六类之外的新类型；不进一步收窄算术混
合（如 MUL 跨族）——留给后续契约；不修改既有产物的解读。

## 3. 不变量（属性测试直接断言）

1. `len(tree_to_tokens(tree)) == len(tree)` 与统一 token 长度上限（§7 既有
   不变量）在类型化 pset 下保持。
2. 三个掩码路径（RL/Random/TPE）采样出的每条公式都通过共享校验器
   `formula_types_legal(tokens)`；GP 树映射的 token 序列同样通过。
3. 掩码只依赖 (栈类型状态, done, step, max_len, vocab, feature_ids) 与注册
   表——确定性；同 seed 同序列。
4. `formula_types_legal` 与采样掩码读同一 `OPERATOR_REGISTRY`（测试通过
   monkeypatch 签名改变校验结果，证明无第二份硬编码类型表）。
5. EOS/PAD 规则不变（EOS 仅 stack==1；PAD 仅 done 后）。
6. 词表 feature_version 因 grammar 进入 hash 而改变；公式按名解析不受影响。

## 4. 版本变化

| 版本 | 变化 | 理由 |
|---|---|---|
| `GRAMMAR_VERSION` 2→3 | 采样合法性（类型感知掩码） | 词表 hash 变化；旧 artifact 按名解析仍合法 |
| `SEARCH_CONTRACT_VERSION` 1→2 | 有效搜索空间（合法公式集）收紧 | SearchResult 的 contract_version 记录可比性 |
| `PROTOCOL_VERSION` "24"→"25" | 候选池来源变化（先例：v6 词表扩张即 bump） | 产物可比性声明 |
| 不 bump：`REWARD_VERSION`、`MODEL_VERSION`、`FEATURE_REGISTRY_VERSION`、`DATA_TIER_VERSION`、`EXECUTION_SPEC_VERSION` | — | 评分/模型/注册表语义不变 |

## 5. 迁移/拒绝策略

- 既有 formula/策略/协议产物**不解码失败**：`resolve_formula_tokens` 按名
  解析不变；类型违规的旧公式仍可被 VM 执行（执行语义不动），但按
  `artifact_versions` 既有规则，grammar/protocol 版本不匹配会使其被分类为
  legacy——fail-closed，不冒充 current。
- 新采样严格 fail-closed：特征缺元数据、算子缺签名 → 构表/构掩码即
  `ValueError`，不允许静默回退到无类型采样。
- 配置兼容：无新配置项；类型合法性不可经配置关闭（研究语义不设开关）。

## 6. 预期 RED 测试（先写后实现）

1. 校验器：每类违规各一例（事件入除法、事件入 MA、CS_NEUTRALIZE 作用于
   价格量、GATE 异型分支、CORR 事件参数）→ `formula_types_legal == False`；
   合法例（GATE 事件条件、CORR 异型连续、DIV 跨族）→ True。
2. GP：类型化 pset 生成的 N=200 棵树（固定 seed）全部通过校验器；
   `tree_to_tokens` 长度不变量保持；`feature_ids` 域限制仍生效。
3. 掩码：Random/TPE 路径采样 N=200 条（固定 seed）全部通过校验器；
   toy 词表（真实特征名子集）可用；未知特征名 → `ValueError`。
4. legacy：构造一条类型违规的旧公式 payload（含旧 grammar_version）→
   `resolve_formula_tokens` 成功、VM 可执行；同时其 protocol_version 不匹配
   使 `artifact_versions` 判 legacy。
5. 版本钉死：GRAMMAR_VERSION==3、SEARCH_CONTRACT_VERSION==2、
   PROTOCOL_VERSION=="25"（引用本契约的合法钉死修订）。
6. 确定性：同 seed 两次采样序列相同（掩码路径与 GP 各一）。

## 7. 测量方案、资源上限与裁决

- 验证：新增 `tests/test_semantic_sampling.py`；聚焦
  test_gp_search/test_tpe_search/test_grammar/test_vocab/test_train/
  test_searcher_bench/test_alphagpt/test_evaluation；全量 pytest +
  compileall + `git diff --check`。
- 不变量：全量 passed 相对 Phase D 基线（1181/5/618）只增不减；
  warnings 不增；wallclock 无显著回归（训练循环新增类型维护为 O(1)/步）。
- 资源上限：不新增长时运行；类型化 pset 的搜索行为验证用小预算
  engineering 运行（semantically-labeled，不得作为研究证据）。
- 裁决：类型化前后的搜索结果**不得宣称 matched comparison**；本阶段不产
  生任何 alpha/晋级结论。测量只回答"合法空间收缩与不变量保持"。
- 停止条件：(a) DEAP 类型化机制与本文 §1 签名注册冲突且无等价机制；
  (b) 类型化使某搜索路径无法在合理步数内产出合法公式（空间过窄，需回本
  契约修订格而不是放宽实现）；(c) 全量测试出现无法归因于本契约的回归。
