# P16 数据新鲜度门禁（G8）预注册契约

状态：已批准、已实现（IP-05a 预注册 → IP-05 实现落地）。立项依据：用户已批准的
`AlphaGPT-improvement-plan-20260903.md` IP-05（[03-F-03]、[01-A5]、[05-⑦]）。
批准状态：**已获用户治理裁决正式批准（2026-09-03）**。实现时序（git 可查）：
预注册 commit `1745773`（2026-09-03 11:01 +08:00）→ 用户治理批准广播 →
t13 实现 commit `50d0753`（15:03 +08:00，G8 门禁 + 六项同 commit 对齐）→
合并 `2fb4910`（15:21 +08:00，improve/data-infra → main）→ t20 全量首次
组合验证——实现未先于批准启动，时序合规。仲裁顺序：本文 > 测试 > 实现。
本契约本身不写任何代码、不 bump 任何既有版本常量；它为 G8 的唯一实现固定
机器标准。

术语约定：本文所称门禁编号属**数据资格空间**（`ashare_data/gates.py` 的
`ProductionGateRunner`），与 `ashare_model/promotion.py` 的晋级门禁空间
（G1–G7，`PROMOTION_RULE_VERSION="2"`）相互独立、禁止混用。G8 落地后数据
资格空间为 G1–G8；激活前（本契约未实现）数据资格空间仍为 G1–G7。

## 0. 问题陈述（逐项经代码勘察与实测证据确认）

1. **无新鲜度门禁**：G1–G7 全部只审计"库内一致性"（成员区间、上市日、
   日历覆盖日线窗口、区间结构、PIT 契约、eligible 下限、零 bar 存续），
   没有任何门禁把 daily_bar 的最大交易日与"最近已收盘开市日"比对。
   实测证据：`docs/p15_data_freshness_measurement_log.md` §2——2026-09-02
   同步前 daily_bar 停在 20260821，落后日历口径 **8 个开市日**，而
   research_doctor 与 formal G1–G7 **全部 PASS**（该日志 §2 门禁基线行）。
   陈旧数据可在全绿门禁下进入正式运行。
2. **AGENTS §5.5 无落点**："数据 readiness 必须检查最近开市日、缺失交易日、
   覆盖率和 manifest，而不是只检查表存在"——缺失交易日由 p8 §5.1 的
   `missing_trading_days` 承接，"最近开市日"维度至今无机器检查。
3. **交易日历可被伪造**：`ashare_data/akshare_client.py:242-246` 在抓取
   失败（或空结果）时 `except Exception` 后静默回退
   `pd.bdate_range`（周一–周五），仅 warning；`ashare_data/sync.py:86` 把
   返回的所有日期写为 `is_open=True`。一次网络故障可让"周一至周五"冒充
   交易日历持久化入库，G3"日历覆盖日线窗口"恒真，未来任何以日历为基准的
   检查（含 G8）都被污染。

后果：数据新鲜度这一 readiness 维度只能靠人工比对，与 §1 原则 4（可复现）
和 §4.3（fail closed）相悖。本契约把"最近开市日"维度换成机器裁决，并把
日历来源的可信性从"静默兜底"改为"显式失败"。

## 1. 假设、范围与非目标

**假设**：`trade_calendar` 表由 sync 从真实交易日历源刷新，预列至配置
`end_date`（实测 19901219…20261231）；`daily_bar` 逐日 upsert；两者
date_range 已在库内，G8 只消费不新采。

**范围**：G8 门禁的精确机器标准（参考日、容差、数据来源、formal/dev
裁决语义）；交易日历在线获取失败路径的 fail-closed 不变量；G8 落地时
受影响契约文本与测试的**同 commit 对齐义务**（§4）。

**非目标**：不新增数据表、数据源、依赖或第二套日历实现；不改变 G1–G7
任何判定逻辑；不修改 p8 §5.1 `missing_trading_days`（spec 窗口内的
expected vs actual 比对，与 G8 的"now 对比"正交）；不实现 broker 级
盘中新鲜度；不动 webui/dashboard 展示；不做补同步（数据运维仍走独立
任务，如 p15 日志记录的窗口）。

## 2. G8 机器标准（唯一实现：`ashare_data/gates.py`）

### 2.1 定义

- **评估日 `today`**：`Asia/Shanghai` 时区当次 `run()` 调用时的本地日期，
  一次 run 内取一次并写入 G8 detail（可审计）。runner 增加仅测试用的
  显式注入参数（默认墙钟）；生产调用方不传。禁止用配置项覆盖评估日
  ——那是 bypass，不是注入。
- **参考会话 `reference`**：`trade_calendar` 中 `is_open=true` 且
  `trade_date < today` 的最大 `trade_date`。严格早于 today：当日节
  （无论是否已收盘）不计入预期，与 p15 日志"至最近已完成开市日"口径
  一致且对盘后清算窗口保守。
- **滞后 `lag`**：`trade_calendar` 中开市会话落在 `(daily.mx, reference]`
  的个数（按 open session 计数，**禁止按日历日计数**——长假会造成
  误杀，此为硬性口径）。
- **容差 `N`**：`FRESHNESS_TOLERANCE_SESSIONS = 3`（module 常量，
  constructor 覆盖 knob，与 `min_eligible` 同一"部署调参、永不 bypass"
  定位；默认 3 = 容忍一次错过的隔夜同步周期，捕捉 p15 的 8 会话漂移）。

### 2.2 判定

- `ok = (lag <= N)`；`detail` 至少含 `daily.mx`、`reference`、`lag`、
  `N`、`today`（如 `daily 20260821 vs reference 20260901, lag=8 > 3,
  today=20260902`）。check 名以 `G8` 开头（生产者命名约定
  `^G8\b`，供 lifecycle 映射，见 §4.2）。
- **fail-closed 边界**：daily 表无行、`reference` 不存在（日历无
  `is_open=true` 且早于 today 的会话）、或日历预列视界耗尽
  （calendar 最大会话 < today 且无可用参考）⇒ G8 fail，detail 写明
  原因。任何"无法计算"不得折算为 pass。
- **formal**：G8 fail 计入 `result.checks`，`require_production()` 按
   既有语义 raise，拒绝一切正式入口；全部正式入口经该单点自动获得
   G8 执法，无需逐入口改动。
- **dev**：同一计算；fail 体现为 check fail + `degraded=True`，绝不
  raise（沿用 `run()` 既有语义）；dev 结果永远不可支持正式结论。
- **无开关**：不存在关闭 G8 的配置/flag；新增此类开关即契约违反
  （与 p8 §5.3 第 5 条同构）。

### 2.3 与既有门禁的关系

G3 证明"日历 ⊇ 日线窗口"（库内一致），G8 证明"日线窗口贴近 now"
（库外时序）——正交且互补。G8 不替代 G3，也不替代 p8 §5.1 的
`missing_trading_days`。G8 数据来源仅为：`daily_bar` max(trade_date)、
`trade_calendar` is_open 会话、评估日墙钟。禁止在门禁路径内发起网络
抓取或读 live 源。

## 3. 交易日历回退 fail-closed（工程变更，随实现 PR 落地）

不变量：**在线路径**（`akshare_client.get_trade_calendar` 非 offline
分支）抓取失败或返回空 ⇒ 显式失败（raise，由 sync 终止日历步骤），
**禁止** `bdate_range` 结果冒充交易日历返回或入库；离线 fixture 路径
（`offline=True`）保持现状，且文档标明仅 fixture。sync 写入路径不得
再出现"静默伪造日历"分支。该修复为工程变更（§3.1）：正常路径
golden/parity 证明逐字节不变（成功抓取时输出不变），RED 先行。

## 4. 跨契约与文档对齐义务（G8 实现 PR 内**同一原子语义提交**完成）

实现 PR 不得只改 `gates.py`；以下漂移点必须在同一 commit 内对齐，
否则形成互相漂移的第二份门禁清单（AGENTS §10.2）：

1. **`ashare_model/lifecycle.py`**：`_gate_check_names` 的
   `G([1-7])` 与 `_validate_data_report`/`build_data_qualification_report`
   的 `range(1, 8)` 完备性判定扩展为 G1–G8。现状是 **fail-open**：
   G8 落地后其 fail 会被 lifecycle verdict 静默忽略（正则不匹配）。
   `report_schema_version` 保持 1：payload 形状不变、完备性规则由本
   契约预注册、且 lifecycle v1 激活前无任何存量
   DataQualificationReport 需迁移。
2. **p8 契约文本**（`docs/p8_research_lifecycle_contract.md`）：§0
   术语约定、§3 `DATA_QUALIFIED` 证据行、§5.1 各处的"数据资格
   G1–G7"更新为 G1–G8（注明"G8 由 p16 引入"）。
3. **`tests/test_p8_lifecycle_contract_doc.py`**：t2 修订新增的
   `数据资格 G1–G7` 锚点同步为 G1–G8（§10.1 白名单第 2 类，引用本
   契约为批准依据；断言强度不降）。
4. **AGENTS.md §5.4**："G1–G7 数据资格门禁"一行更新为 G1–G8（单行
   改动，≤36 KiB 预算不受威胁；与 IP-06 的 AGENTS 指针修订不同
   hunk，按合并顺序错峰）。
5. **`scripts/check_production_gates.py`** docstring 门禁清单补 G8；
   **`ashare_model/research_doctor.py`** 门禁节 docstring 同步（doctor
   消费 runner 输出的行为不变——单一实现，无第二套新鲜度判定）。
6. **测试 fixture**：`tests/test_run_store.py:396`（七门 fixture）与
   `tests/test_research_doctor.py:54,263`（G1..G7 构造）按新完备性
   规则扩展至八门（§10.1 白名单第 2 类，引用本契约）。

## 5. 版本影响

无任何 `*_VERSION` bump：G8 是 `ProductionGateRunner` 运行时检查项的
扩展，经既有类型化通道（`GateCheck`/`GateResult`/
`DataQualificationReport`）消费，不新增 artifact schema 字段、不改变
`DATA_TIER_VERSION`/`MANIFEST_VERSION` 等任何版本所有者语义；
`report_schema_version` 维持 1（理由见 §4.1）。GateResult 兼容性由
既有 `check(name)`/`checks` 消费面保证（追加式，无破坏）。

## 6. 预期 RED 测试（实现 PR 先行举证，引用本契约为 oracle）

1. `tests/test_gates.py`：陈旧 fixture（daily 停在 reference − 8
   会话）⇒ formal `result.ok is False` 且 G8 fail、detail 含 lag；
   新鲜 fixture ⇒ G8 pass；dev + 陈旧 ⇒ `degraded=True` 且不 raise。
2. 容差 knob：`N=1` 与 `N=8` 下同一 fixture 判定翻转/通过，证明容差
   语义按 open session 计数（含跨长假夹具）。
3. 无 bypass：断言 runner 无任何禁用 G8 的配置面（源级检查）。
4. `tests/test_run_store.py`/lifecycle：G8 fail 的
   DataQualificationReport ⇒ `DataQualificationError`；八门完备性
   通过、七门报告被拒（新完备性规则）。
5. `tests/test_akshare_client.py`（或同域测试文件）：在线日历抓取
   失败 ⇒ 不返回 `bdate_range`（显式异常）；成功路径输出与修复前
   逐字节一致（parity 证据）。

## 7. 验证、资源与停止条件

验证命令（实现 PR 在精确候选 commit 上运行）：聚焦
`python -m pytest -q tests/test_gates.py tests/test_run_store.py
tests/test_research_doctor.py` + 日历回退聚焦测试；`python -m
compileall` 相关包；`git diff --check`；全量回归由统一验证窗口执行。
资源：不新增运行、不新增依赖；测试用 tmp DB fixture，注入评估日，
禁止依赖真实墙钟。

**停止条件**：(a) 参考会话/滞后无法用上述机器可查标准表达；(b) G8
与 G3/`missing_trading_days` 发现职责不可调和重叠；(c) 日历源无法
在不伪造的前提下提供 `is_open` 会话表。

## 8. Retirement

本契约不退役任何代码路径。前瞻义务：`bdate_range` 静默回退分支在
§3 落地后即退出现网路径（其删除属实现 PR 范围，fixture 路径除外，
保留原因=离线测试唯一日历来源）。
