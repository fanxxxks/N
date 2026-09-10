# P26 勘误 1：A02 复合身份覆盖面声明与实现的归一（composite coverage erratum）

- 状态：**APPROVED（captain 依用户转授批准；批准决策记录
  `reports/alphagpt-ab/00_authority_exception_record.md` §3.2/§4ter——
  「所有行为我强制要求你来替我批准」+「D：以上全部，凡我代批的都确认」）**
- 日期：2026-09-11（Asia/Shanghai）
- 勘误对象：`docs/p26_a02_dataset_snapshot_identity_draft.md` §3「本轮明确
  不纳入的域」表的**声明语义**；不改写该表的历史原文（I9：历史不改写），
  本勘误是其后置澄清，效力高于原表述中与实现矛盾的读法。
- 触发：A02 合规双审（t13 半）阻断项 B2——探针证明向 `v2
  holder_number_pit`（原「未纳入」清单成员）写入一行即旋转复合身份并
  拒绝正式运行，而机器可读的 `NOT_INCLUDED_DOMAINS` 声明了相反事实
  （AGENTS §11.3/§11.4 一致性义务）。

## 1. 事实（实现与证据）

1. 自 `1ea4405` 起，复合身份的 v2 成员是**完整 v2 数据集 manifest**
   （`_V2_PARTITION_SPEC` 全部 19 张表）。因此四个事件域
   （`earnings_forecast_pit`/`earnings_express_pit`/`repurchase_event_pit`/
   `holder_number_pit`）与 `index_weight_pit` 的**任何内容变化都会旋转
   复合身份**并使正式运行被拒绝（fail-closed）。
2. 合规审查探针（2026-09-11，临时库）：向 `holder_number_pit` 插入一行
   → `confirm_composite_snapshot` 拒绝并指名 `holder_number_pit`
   drifted。该行为是**实现事实**，且在 fail-closed 方向安全（不产生
   身份空洞，p26 I5 不受影响）。
3. p26 §3「不纳入」表对主库三表（`fundamental_pit`/`margin_balance`/
   `sw_industry_member`）的「正式输入已切 v2」表述对 `fundamental_pit`
   **事实错误**：loader 的 `build_pit_frames → _fetch_fundamental_rows`
   仍读主库该表（`ashare_data/fundamentals.py:294-312`），经典财务族
   因子消费它。故「按 p26 §3 原表做域受限 v2 成员」的方案 (b) 需要
   另行裁决覆盖面，不能直接执行。

## 2. 裁决（captain 依转授，2026-09-11）

采用合规审查建议的方案 (a)——**裁决式勘误**，不回退任何已交付提交：

1. **两个概念显式分离**（取代原「未纳入」单一清单的歧义）：
   - **身份覆盖（identity coverage）**：复合身份绑定完整 v2 数据集
     manifest + 主库 8 表（`COMPOSITE_DOMAINS`，即超集覆盖）。p26 I3
     的措辞本就是「至少覆盖」，超集满足其字面与 fail-closed 方向。
   - **研究准入（research admission）**：事件域与 `index_weight_pit`
     的**研究准入**仍待 F3（p25）与 B01 裁定（captain 裁决 3 的本意：
     不替未来研究预判结论）。未准入域的因子不得进入正式研究；其
     数据变化旋转身份是**预期的 fail-closed 行为**。
2. **R15 的处置**：R15「未走 I3-EXT 扩展契约的域不得进入身份」约束的是
   **身份表范围的变更**（增删指纹表），该约束维持不变——任何改变
   `COMPOSITE_DOMAINS` / v2 成员表集的变更仍须 I3-EXT 勘误 + bump
   `SNAPSHOT_PUBLICATION_VERSION`。**研究准入决定不改变身份表范围**，
   不触发该门（准入是研究流程状态，不是身份算法定义）。
   事件域自 `1ea4405` 起事实进入身份覆盖，本勘误将其**显式合法化**并
   记录于案（而非静默维持一个与声明矛盾的实现）。
3. **机器可读声明修正**：`NOT_INCLUDED_DOMAINS` 更名为
   `PENDING_ADMISSION_DOMAINS`，`composite_snapshot_domains()` 的键
   `not_included` 更名为 `pending_admission`，语义如上。同步修正
   `tests/test_manifest.py::test_a02_composite_domain_coverage_is_explicit`
   并新增 `test_a02_pending_admission_domain_is_identity_covered`（把
   探针行为钉为规范：未准入域的内容变化旋转身份、拒绝正式运行、指名
   漂移表）。
4. **操作性代价的接受**：B01/F3 期间对事件域的数据工作会旋转身份，
   需要新的复合代际发布才能恢复正式研究。接受该代价的理由：(i) 它是
   fail-closed 的（宁可重发一次代际，不可让未准入数据静默流入已
   冻结身份）；(ii) 每次发布本就应在新数据落库后进行（发布边界的
   本意）；(iii) 替代方案 (b)（域受限 v2 成员）被 §1.3 的事实错误
   阻塞，需要独立裁决，不应在 t46 前仓促执行。

## 3. 对 p26 的其余影响

- 无。I1–I12 的其余条款、版本链（`MANIFEST_VERSION="2"`、
  `SNAPSHOT_PUBLICATION_VERSION="1"` 等）均不受本勘误影响。
- 本勘误不改写 p26 原文（历史保留）；引用优先级：本勘误 > p26 §3
  「不纳入」表的旧表述。

## 4. 撤销方式

用户可随时撤销本勘误（撤销方式 = 回退本勘误文件与对应实现提交中的
声明部分 + 不采用对应实现 commit）。
