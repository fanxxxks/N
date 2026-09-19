# P40 + P41 测量日志（工作包 5A/5B：统计重采样修正 + 候选与统计证据绑定）

按 §11 只记录实际发生的事。研究语义依据：`docs/p40_statistical_resampling_draft.md`
（含勘误 1/2）与 `docs/p41_candidate_evidence_binding_draft.md`（含勘误 1）。
**本批次不产生任何 alpha/晋级研究结论**——软件行为验证与校准测量而已。

## 0 环境与基线

- 分支 `codex/b04-load-cache-performance`；基线 `25f9983`（p39 收口）。
- Windows 10 22621 x64，Git Bash；Python 依仓库 venv（`python -m pytest`）。
- 数据：仅合成测试夹具（populated_db，2024-01-02..2024-02-23 的 40 会话
  业务日历）与解析合成矩阵；**未触碰生产数据库、未运行任何正式研究入口**。
- 提交序列：`7c5de01`（p40 契约）→ `298a763`（p40 RED）→ `91278d6`
  （p40 GREEN）→ `0c6d6b5`（p41 契约）→ `2f633c9`（p41 RED）→
  p41 GREEN（本文件同 commit，SHA 见交付段）。

## 1 P40 测量（统计重采样）

### 1.1 实现前诊断（旧实现拒绝率，R=240、n_perms=200、α=0.05、n=250）

| 体制 | 旧实现拒绝率 | 判定 |
|---|---|---|
| 相关噪声 载荷0.95 K=8 | 0.0625 | 带内（旧实现尺寸扭曲温和，起草预期"坍缩"被证伪 → 契约勘误 1） |
| 相关噪声 载荷0.95 K=16 | 0.0500 | 带内（同上） |
| 相关噪声 载荷0.95 K=40 | 0.0292 | 带内（同上） |
| 相关噪声 载荷0.99 K=16 | 0.0333 | 带内（同上） |
| 重复候选 3×3 K=9 | 0.0208 | 带内（同上；区分判据是 N2 精确不变量） |
| 短序列 n=6 K=4 | **0.9333** | 旧反保守退化实证（p=1/(n_perms+1)） |

### 1.2 实现期测量（联合抽取设计参数冻结依据）

- b = min(block_size, n//2) 规则下 n=6/b=3：12 个固定 seed 中 5 个
  p=1/501——任何合法重采样在该尺度不可校准 → 改为比率规则 + 30 日期底线
  fail-closed（契约勘误 2）。
- n/b=2 尺度（n=20..24, b=10）：0.13–0.24——比率规则（n//6）依据。
- 新实现尺寸（b = min(10, n//6)，n_perms=200，R=240）：

| 体制 | n | K | 拒绝率 | 带宽 |
|---|---|---|---|---|
| iid | 250 | 8 | 0.0583（K=4: 0.0792） | [0.010,0.115] ✓ |
| 相关 0.95 | 250 | 8/16 | 0.0917 / 0.0417 | ✓ |
| AR(1) φ=0.4 | 250 | 8 | 0.0792 | ✓ |
| 重复 3×3 | 250 | 9 | 0.0583 | ✓ |
| iid（=底线） | 30 | 4 | 0.0958–0.1083 | ≤0.18 ✓ |
| 相关 0.95 | 30 | 4 | 0.0750–0.1208 | ≤0.18 ✓ |
| 功效（1 真信号 + 7 噪声） | 250 | 8 | 1.000（R=120） | ≥0.90 ✓ |

- N2 精确不变量：复制列 p 值逐位相等（联合抽取）；旧实现必不等（RED 实证）。

### 1.3 验证（命令与结果）

- RED：`pytest -q tests/test_p40_statistical_resampling.py
  tests/test_eval_resampling_calibration.py` → **13 failed**（全部为 p40
  契约缺口；证据 `D:\minequant\reports\p40_red_evidence.txt`，仓库外证据目录，
  p37–p39 先例）。
- GREEN 聚焦：同命令 → **14 passed**（N4/C4 按勘误 2 重设计后 9+5）。
- 相邻：test_evaluation 51P、test_stitched_oos 20P、promotion+module_split+
  p34 批 75P、schemas/versions/research_domain/p27/baseline/candidates/ir
  批 76P；p41 落地后 p40+校准复跑 14P。

## 2 P41 测量（候选与统计证据绑定）

- RED：`pytest -q tests/test_p41_candidate_evidence_binding.py` →
  **8 failed**（证据 `D:\minequant\reports\p41_red_evidence.txt`）。
- GREEN 聚焦：同命令 → **8 passed**。
- J2 实测：本次 3 噪声 + 1 历史强试验 → `adjudicated_candidate` ∈ 本次、
  n_trials=4、含历史时 dsr 低于无历史（多重折减）、
  `extra_outranks_current=true`。
- J3 实测：历史强试验（W 覆盖）下 max-t `significant_95=False` 且被裁决候选
  为本次候选——历史显著性不再被授予。
- 相邻：test_stitched_oos 22P（补零钉住由 B1 契约替换为两条显式失败断言）、
  test_promotion 42P（版本 pin 2→3）、module_split 12P、artifact_versions
  批、p27/research_domain/artifact_schemas 批共 145P。
- 版本：PROTOCOL_VERSION 28→29（p41 §4）、PROMOTION_RULE_VERSION 2→3；
  v≤28 artifact 对 promotion 走 G1 版本拒绝 + G2 身份缺失拒绝（双保险）。

## 3 全量门禁（精确候选 commit 上）

- `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio
  ashare_trading scripts webapi` → 干净。
- `python scripts/check_test_shards.py --check` → 116 文件 / 4 分片，并集=全集。
- `git diff --check` → 干净。
- `python -m pytest -q tests -n auto`（本地并行全量）→ 第一轮
  **2037 passed / 4 failed / 6 skipped / 3 xfailed**（17:23）——4 个失败全部
  是 PROTOCOL_VERSION pin 未随 29 更新（test_p4_search_contract /
  test_price_semantics / test_semantic_sampling / test_b03 评估 CLI 折窗口
  低于 p40 底线），逐一按 §10.1 规则 2 以 p40/p41 §4 为依据更新后，四个
  文件复跑 78P；第二轮全量（精确候选工作树）**2041 passed / 0 failed /
  6 skipped / 3 xfailed**（17:05）。基线（p39 收口时）为 2010P/0F——本批次
  净增 31 个测试（p40 14 + p41 8 + 补零钉住替换净增 1 + 夹具长度参数化等），
  skip/xfail 数量无变化。

## 4 未运行项与声明边界

- 未运行：生产数据入口（train/evaluation/backtest/r01/r02 正式运行）、
  `scripts/check_production_gates.py`、`research_doctor`、web 构建——本批次
  零数据/universe 变更，非目标。
- 并行 vs 串行对账：CI 前提项，本批次未 push，按 §10.2 全量为本地并行；
  push 前需完成 parallel-vs-serial parity 记账。
- 研究声明：本批次全部结果为软件正确性与统计方法校准证据；**不构成任何
  alpha、晋级或正式统计结论**。旧（v≤28）artifact 的统计块按新规则不再可
  晋级——历史结论需在新协议版本下重测方可引用。
- Mimosa 钩子在多次 commit 前报告 scanner_enobufs（未获完整扫描结论）；
  按其提示不做"项目安全"声明，建议择机重跑完整审计。
