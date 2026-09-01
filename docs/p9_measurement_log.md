# P9 测量日志（实际发生什么——与契约分离）

- 运行：t5 精简后特征全集最终审计（v4 词表全量重测），2026-09-01
- 被测实现：commit **781f5e166c4513da274bbb846b8de9330629afe2**（codex/p9-factor-families，含 P9 实现与合规收尾）＋ 同分支裁决后提交（见本文末"裁决后提交"）
- 环境：Windows / Python 3.13.12 / duckdb 1.5.5 / pandas 3.0.5 / numpy 2.5.2；PYTHONDONTWRITEBYTECODE=1；本地串行
- 数据：`data/ashare.duckdb`，dataset_id = `a839ecf2284b354a5ab6ed3228d13fc5d7f3d93a2fadba0b08d8c909edf194fd`（与 t2 完全一致，未重同步，data_end=20260821——用户选项 A）
- 窗口/口径：与 t2 逐项一致（2015-01-05..20260821；1630 股 × 2828 日；h∈{1,2,3,5,10,15,20}；min_stocks=10；IS<20220101 / OOS≥20220101；成本模型=cost_matrix.round_trip_cost@10 万/20 仓）
- 脚本：`docs/factor_inventory_audit_v4_20260901/audit_run_v4.py`（t2 脚本的同口径适配：73 特征断言、provenance 增加 grammar/factor_compute 版本）
- 产物：`docs/factor_inventory_audit_v4_20260901/metrics.json`（73 因子 × 7 horizon × 逐年 + 73×73 相关矩阵）；运行一次成功，runtime 1325.8s，无失败产物
- 决定性版本组：protocol 25 / reward 14 / model 3 / **grammar 4（重测时点）→ 5（裁决后提交）** / **factor_compute 1** / **feature_registry 4（重测时点）→ 5（裁决后提交）** / research_domain 2 / 其余不变

## 1. 无回归校验（合并族无信息丢失的实证）

- 57 个非稀疏特征（t2 全集 − 5 个稀疏事件特征）的 IC(h=1/10/20) 与覆盖率与 t2 **逐位一致**（容差 1e-9，0 失配）——v4 实现对非稀疏路径零影响，deprecated 稠密特征（RET_5 等）的信息完整保留（仍计算、仍可解析）。
- 稀疏事件特征（LIMIT_*）因 F1 修复（稀疏安全标准化）指标变化——这正是修复目的，见 §2。

## 2. 契约 §7 逐项裁决（预注册规则执行）

### 2.1 F1 覆盖率硬门槛（族③前置）

| 特征 | 非退化天数(t2→v4) | 最少年度天数 | 门槛 |
|---|---|---|---|
| LIMIT_UP_EVENT | 19 → **719** | 21 | **PASS**（≥400 且逐年 ≥20） |
| LIMIT_UP_CNT_20 | 537 → 2566 | 146 | PASS |
| LIMIT_BREAK | 122 → 2003 | 111 | PASS |
| LIMIT_UP_CNT_5（新） | — → 1683 | 69 | PASS |
| LIMIT_BREAK_5（新） | — → 2797 | 152 | PASS |
| LIMIT_DOWN_EVENT | 8 → 392 | **4** | 年度覆盖不足（诚实记录，见 2.3 备注） |
| LIMIT_DOWN_STREAK（新） | — → 392 | **4** | 年度覆盖不足 |

族③前置硬门槛（以 LIMIT_UP_EVENT 为判据）**PASS**，族③进入增量裁决。

### 2.2 族级增量裁决（ΔIC_OOS ≥ +0.005 且残差符号与预注册方向一致；基准=7 信号等权，OOS IC -0.0134）

| 族 | 成员（dIC_OOS / 残差 IC / 方向一致） | 裁决 |
|---|---|---|
| ① 行业残差化动量/反转 | IND_REL_RET_60 +0.0271 / -0.0201 ✓；IND_REL_RET_120 +0.0091 / -0.0108 ✓ | **PASS**（两成员均过） |
| ② 流动性冲击/萎缩/量价背离 | LIQ_SHOCK_20 +0.0054 / -0.0019 ✓；PV_DIV_20 +0.0160 / -0.0268 ✓；VOLUME_SHRINK_5_20 +0.0001（未达门槛） | **PASS**（两成员过；VOLUME_SHRINK_5_20 记特征级负结果） |
| ③ 涨跌停事件条件 | LIMIT_UP_CNT_5 +0.0010；LIMIT_DOWN_STREAK +0.0000；LIMIT_BREAK_5 +0.0025——无成员达 +0.005，且预注册的延续方向（LIMIT_UP_CNT_5=+1）被证伪（实测 h=10 IC -0.0151，呈反转） | **负结果**（预注册规则：特征保留计算与采样、promotion_allowed=False） |
| ④ 横截面拥挤度 | CROWD_TURNOVER_60 +0.0202 / -0.0158 ✓；CROWD_AMOUNT_60 +0.0293 / -0.0210 ✓；MARGIN_CROWD_60 +0.0395 / -0.0287 ✓ | **PASS**（三成员均过） |

### 2.3 相关性与二次精简（0.9 阈值）

- **LIMIT_STREAK 条件弃用触发**：修复后 |ρ(LIMIT_UP_EVENT, LIMIT_STREAK)| = **0.980** ≥ 0.9 → 按预注册弃用（LIMIT_UP_EVENT 为保留原子）。
- **二次精简触发 2 对**：
  - LIQ_SHOCK_20 ~ VOLUME_RATIO 0.988（VOLUME_RATIO 已弃用，无行动）；LIQ_SHOCK_20 ~ **VOLUME_IMPACT 0.968**（活特征）→ 裁决弃 LIQ_SHOCK_20（其族门槛边际 +0.0054，VOLUME_IMPACT 为既有更稳健代表）。
  - **CROWD_TURNOVER_60 ~ CROWD_AMOUNT_60 0.971** → 裁决弃 CROWD_TURNOVER_60（CROWD_AMOUNT_60 ΔIC 与 ICIR 更强，为族④代表）。
  - LIMIT_DOWN_STREAK ~ LIMIT_DOWN_EVENT 0.985：两成员同属负结果族③（已 promotion_allowed=False），保留计算、不再另行精简。
- 新特征与其余既有特征无其他 ≥0.9 对。

### 2.4 裁决后的有效词表

- 词表 73 名（含 9 个 P9 前 deprecated → **12 个 deprecated**：+LIMIT_STREAK、+LIQ_SHOCK_20、+CROWD_TURNOVER_60）。
- 可采样 61 名；可晋级（promotion_allowed）= 61 − 3（族③负结果成员）− 0（其余全部允许）= **58**。
- 族③的 LIMIT_DOWN_EVENT/LIMIT_UP_EVENT/LIMIT_UP_CNT_20/LIMIT_BREAK（P9 前既有事件特征）不在负结果降级范围（契约裁决对象是族③新成员）；它们的覆盖修复属于 F1 修复红利，其中 LIMIT_UP_CNT_20/LIMIT_BREAK 首次成为真正可用的活特征。

## 3. 裁决执行（预注册后果的实现）

- `LIMIT_STREAK`、`LIQ_SHOCK_20`、`CROWD_TURNOVER_60` → DEPRECATION_REASONS（退出采样；token/解析不变）。
- `LIMIT_UP_CNT_5`、`LIMIT_DOWN_STREAK`、`LIMIT_BREAK_5` → feature_metadata `promotion_allowed=False`（保留采样、禁止晋级）。
- 版本：GRAMMAR 4→5（采样空间变化）、FEATURE_REGISTRY 4→5（裁决记录）；其余不变。
- 提交：见本文末"裁决后提交"哈希。

## 4. 运行统计与偏差

- 运行一次成功，无中途失败；无失败产物混入。
- 控制台 10 条 numpy RuntimeWarning 来自 test_diagnostics 的常数截面测试设计（与本审计无关，t2 时已存在）。
- 本日志只记实际发生之事；预注册内容见 docs/p9_factor_family_contract.md（APPROVED）。

## 5. 裁决后提交

- 分支：codex/p9-factor-families
- 哈希：（裁决后提交的 hash 以 git log 为准——见 t5 任务 output）
