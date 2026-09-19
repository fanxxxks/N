# P39 R02 第二轮（周期比较）测量日志（evidence-only）

本日志只记录实际发生了什么；计划见 `docs/p39_r02_periods_draft.md`
（APPROVED，用户 2026-09-19「全部授权」）。

## 1 软件交付

| 项 | 值 |
|---|---|
| 提交链 | 契约 `de5724e`（+ CONTRACTS.md）→ RED `ed3cb9f`（8 失败全在能力缺口；证据 `D:\minequant\reports\p39_red_evidence.txt`）→ 实现 `013a11a` → 修复 `3a9281a`（见 §3 偏差） |
| 实现 | `R02_PERIODS`（域合法集恰两点，P1 对 `ashare_domain` 权威断言）；`period_backtest_variant`（恰两字段）；`run_r02_panel` 增 `cells` 参数（默认全格，既有调用零变化）；`_arm_verdict`/`_has_gross_info` 共享辅助（round-1 裁决重构后行为同一，p37 17P 回归绿）；`decide_r02_periods_adjudication`；`run_r02_periods` 驱动 + `scripts/r02_periods.py` CLI |
| 验证 | 聚焦 p39 **8P**；回归 p37+p38+p34 **36P**；compileall/分片 113/113/diff 干净；全量见 §4 |

## 2 正式运行

### 2.1 首次运行（**超范围，已披露偏差**）

时间 17:4x–18:0x，`@013a11a`：run 落 `period_panel_no_reliable_info` →
RETIRED，**620 行**。事后核对发现 `cells` 参数只加了签名、循环体仍迭代
全格表——实际执行了六格 × 两周期（超出预注册的参考格范围）。处置：

- **裁决范围未受污染**：`period_summaries` 只消费参考格 `n20_m5000` 的臂
  （契约 §5 冻结口径），verdict 在预注册范围内计算；
- 超范围行（n10/n20 其余五格 × 两周期）**不作为证据使用**，仅随 run store
  原样保留；
- 修复 commit `3a9281a`（循环接线 + P4 cell 集合断言加强——原断言经第一
  个 `::` 分割后跨格塌缩，属覆盖缺口，加强非弱化）先于重跑提交；
- 契约 §3 未修订——重跑即按其原义（仅参考格）执行。

### 2.2 正式重跑（预注册范围）

命令（工作树干净 @ `3a9281a`，实时契约 sha
`9f82ad31af63138b470b63adbabaea55c27326cc355bdd5ddbf14a13f4271ce4`）：

```
python scripts/r02_periods.py --contract-sha256 9f82ad31… --out-root data/r1/r02_periods
```

| 项 | 值 |
|---|---|
| 身份 | dataset `d36024cd1b19…`（p38 修复后复合，实时重确认）；cutoff `20260916`；purpose `historical_research`；`execution_spec_version=4`；factor cache **hit**（4ab7587a…/d36024cd 窗口，第一轮六格运行已写入） |
| 范围 | **120 行**：cells={n20_m5000} ✓、periods={every_5_days, every_10_days} ✓（P4 同款断言在测试中钉住） |
| 数据资格 | G1–G8 8/8、loader frames **clean**、completeness **complete**（p38 解除后的稳态） |
| **verdict** | **`period_panel_no_reliable_info` → RETIRED**：参考格上两周期均无成立臂（5 臂全部 no_net_space；every_10_days 上 4/5 臂叠加 low_activity，投入 0.41–0.53）；**无任何 (周期,臂) 毛信息**（0x 拼接超额全负） |
| 墙钟 | ≈10.4 min（625.8s，含 confirm ~5 min + 双周期面板） |

**判读**：在冻结的参考格（n20_m5000，= r1_medium_a 原值）上，合法周期轴
**不构成瓶颈也不构成解药**——every_5_days 与 every_10_days 都无法让该
组合配置显示净空间，毛信号在该配置下也无信息。周期轴按冻结规则暂停扩展
（verdict 4 的预注册含义）。与第一轮结论并列（不互替）：第一轮的净空间
存在于 **n10 持仓 × 全部三档最低订单** 区域（combo:all3），即瓶颈是
持仓结构而非周期；n10 × 周期的交互**未预注册、未测量**（超范围运行的
相应行不作为证据）。

## 3 偏差登记

1. 首次运行的超范围面板（§2.1）——实现缺口 + 测试覆盖缺口，双重修复后
   重跑；超范围数据保留不使用。
2. 六格运行启动两次因树脏被拒（16:47 与 16:51，spec 构建前，无 run 产生、
   无数据改动）——与 p39 实现提交的时间窗竞争所致，流程教训：正式运行
   启动后至 spec 冻结前不动跟踪文件。

## 4 全量验证

修复 commit `3a9281a` 后 `python -m pytest -q tests -n auto`：
**2018 passed, 0 failed, 6 skipped, 3 xfailed**（1068.52s；6 skip 与
p37/p38 日志同源——1 锁审计基线 + 5 requires-CUDA 环境条件，无本包新增）。
