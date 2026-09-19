# P38 数据资格解除测量日志（evidence-only）

本日志只记录实际发生了什么；计划见
`docs/p38_data_qualification_unblock_draft.md`（APPROVED，用户 2026-09-19
「全部授权」；Rev 1 修正 688033 勘察陈述于该步写入之前）。

## 1 软件交付

| 项 | 值 |
|---|---|
| 提交链 | 契约 `4ab696e`（+ CONTRACTS.md + p19 指针修订）→ RED `6f56636`（N1/N3 失败在能力缺口，N2/N4 守卫绿；证据 `D:\minequant\reports\p38_red_evidence.txt`）→ 实现 `1019362` |
| 实现 | `ashare_model/data_loader.py` `load_data` 的 missing_limit 判定一处：必需格网 = bar-present（open 有限且 >0）∩ 缺失定义对齐 exact 矩阵（非有限或 ≤0）；逐日交易语义零变化（N4 逐字 pin） |
| 验证 | 聚焦 p38 **4P**；相邻 53P（data_loader/capital_flow/p34-baseline）+ 96P（p37/wp3/backtest/evaluation）+ 63P（price_semantics/golden_parity/execution）；compileall/分片 112/112/diff 干净；全量 **2010 passed / 0 failed / 6 skipped / 3 xfailed**（1045.92s；6 skip 逐项= 1 锁审计基线 + 5 requires-CUDA，torch 2.11.0+cpu 环境条件，非本包新增） |

## 2 数据修复（先备份→审计→执行→复核；脚本 `D:\minequant\reports\p38_data_repair.py`）

备份：`data/backup_20260919_p38/`（主库 906M + v2 7.6G 全量 copy，写前完成）。

| 步 | 执行记录 | 复核 |
|---|---|---|
| R1 001914.SZ | 000043.SZ 前身限额 1,370 会话重映射（authority 拉取 + ts_code 重写，TushareStore `…T080812Z_f274bb42`，L0+ledger） | missing 1370→**0**，价格带违例 **0** |
| R2 000022.SZ | 001872.SZ 换码后继 20181219/20 两日重映射（`…T080812Z_07d47eed`） | missing 2→**0**，带违例 **0** |
| R3 688033.SH | 按码补拉 1,739 行（`…T080812Z_484356bf`，幂等 ON CONFLICT）——源首日行缺失（按码/按日均无，Rev 1 据实修正）；按 Rev 1 规则推导补行 20190722（up=100000/down=0.01=其自身 0723–25 与 24 家同伴首日同编码；pre_close=20.37=自身 bar 发行价，链路与其次日行 pre_close=40.08=首日收盘吻合；`…T081036Z_5058f1f5`，params 明示 derived） | missing 1→**0**，带违例 **0** |
| R4 申万主库 | AkShareClient 权威 append-only：31 码 × 9 会话（20260908..20260918）= 279 行；`sync_sw_index_source` 刷新 v2 | 逐码 min(MAX)=**20260918** ≥ 20260916；v2 max=**20260918** |

## 3 发布与解除验证

- 复合发布：`publish_composite_snapshot` → composite `d36024cd1b19…`（publication 6；primary `5a6fc892…`、v2 `b1237612…`）。
- 解除验证：随后 p37 六格面板正式运行（§见 p37 日志重跑节）装载后
  `loader_frame_status = clean`、`data_completeness = complete`——
  三个阻断全部解除，§5.1 门首次在真实数据上通过。

## 4 偏差与备注

- 修复脚本首跑在只读连接与 store RW 句柄生命周期上失败一次（**任何写入
  之前**，无部分状态）；调整为审计只读段先行后重跑成功。
- `funnel_totals_1x` 聚合键表（r01 既有常量）未含 p37 新增的首建仓三键——
  行级字段完整，聚合缺三键属报告便利缺口，非数据问题；本包未顺手改
  （单一权威常量属 r01_baseline，另立小包处理更妥）。
- 688033 推导补行若被裁决不可接受：回滚 = 从备份恢复 v2 后该股回到
  降级态并阻断（停止条件预设，未触发）。
