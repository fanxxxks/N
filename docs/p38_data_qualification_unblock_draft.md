# P38 数据资格解除：p19 §2.4 限额必需格网收窄 + 三项限额修复 + 申万源追新 预注册契约

- 状态：**APPROVED**，用户 2026-09-19 会话直接指令「全部授权」（对 2026-09-19
  工作包 4 交付报告所列后续项的整体授权：三个数据阻断的契约治理决策与实施、
  六格面板重跑、第二轮周期比较。本契约承载其中的数据解除部分；面板重跑与
  第二轮按 p37 契约与后续指令执行）。
- 起草：执行模型（GLM Harness 会话），分支 `codex/b04-load-cache-performance`。
- 勘察基线：`codex/b04-load-cache-performance @ 1d1ef9f`（p37 全链已交付）。
  `git status` 干净（仅 `.mimosa/` 未跟踪）。
- 上位依据：p19 市场价格语义契约 §2.4；p17 §12（申万源新鲜度）；p34 测量日志
  §5.4/§7.3（三个阻断的类型化清单）；AGENTS §3.2/§5.6（数据变更有状态操作：
  先只读审计、再执行、再复核）；p37 §9（解除条件）。

## 1 问题陈述

p37/R01 正式运行被 §5.1 loader 帧降级 fail-closed 拒绝，阻断集合（2026-09-19
只读复核，与 WP0/WP3 逐项一致）：

1. **1,630 码 price_semantics_degraded**：p19 §2.4 预注册了「当日无涨跌停价
   记录 → 该股当日 fail-closed 不可交易 + degraded 记录」的**逐日**语义，
   但未定义限额记录在哪些 (股, 会话) 上是**必需**的。现行实现对全
   (码 × 加载会话) 格网要求，把无 bar 的会话（未上市前、退市后、停牌无行）
   也计入缺失 → post-2014 上市的 1,630 码被虚假披露。WP0 已认定「收窄属
   p19 契约修订决策」，即本契约。
2. **001914.SZ 前身时段 1,370 会话缺限额**：其 2014 起 bar 已被 D3 回填映射
   到现码，但限额未映射；前身 000043.SZ 的限额在本地库与 tushare 源均完整
   （1,452 行，止于 20191213，恰与 001914 自身限额起点 20191216 相接）。
3. **000022.SZ 两会话（20181219/20）与 688033.SH 一会话（20190722，其上市
   首日）缺限额**：勘察证实均为真实成交会话（vol>0），非停牌；001872.SZ
   （000022 的换码后继，同一公司持续身份）名下含这两日限额且价格带吻合；
   688033 的首日行 tushare 源有（其余科创板首日行存在，无限额编码为
   up=100000/down=0.01，个股缺口非规则性缺位）。
4. **申万指数收盘源滞后 7 会话**：主库 sw_industry_index 止于 20260907。
   2026-09-19 只读探测：AkShare 源已恢复（index_hist_sw 返回至 20260918，
   领先 bar 截止 20260916）；当前环境 akshare 1.18.91 下 WP0 记录的本地解析
   错误不复现。

**目标**：在不改变任何逐日交易语义的前提下恢复正式数据资格（loader 帧
clean），使 p37 六格面板可执行。

**非目标**：不改价格/标签/成交语义（p19 §2.1–2.3 零变化）；不改 exact
可交易矩阵；不改 G1–G8/D7 阈值；不删任何既有降级披露路径（未来真实缺口
仍 fail-closed）；不做全量数据重采；不改生产默认配置。

## 2 修订语义（p19 §2.4 增补，冻结）

**限额必需格网**：`(股, 会话)` 需要限额记录 ⟺ 该股在加载窗内该会话**有
bar 行**（open 有限且 > 0——与 exact 矩阵的 dead-bar 约定同一判据）。
无 bar 会话（未上市前、退市后、停牌无行）结构性地不存在「当日涨跌停价」，
不计入必需格网。**缺失定义与 exact 矩阵一致**：up/down 非有限或 ≤0 视为
缺失。必需格网上缺失 → 该股 price_degraded（不变，fail-closed）。

**逐日语义零变化**（收窄的合法性依据）：缺限额会话已由
`_exact_blocked_parts` 双向 blocked（`up≤0`/NaN → missing → blocked）；
无 bar 会话已由 dead-bar 双向 blocked。披露格网收窄不触碰任何 mask/价格/
标签路径——`tradability_masks`、引擎、构造器零改动。

**版本影响**：无版本 bump。无 artifact schema 消费该格网定义；价格/标签/
成交/晋级语义零变化；资格结果按运行记录于各 run 的数据报告。p19 契约文件
由本契约以指针方式修订（§7），p19 自身版本承载不变。

## 3 数据修复（同包授权；先只读审计 → 执行 → 复核，全部留档）

1. **001914.SZ**：将本地 daily_price_limit 中 000043.SZ 的行（权威源已
   ingested）按 (trade_date, up_limit, down_limit, pre_close…) 原值重映射为
   001914.SZ，仅插入 001914 无行的会话（append-only，不覆盖既有行）。复核
   判据：001914 全部 bar 会话缺限额数 = 0；重映射行与其 bar 的价格带抽查
   一致（open ∈ [down_limit − 0.01, up_limit + 0.01]）。
2. **000022.SZ**：同法从 001872.SZ 重映射 20181219/20181220 两行。复核：
   缺口 = 0 + 价格带一致（勘察已证：000022 20181219 open 17.29 ∈
   [15.66, 19.14]；20181220 open 16.71 ∈ [15.05, 18.39]）。
3. **688033.SH**：经 TushareStore 权威按码补拉 stk_limit（源有 1,737 行，
   本地缺 20190722 一行；WP0 回填脚本同型：L0 快照 → 原子 L1 → ledger 行）。
   复核：缺口 = 0。
4. **申万主库追新**：经 AkShareClient.get_sw_index_history（唯一 AkShare
   权威）对主库 sw_industry_index 的全部既有 index_code 追加
   trade_date > 现最大值的行（append-only，禁止改写既有历史）；随后运行
   `sync_sw_index_source` 刷新 v2 副本（既有权威函数，历史复制 + 新鲜覆盖）。
   复核：各 index_code MAX(trade_date) ≥ 20260916；行数增量留档。
5. 修复后重发布主库与 v2 manifest 并重新实时确认复合身份（A02 发布边界；
   dataset id 必然变化——新 id 入 p37 重跑的 provenance）。

以上修复脚本置于 `D:\minequant\reports\`（WP0 先例），测量日志记录命令、
前后行数、复核查询与结果。任何一步复核不过 → 停止并如实报告，不部分放行。

## 4 预期 RED 测试（实现前失败）

`tests/test_p38_limit_requirement_grid.py`（同 commit 注册分片）：

- N1 **pre-listing 收窄**：迷你双库中构造一只 bar 自窗中起步的股票（其 bar
  会话限额齐全）→ `loader_frame_status` **clean**（实现前：degraded）。
- N2 **必需格网内缺失仍降级**：删除该股一个 **bar 会话**的限额行 → 该股
  price_degraded（不变量：收窄不放松真实缺口披露）。
- N3 **缺失定义对齐**：限额行 up_limit=0（非有限→≤0）在 bar 会话上 →
  降级（与 exact 矩阵同判据）。
- N4 **逐日语义 pin**：缺限额会话双向 blocked、无 bar 会话双向 blocked
  （exact 矩阵行为逐字不变——防止收窄误伤）。

## 5 实施落点

`ashare_model/data_loader.py` `load_data` 的 missing_limit 判定（一处）：
以 bar-presence 掩码（open 有限且 >0）与对齐后的缺失定义（非有限或 ≤0）
相交替换现行全格网判定。无其他代码改动。

## 6 验证矩阵

1. `python -m pytest -q tests/test_p38_limit_requirement_grid.py`
2. 相邻：`tests/test_data_loader.py tests/test_p34_r01_baseline.py
   tests/test_p19_price_semantics*.py`（若存在）`tests/test_capital_flow.py`
3. `python -m compileall -j 0 -q …`；`python scripts/check_test_shards.py --check`；
   `git diff --check`
4. 全量 `python -m pytest -q tests -n auto`
5. 数据操作：审计/执行/复核记录进测量日志（§3）。
6. 解除验证：`loader_frame_status` 在真实库上 clean（只读探针）；随后 p37
   六格面板重跑（另按 p37 契约）。

## 7 对 p19 的指针式修订

p19 §2.4 增补一句（由本契约冻结）：「限额记录必需于 (股, 会话) 有 bar 行
（open 有限且>0）之处；无 bar 会话不属必需格网；缺失=非有限或≤0，与
exact 可交易矩阵同判据。」p19 文件头部加修订指针指向本文件；p19 其余
条款零变化。

## 8 停止条件

- 任何复核判据不过 → 停止，保留审计记录，不部分放行。
- 修复后 §5.1 仍因**其他**未披露降级拒绝 → 如实报告新阻断集合，不扩权
  处置。
- 发现修复引入价格带矛盾或行数异常 → 回滚该步（备份先行：主库与 v2
  修复前各留一份 copy，WP0 §5.1 先例）。
