# P33 B04 装载与缓存性能测量日志（evidence-only）

契约：`docs/p33_b04_load_cache_performance_draft.md`（rev 1，APPROVED，用户 2026-09-16
直接指令「完成B04」）。本文只记录实际发生什么；计划在契约，二者不混写（AGENTS §11.1）。

## 1 运行环境与提交链

| 项 | 值 |
|---|---|
| 分支 / base | `codex/b04-load-cache-performance`，base = `codex/b03-resumable-execution-budget` @ `64b0887`（未 push/未合并） |
| 提交 | 契约 `589fceb`（rev 1 `a8788a6`）→ RED `dcf9d27` → 实现 `95ecfa6` → 基准脚本 `ffe3361` |
| 环境 | Windows（本机 16 GB 主机），Python 3.13.12，torch 2.11.0+cpu，numpy 2.5.2，psutil 7.2.2，DuckDB（仓库锁定版本） |
| 真实库身份 | 确认复合 id `b369c49b20f18a0adc4b5f57b4050a1bd735fdd788de4d9076f13be9e86874cf`（基准期间 `confirm_input_snapshot` 通过，发布代际无漂移） |
| 真实配置 | `configs/r1_medium_a.yaml`，sha256 `3DC7F6CA…924D4D` |
| 本包未运行 | Web 侧无改动故未跑 npm；未做数据库内容全量校验（A02 confirm 即本包身份权威）；未做串行全量对照（沿用既往里程碑的 parallel-vs-serial 对账基线） |

## 2 RED 证据（commit `dcf9d27`）

* `python -m pytest -q tests/test_b04_load_cache_performance.py -x` → 收集失败：
  `ImportError: cannot import name 'loader_factor_cache_identity_with_reason'`——契约
  API 缺位。
* 行为级探针（临时测试文件，取证后删除、未入仓；同 `populated_db` fixture + 身份源
  种子）：冷载后 `{pit:1, capital:1, factor:1}`，同身份热载后 `{pit:2, capital:2,
  factor:1}`——**热载重建 PIT/capital（现状实锤）；factor 本就在 miss 分支内**。
  该事实触发了契约 rev 1 勘误（§1/§3.1 的基线表述更正，先于实现提交 `a8788a6`）。

## 3 GREEN 与回归

| 命令 | 结果 |
|---|---|
| `python -m pytest -q tests/test_b04_load_cache_performance.py` | 10 passed（33.97s；R1–R7 + 兼容投影 + loader 级 disabled reason） |
| `python -m pytest -q tests/test_b04_load_cache_performance.py tests/test_load_cache.py tests/test_data_loader.py` | 首轮 41 passed + 1 failed（R7 缺 `import pandas`，RED 阶段笔误，随实现提交修复，断言未弱化）；修复后全绿 |
| `python -m pytest -q tests/test_capital_flow.py tests/test_semantic_cache.py tests/test_runspec.py tests/test_price_semantics.py tests/test_manifest.py` | 127 passed（93.50s） |
| `python -m pytest -q tests/test_p31_b02_entry_canary.py tests/test_p27_a03_lifecycle_runcontext.py tests/test_p29_b01_data_quality.py` | 66 passed（46.49s） |
| `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` | exit 0 |
| `git diff --check` | 干净 |
| `python scripts/check_test_shards.py --check` | 106 files across 4 shards cover all 106 test files（新测试同 commit 注册 shard-data） |

全量验证见 §6。

## 4 基准测量（`scripts/bench_load_stages.py` @ `ffe3361`）

方法：脚本进程内包装 loader 的阶段函数（生产代码零打点）；`torch.tensor` 归入
`copy` 阶段；未归入残差记 `other`；psutil 50ms 采样峰值 RSS；等价断言为
`np.array_equal(..., equal_nan=True)` 逐键逐值相等（NaN 感知；契约 §4 操作性定义）。原始 JSON 与 sha256：

* `D:\minequant\reports\AlphaGPT-b04-benchmark\b04_bench_synthetic.json`
  （`2B64613068A85B819D1EAADEB3D3AE52663F66C6304BEA80332A33FF2AB0C0FC`）
* `D:\minequant\reports\AlphaGPT-b04-benchmark\b04_bench_real.json`
  （`25193109AA46E433934AF959E426F25A943CDC93A17E9F9E82B6FCFBBA1B188A`）
* `D:\minequant\reports\AlphaGPT-b04-benchmark\b04_bench_real_window.json`
  （`85FA192BAA8954B2924CDFEB556427E614E4710A520D855C3CB047A78E017A9E`）

### 4.1 合成 fixture（60 码 × 240 会话，真实发布路径，fixture 构建 49.5s）

| 相 | 结果 | 墙钟 | 峰值 RSS | pit | capital | factor |
|---|---|---|---|---|---|---|
| cold | miss | 14.29s | 348MB | 6.72s | 0.54s | 5.29s |
| hot | hit | 1.73s | 356MB | **0** | **0** | **0** |
| invalidate | miss | 13.84s | 368MB | 6.41s | 0.64s | 5.12s |
| hot2 | hit | 1.72s | 364MB | 0 | 0 | 0 |

六项检查全 TRUE（cold_is_miss / hot_is_hit / hot_tensor_equals_cold /
invalidate_is_miss / hot2_is_hit / hot2_tensor_equals_cold）。
**热/冷墙钟比 12.1%**（≤30% 建议目标达成）。

### 4.2 真实库·全历史（额外行；脚本窗口 bug 披露，见 §5）

第一次 `--real` 运行的窗口字段按预注册写了 300×260，但脚本未把窗口传给
`load_data`——实际装载了**全部已解析 universe × 全部 3084 个会话**
（缓存键 `window=20140102..20260907:3084` 为证；脚本随即修复并按预注册窗口重跑，
本行数字保留为全历史测量）。dataset = medium 域配置全历史：

| 相 | 结果 | 墙钟 | 峰值 RSS | confirm | read | tensor_maps | pit | capital | factor |
|---|---|---|---|---|---|---|---|---|---|
| cold | miss | 2892.47s | 7959MB | 138.07s | 28.32s | 67.16s | 2329.31s | 37.69s | 270.17s |
| hot | hit | 331.96s | 7860MB | 216.57s | 31.47s | 64.95s | **0** | **0** | **0** |

三检查全 TRUE（NaN 感知逐值相等，冷热一致）。**热/冷墙钟比 11.5%**。
PIT as-of 占冷载 80.5%——方案 B04「优先消除重复工作」的靶心得到真实库证实。

### 4.3 真实库·预注册窗口（300 码 × 260 会话，20250813..20260907，cutoff 20260907）

| 相 | 结果 | 墙钟 | 峰值 RSS | confirm | pit | capital | factor | read |
|---|---|---|---|---|---|---|---|---|
| cold | miss | 516.72s | 4404MB | 130.09s | 366.91s | 2.33s | 8.66s | 0.72s |
| hot | hit | 200.73s | 4765MB | 192.66s | **0** | **0** | **0** | 0.66s |

三检查全 TRUE（NaN 感知逐值相等，冷热一致）。python exit 0。

**30% 建议目标的裁决（契约 §3.4 预注册的如实披露义务）**：

* 端到端热/冷墙钟比 = **38.8%**，未达 30%；
* 分解：热载墙钟的 **96% 是 A02 内容确认税**（192.66s / 200.73s，两库全量未缓存
  重哈希，契约 §3.1 预注册为热路径不豁免）；剔除确认税后——
  热去确认税 8.07s / 冷去确认税 386.63s（=(516.72−130.09)）= **2.1%**（合规审
  F-2 勘误：初版印刷分式分子分母颠倒，结果值本身正确）；
* 本包靶定的重复工作（pit 366.91 + capital 2.33 + factor 8.66 = 冷载去确认税的
  97.7%）在 hit 路径**全部归零**——「重复工作消除」目标达成；端到端小窗口比值
  由 A02 身份确认下界决定。若未来需要压缩端到端热载耗时，必须经 A02 契约层的
  确认成本变更（如已发布代际的增量确认），属独立预注册工作包，本包不做。

### 4.4 基准期间产生的本机产物

* `data/r1/medium_a/factor_tensor_cache/`：
  `a68b08a4….npz`（1,680,593,190 字节，全历史身份的合法内容寻址条目，对未来同身份
  装载即热条目）与 `532f3313….npz`（37,551,746 字节，300×260 窗口条目）。二者为
  A01 语义下的允许输出（data_dir 派生缓存），未入仓、未触任何数据库写入。
* 真实库装载记录了既有 688xxx（科创板）`daily_price_limit` 缺记录告警——B01 已
  登记的存量数据缺口经 p19 §2.4 fail-closed 路径如实呈现，与本包无关。

## 5 偏差与调试记录（契约 §8 义务）

1. **契约 rev 1**（实现前）：基线事实勘误——factor 计算本就在 miss 分支内，命中路径
   重复工作为 PIT/capital 帧（行为级探针实测），§1/§3.1 更正。
2. **R7 golden parity 抓到两个真问题**：(a) sw-index 查询下推初版缺 `WHERE`
   关键字（语法错误，被既有降级 except 吞成「industry closes unavailable」告警）；
   (b) 测试夹具自身不对称（full 夹具窗口 circ_mv 因前置窗口外日期错位），修正夹具后
   parity 全绿。**技术审 F1 勘误（rev 2，2026-09-16）**：初版 R7 的 6 会话窗口下
   INDUSTRY_MOMENTUM 恒为全 NaN——上述 (a) 缺 WHERE bug 的暴露机制是开发期观察
   告警文本，**不是已提交态 R7 断言**（技术审降级探针实证：sw 源不可用时五帧逐位
   相同、旧 R7 全绿）。R7 已按契约 rev 2 重做：24 会话窗口 + momentum/CHG 非 NaN
   手算单元格；降级探针复跑：健康 col20 = 0.04，sw 源降级 col20 = NaN——牙齿实证
   生效。
3. **基准脚本两处运行期修正**（先于测量证据采集）：`AshareDataLoader` 导入缺失、
   `_loader_args` 解包；以及 §4.2 披露的窗口传递 bug（修复后按预注册窗口重跑）。
4. **confirm 耗时波动**（130.09s ↔ 192.66s ↔ 216.57s）：同一对库、同一实现；归因于
   OS 文件缓存与页缓存热度（全历史冷载先行预热），未单独控制变量；三值如实并列。
5. 退出码：pwsh 管道把 stderr 告警升级为 ErrorRecord 导致任务级 `exit 1` 噪声；
   python 退出码以 `python_exit=$LASTEXITCODE` 显式核实（窗口运行 = 0）。

## 6 全量验证

| 命令 | 结果 |
|---|---|
| `python -m pytest -q tests -n auto`（PATH 首位 = miniconda3） | 1946 passed, 6 skipped, **2 failed**——`test_lock_files.py` 两例：`PackageNotFoundError: tushare`。与 p31 测量日志记录的同一环境事实（miniconda3 缺 tushare 发行元数据；正确运行时为 `.venv`）；本包 diff 不触依赖/打包，失败与改动无关 |
| `D:\minequant\.venv\Scripts\python.exe -m pytest -q tests -n auto`（onboarding §9.2 声明运行时） | **1953 passed, 1 skipped, 0 failed**，879.93s |

## 7 裁决

* 工程（AGENTS §4.1）：F07 关闭——命中判定先于 PIT/capital 帧构建；缓存四态
  （disabled/miss/hit/invalidated）与原因可观测；capital 帧日期下推行为不变。
* 等价性（契约 §4）：同一 fixture 与固定真实窗口，冷/热输出按契约操作性定义一致
  （`np.array_equal(..., equal_nan=True)` 的 NaN 感知逐值相等，rev 2 措辞澄清）；
  依赖变化（新增 ingest + 重发布）必然 miss，旧条目保留。
* 性能（方案 B04 建议目标，待标定口径）：合成 12.1%、真实全历史 11.5%、真实
  预注册窗口 38.8%（去确认税 2.1%）——重复工作消除达成，端到端小窗口下界为
  A02 确认税；不以裁剪窗口/跳过门禁冒充加速（装载路径的读取、资格与确认步骤
  全部保留）。
* 研究声明边界：本日志不含任何 alpha/收益/晋级结论；R01 起的研究裁决另依预注册。

## 8 双审处置（2026-09-16）

两位审查子agent对 `64b0887..2891d42` 独立审查（全程只读，仓库零修改）：

**合规侧：PASS**，无 blocker/high/medium；4 条 low 全部处置：
* F-1（fixture 构建 49.2s → 49.5s）：本日志 §4.1 已更正（原值来自首轮 stdout 摘要，
  JSON 实测 49.5）。
* F-2（§4.3 去确认税分式分子分母颠倒印刷）：已更正（§4.3，结果值本身正确）。
* F-3（实现 commit message 对测试 helper 改动披露不完整）：合规审逐行核实断言
  零弱化、R7 三条下推探针仍仅存在于 full 库、且测量日志 §5.2 已如实披露——记录
  勘误不改写历史；后续 commit message 完整列举测试文件改动。
* F-4（测量日志 commit 信息「six/five-three protocol checks TRUE」计数含混）：
  实际为合成 6 项、真实全历史 3 项、真实窗口 3 项检查全 TRUE（日志本身精确）——
  记录勘误不改写历史。

**技术侧：NEEDS_REVISION（轻度）**，1 medium + 2 low 修复，3 条 info 采纳：
* F1（medium）：R7 对 sw 下推腿无回归牙齿（6 会话 < `_INDUSTRY_WINDOW`=20，
  INDUSTRY_MOMENTUM 恒全 NaN；降级探针实证旧 R7 在 sw 源不可用时仍全绿）→ 契约
  rev 2 + R7 重做（24 会话 + momentum/CHG 非 NaN 手算单元格），降级探针复验牙齿
  生效（健康 col20 = 0.04，降级 = NaN）；§5.2(a) 暴露机制表述已更正。无生产代码改动。
* F2（low）：loader 级 `factor_cache_reason` 断言从 1/5 扩到 4/5（新增 R6d：
  ingest_source_empty / ingest_source_missing / marker_tables_missing 经 query 模式 +
  表删除构造）；identity_derivation_failed 保留函数级并经契约 rev 2 记录理由。
* F3（low）：bench StageClock `__enter__` 部分失败自解绕（先于传播回滚已安装包装）。
* F4（low）：等价性措辞按契约 rev 2 澄清为 NaN 感知逐值相等（本文 §4/§7 同步）。
* F5/F6/F7（info）：RSS 为 50ms 采样下界（已披露）；`aggfunc="last"` 唯一性假设
  已加注释记录（capital_flow，主键实证零重复组）；confirm 失败次序探针实证无产物
  差异——均无需整改。

**修复后回归**：`tests/test_b04_load_cache_performance.py` **11 passed**（新增 R6d，
R7 重做）；`tests/test_load_cache.py`、`tests/test_capital_flow.py` 随行复跑全绿；
`git diff --check` 干净。修复提交不含任何生产语义变化（测试/脚本/文档 + capital_flow
注释一行）。
