# Phase 0 测量日志（P0-01 … P0-06）

> 记录时间：2026-08-27（Asia/Shanghai）  
> 分支：`phase0/doctor-and-consistency`（合并前基线 `56f8e53`）  
> 规则：改动前先 commit 当前状态 → 分支开发 → 验证 → 合并回 main。

## 1. 提交前后不变量

| 不变量 | 改动前（main @ 74f833e） | 改动后（branch @ 56f8e53） | 验证方式 |
|---|---|---|---|
| 全量 Python 测试 | **无法收集**：`starlette.testclient` 缺 `httpx2`（`tests/test_webapi.py` 收集即失败） | **921 passed, 0 failed**（760 s） | `pytest -q tests`（`logs/pytest_phase0_final.log`） |
| Web 前端构建 | 通过（44.9 s，chunk 大小警告除外） | 通过（含 LEGACY 徽标改动） | `npm run build`（webui） |
| 依赖 pin 一致性 | `freeze_lock.py --check` 通过 | 通过（且新增 CPU/CUDA 一致性校验） | `freeze_lock.py --check` + `tests/test_lock_files.py` |
| 干净环境安装 | 未验证（基础 pin 含 `torch==2.11.0+cu128`，PyPI 无此 wheel，CI 安装必然失败） | 全新 venv 安装 `requirements.txt` + `requirements-optional.txt` 成功（CPU torch），锁定测试通过 | 见 §3 |
| 数据集 dataset_id | 无（`dataset_manifest` 表不存在 → `resolve_dataset_id` 返回 None） | `b927074a455a25c65698b61dbee9da48097d3121fc759585e73543c8d56d4318`（11,003,350 行 / 8 表） | `python -m ashare_data.manifest` |
| G1–G7 生产门禁 | 全部通过（只读核查） | 7/7 PASS（formal 模式） | doctor 报告 |
| 旧产物标记 | `best_ashare_strategy.json`（reward v10）、`protocol_result.json`（protocol v12 / reward v10）无 legacy 标记 | 两者均已盖章 `legacy: true` + 原因 + 时间戳（幂等） | `scripts/stamp_legacy_artifacts.py` |
| doctor 版本冲突 | 不适用（工具不存在） | **无冲突（healthy=true，exit 0）**，3 条 info 级 legacy 说明 | `python -m ashare_model.research_doctor` |

## 2. 提交清单（每 commit 一件事）

| Commit | 内容 |
|---|---|
| `74f833e`（main） | docs: 基线当前状态（PROJECT_ONBOARDING + 移除过期的 2026-08-23 评估报告） |
| `a2b4033` | **P0-06** deps: CPU 基础依赖与 CUDA 安装分离 |
| `32d749c` | **P0-01** searcher: 默认搜索器统一为 GP（Python/YAML/README；deap 进基础依赖） |
| `e4100da` | **P0-02** docs: 更新 reward v13 / MTPHead / screening 预算说明 |
| `1643395` | **P0-03** doctor: 只读 research doctor + manifest CLI |
| `e4aa1c8` | **P0-04** legacy: 旧 v10/v12 产物标记 + 消费端防护 |
| `caa9a94` | **P0-05** ci: React 构建 + 依赖检查 |
| `56f8e53` | fix(doctor): 人类可读输出改用 ASCII 破折号（Windows 控制台乱码） |

## 3. 关键测量

- 全量测试：**921 passed, 0 failed, 78 warnings（既有），760.08 s**（Python 3.13.12，venv `D:\minequant\.venv`，torch 2.11.0+cu128 本机）。
- 新增测试：`test_lock_files.py`（+CUDA 一致性契约）、`test_artifact_versions.py`（11）、`test_research_doctor.py`（13）、`test_manifest.py`（+CLI）。
- 干净环境安装（验收项）：全新 venv（Python 3.13）仅按 pin 文件安装，`torch==2.11.0+cpu` 从 PyTorch CPU index 解析成功；安装后锁定相关测试与 `freeze_lock --check` 通过（详见 `logs/` 下 phase0 安装日志）。
- doctor 运行（正式模式，只读）：`data/research_doctor.json` —— gates 7/7 PASS；依赖 12 项版本齐全；预计运行量：train_default ≈ 9.4 min、screening 单跑 ≈ 9.4 min、confirmation 单跑 ≈ 12.8 min、protocol 全量训练 ≈ 141 min（README 记载节奏的估算，非实测）。
- legacy 盖章：strategy（5 条原因：无 searcher / reward v10≠13 / 无 protocol_version / 无 model_version / 无 dataset_id）；protocol（5 条原因：protocol v12≠20 / reward v10≠13 / 无 dataset_id / 无 stitched / 无 ledger）。重复执行无改动（幂等）。

## 4. 迁移与拒绝策略（P0-04）

- 旧产物**不删除、不转换**：仅原地盖章 `legacy/legacy_reason/legacy_stamped_at`，保留可读性。
- 消费端防护：web API 对未盖章的旧产物按分类规则现场补 `legacy` 标记；`backtest` / `run_sim` 加载旧产物时输出 LEGACY 警告（仅警告，不阻断——旧公式回测属于合法存档用途）；晋级（promotion）与协议（evaluation）本就拒绝旧版本产物。
- 需要当前代 champion 时：按当前版本（reward v13 / protocol v20 / GP 默认）重新训练并回测，产物自然不带 legacy 标记。

## 5. 遗留说明

- `docs/PROJECT_ONBOARDING.md` 是 2026-08-27 的时点快照（分析基线 c5b801e），其中"DEAP 在 optional"等观测已被 P0-01/P0-06 修复；作为快照文档未回改。
- 未运行正式训练/协议/回测/模拟盘/同步（验收约束）；仅执行了 manifest 构建（数据写入，非训练）、legacy 盖章（产物元数据写入）与只读 doctor。
