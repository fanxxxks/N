# experiments/ — 实验留档

每次训练 / 回测 / 模拟盘跑完后，用 `scripts/archive_run.py` 把结果快照到
`experiments/<YYYYMMDD>_<公式>/` 并提交，让"哪个公式 + 哪份配置 + 哪份代码 + 什么结果"
永久可追溯。

```bash
python scripts/archive_run.py --mode backtest --commit
python scripts/archive_run.py --mode train --commit
python scripts/archive_run.py --mode sim --commit
python scripts/archive_run.py --mode protocol --commit
```

## 目录内容

- `manifest.json`：运行模式、创建时间、代码 commit SHA、工作区是否干净、
  数据末端日期（DuckDB `MAX(trade_date)`，尽力获取）、各产物 SHA-256 与存储决策；
  protocol 模式额外带 `protocol` 块（`version` / `frequency` / `horizon` / `tier` /
  `steps` / `batch_size` / `n_folds` / `n_seeds` / `n_candidates` / `dsr` / `max_t` /
  `top_candidate`，T4-01 起另含 `n_stitched_trials` / `ledger` / `data_regime`）。
- `formula.json`：公式 token 序列 + `formula_text` + `best_reward`。
- `config.yaml`：本次运行使用的配置快照。
- `metrics_summary.json`：关键指标摘要（顶层标量与 `metrics` 子字典中的标量），始终写入；
  protocol 模式改为 `summarize_protocol` 摘要（来源标量 + 候选聚合 + top trial +
  stitched/ledger/data_regime 信息），逐折逐种子的原始行始终保留在 `metrics.json` 里。
- `metrics.json`：完整指标文件；超过 `--max-metrics-size-mb`（默认 5）时只保留摘要。
- `model.*`：模型权重；超过 `--max-model-size-mb`（默认 2）时**不复制**，
  只在 manifest 里记录 SHA-256，权重留在本地 `data/`。

## 约定

- 实验目录只增不改；同一日期同一公式重复归档时自动加 `_2`、`_3` 后缀。
- `--commit` 只提交本次实验目录，不触碰其他未提交改动，也不会 push。
- 大文件（DuckDB / parquet / logs / node_modules）永远不入库，见根目录 `.gitignore`。
