# 四路公式搜索：本地交付与云端操作

本轮只准备代码、核验数据和运行最小功能验证。没有推送、没有登录目标云服务器、没有执行 1,000 / 10,000 / 50,000 次正式搜索。旧年份均为反复研究过的历史数据，后续测试区间是滚动研究测试段，不声称全新独立保留集。

## 1. 代码与整合

整合目录 `D:\minequant\AlphaGPT-cloud-integration`，分支 `codex/cloud-four-search`。原主目录基底 `dbaf7296a41ce9813ae736c0e4879d54d9488b0f`，原任务分支检查时为 `dc809e5`。先提交保全的主目录修改 `3c9cfb4`，再人工合并任务成果 `9a948c5`。最新完整交付提交用 `git log -1 --format=%H` 查看；最终检查记录见 `reports/cloud_20261008/`。

冲突按代码块结合：`backtest.py` 同时保留成交拒绝诊断、现金/整手/费用及非调仓日强制退出；`constructor.py` 保留实际最小订单检查、现金分配和退出规则；`run_sim.py` 使用连续目标股数，再按配置整手约束交易差额，避免重复取整。`execution_spec.py` 升到版本 6，因子定义版本保留 5；既有版本、数据身份和语义缓存键继续生效。GP 子代清除父代适应度后重新评分，AMPLITUDE 按股票/日期矩阵计算，训练候选池与历史摘要保持有界。

取消跟踪的是逐笔/每日大结果及明确列出的历史原始 metrics，磁盘文件仍保留。没有执行 reset、clean 或删除原目录文件。源码、配置、依赖、测试、必要契约/小型证据明确列入提交。`.gitignore` 不全局屏蔽 JSON、NPZ 或 reports；`config/.env.example` 保留无真实密钥的例子。交付清单与排除清单见报告，暂存审计扫描当前索引所有文件，报告仅含路径、不打印匹配内容，不等同于审计全部历史提交。

Git 的全局及仓库 HTTP/HTTPS 代理已按本轮要求统一 `http://127.0.0.1:7892`，端口监听及 `git ls-remote origin HEAD` 本地只读检查通过。后续由用户推送：

```powershell
Set-Location D:\minequant\AlphaGPT-cloud-integration
git push -u origin codex/cloud-four-search
```

本轮另外提供离线 bundle，可不依赖推送先上传部署。Bundle 含整合分支可达历史，索引审计不代表历史秘密审计；不将它发布为公开下载。

## 2. 上传数据

只上传以下两个完整搜索数据库，保留成对身份：

| 本地源文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `D:\minequant\AlphaGPT-task1\data\task2_reference_20261007_clean\ashare.duckdb` | 950546432 | `c13f91f526d6a79d46a90455a4b05c1acb088445ad057465705be7fc039c7529` |
| `D:\minequant\AlphaGPT\data\ashare_v2.duckdb` | 8135127040 | `f96e9813861312335a74996e4fa58dad875f9a60a9bd4a39388f94ecfb98dc10` |

部署前必须校验 `reports/cloud_20261008/data_upload_manifest.json`。组合 dataset ID 为 `d36024cd1b197a8edcf976b2d950b3506bb0fae022f121d6aef06f2a281cde55`，publication ID 为 7。两成员 manifest ID 及校验摘要也在清单中。本地完成完整内容身份复核，原始日线源截止 2026-09-16；配置历史研究最晚截止 2025-12-31。五个窗口 G1–G8 与 D7 均通过。每个实际任务按自身测试末日确定研究截止，再完整复核身份与门禁；不会关闭身份/历史校验。

不上传三固定因子回放、旧因子缓存、旧模型、原虚拟环境、WAL、逐笔结果、工具状态或密钥。总数据库约 9.09 GB 十进制。带宽 25 Mbps 的传输时间受网络影响，本文没有实测上传时间。上传期间不得写数据库；若源文件变化，重新复制并校验，不沿用这份身份。完整数据库不能裁表后继续冒用已发布身份。

先在服务器确认挂载点并创建下节的 input 目录，再在本地 PowerShell 上传。下面两项输入是实际服务器登录地址与实际数据盘目录，本轮没有执行这些传输：

```powershell
$cloudRemote = Read-Host '服务器登录地址，例如 user@address'
$cloudDataPath = Read-Host '服务器上的实际 DATA_ROOT 绝对路径'
scp -- D:/minequant/AlphaGPT-cloud.bundle D:/minequant/AlphaGPT-cloud.bundle.sha256 "${cloudRemote}:${cloudDataPath}/"
scp -- D:/minequant/AlphaGPT-task1/data/task2_reference_20261007_clean/ashare.duckdb "${cloudRemote}:${cloudDataPath}/input/ashare.duckdb"
scp -- D:/minequant/AlphaGPT/data/ashare_v2.duckdb "${cloudRemote}:${cloudDataPath}/input/ashare_v2.duckdb"
```

## 3. 确认数据盘与安装环境（Ubuntu 22.04）

先在服务器运行 `lsblk -f`、`findmnt`、`df -hT`，找到实际数据盘挂载点。下面变量中的路径由用户填写，不能使用系统盘代替。代码、Python、虚拟环境、下载缓存、数据库及结果都放到数据盘。系统盘 50 GB，数据盘 170 GB：四个独立任务会复制各自的因子缓存，应监控 `df -h` 和 `du -sh "$DATA_ROOT/runs"`；不要同时永久保存全部窗口的大缓存。输入始终只读，任务之间不共享可写缓存。

```bash
set -euo pipefail
export DATA_ROOT="/填写实际数据盘挂载点/alphagpt"
mkdir -p "$DATA_ROOT"
findmnt -T "$DATA_ROOT"
test "$(findmnt -n -o MAJ:MIN -T "$DATA_ROOT")" != "$(findmnt -n -o MAJ:MIN -T /)"
mkdir -p "$DATA_ROOT"/{bin,python,uv-cache,pip-cache,tmp,input,runs}
export TMPDIR="$DATA_ROOT/tmp"
export PIP_CACHE_DIR="$DATA_ROOT/pip-cache"
export UV_CACHE_DIR="$DATA_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$DATA_ROOT/python"
export UV_PYTHON_BIN_DIR="$DATA_ROOT/bin"
export PATH="$DATA_ROOT/bin:$PATH"

# 使用已上传的 bundle，不需要此时推送。
(cd "$DATA_ROOT" && sha256sum -c AlphaGPT-cloud.bundle.sha256)
git clone --branch codex/cloud-four-search "$DATA_ROOT/AlphaGPT-cloud.bundle" "$DATA_ROOT/code"
# 或在用户完成上述推送后使用：
# git clone --branch codex/cloud-four-search https://github.com/fanxxxks/N.git "$DATA_ROOT/code"
cd "$DATA_ROOT/code"
git status --short
git log -1 --format=%H

curl -LsSf https://astral.sh/uv/install.sh -o "$DATA_ROOT/tmp/install-uv.sh"
env UV_INSTALL_DIR="$DATA_ROOT/bin" UV_NO_MODIFY_PATH=1 sh "$DATA_ROOT/tmp/install-uv.sh"
uv python install 3.12
uv venv --python 3.12 --seed "$DATA_ROOT/venv"
source "$DATA_ROOT/venv/bin/activate"
python -m pip install -r requirements.txt -r requirements-optional.txt
# 基础声明是 CPU PyTorch；按项目声明替换成相同基版本的 CUDA wheel。
python -m pip install -r requirements-cuda.txt
python -m pip check
python scripts/freeze_lock.py --check
nvidia-smi
python scripts/check_cloud_environment.py --expected-gpus 4 --require-python312
python scripts/check_test_shards.py --check
```

uv 的安装目录与 Python 管理目录变量依据 [uv 安装说明](https://docs.astral.sh/uv/reference/installer/) 和 [uv 环境变量](https://docs.astral.sh/uv/reference/environment/)；3.12 安装命令见 [Python 安装说明](https://docs.astral.sh/uv/guides/install-python/)。GPU 检查执行真实 CUDA 小张量操作，相关运行环境要求见 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。本项目固定 torch 2.11.0+cu128 和 Optuna 4.9.0，按仓库声明安装，不另行猜测最新版本或自动更换驱动。若依赖或驱动检查失败，先修复环境，再训练。

可将 `$DATA_ROOT` 及这些缓存变量写入数据盘上的个人启动脚本；不要提交含服务器地址、令牌的机器配置，也不要复制 Windows 虚拟环境。

## 4. 上传后的数据核验

把两个文件上传到 `$DATA_ROOT/input/ashare.duckdb` 和 `$DATA_ROOT/input/ashare_v2.duckdb`。禁止将输出放在 input 下。用仓库内清单校验文件尺寸和全文件 SHA256：

```bash
cd "$DATA_ROOT/code"
export PRIMARY="$DATA_ROOT/input/ashare.duckdb"
export V2="$DATA_ROOT/input/ashare_v2.duckdb"
python - "$PRIMARY" "$V2" <<'PY'
import hashlib, json, pathlib, sys
m = json.loads(pathlib.Path('reports/cloud_20261008/data_upload_manifest.json').read_text())
for kind, name in zip(('primary', 'v2'), sys.argv[1:]):
    p = pathlib.Path(name)
    assert p.is_file() and not pathlib.Path(str(p)+'.wal').exists(), kind
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    assert p.stat().st_size == m['files'][kind]['bytes'], kind
    assert h.hexdigest() == m['files'][kind]['sha256'], kind
    print(kind, 'PASS')
PY
chmod a-w "$PRIMARY" "$V2"
python scripts/run_cloud_search.py --primary "$PRIMARY" --v2 "$V2" \
  --output "$DATA_ROOT/runs/preflight" --validate-data > "$DATA_ROOT/preflight.json"
```

这一步读完整 manifest，不训练。历史截止与数据配对错误、路径错误、输出身份冲突均会在训练前拒绝。正式 worker 还会重复只读复核；这项成本计入任务总墙钟时间。全文数据身份校验在独立数据进程内完成，再导入 torch，避免本地已观察到的原生库同进程初始化冲突。

## 5. 干净检出后的最小功能验证

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1
python -c 'from ashare_model import StackVM; import ashare_model.cloud_search; print("imports PASS")'
python -m pytest -q tests/test_cloud_search.py tests/test_run_sim.py \
  --basetemp="$DATA_ROOT/tmp/cloud-tests" --tb=short
python scripts/make_cloud_smoke_fixture.py --output "$DATA_ROOT/smoke-input"
python scripts/run_cloud_search.py --config "$DATA_ROOT/smoke-input/smoke.yaml" \
  --output "$DATA_ROOT/runs/synthetic" --budget 4 --plan
python scripts/run_cloud_search.py --config "$DATA_ROOT/smoke-input/smoke.yaml" \
  --output "$DATA_ROOT/runs/synthetic" --budget 4 --max-concurrent 2
```

最小数据是带完整发布身份的合成行情，用来检查正式入口、四方法、账户规则、输出隔离、拒绝原因与重跑，不是收益证据。该配置放宽研究筛选阈值，禁止将其用于真实数据。重复最后一条命令应全部 `skipped_completed`。测试还覆盖旧策略不会冒用、数据不匹配、资金配置不可行、超时保留已完成预算与验证选择频率。

## 6. 云端测速：先单进程，再 2 / 4 并发

使用相同的真实数据、种子 42、historical_2025、每组请求 1,000 次不同有效公式评估，初始每进程 8 个 CPU 线程。独立进程绑定独立 GPU，不采用多卡分布式训练。模型策略网络沿用现有 CPU 路径，指定 CUDA 用于因子/VM 计算；不声称整个 RL 更新都在 GPU 上。预算不等于步数乘批大小，默认每组最多预算 × 20 次提案，并受任务总时间上限约束。

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUTF8=1
BASE=(--primary "$PRIMARY" --v2 "$V2" --device cuda --threads 8 \
      --methods random,gp,tpe,rl --seeds 42 --window historical_2025 --horizon 5 \
      --budget 1000 --time-limit 7200)
python scripts/run_cloud_search.py "${BASE[@]}" --output "$DATA_ROOT/runs/bench-1" --plan \
  > "$DATA_ROOT/bench-1-plan.json"
# 同时只有一个任务：四方法依次运行在卡 0。
/usr/bin/time -v python scripts/run_cloud_search.py "${BASE[@]}" \
  --output "$DATA_ROOT/runs/bench-1" --max-concurrent 1 --gpus 0 \
  > "$DATA_ROOT/bench-1-console.json" 2> "$DATA_ROOT/bench-1-time.txt"
/usr/bin/time -v python scripts/run_cloud_search.py "${BASE[@]}" \
  --output "$DATA_ROOT/runs/bench-2" --max-concurrent 2 --gpus 0,1 \
  > "$DATA_ROOT/bench-2-console.json" 2> "$DATA_ROOT/bench-2-time.txt"
/usr/bin/time -v python scripts/run_cloud_search.py "${BASE[@]}" \
  --output "$DATA_ROOT/runs/bench-4" --max-concurrent 4 --gpus 0,1,2,3 \
  > "$DATA_ROOT/bench-4-console.json" 2> "$DATA_ROOT/bench-4-time.txt"
python scripts/run_cloud_search.py --output "$DATA_ROOT/runs/bench-4" --summarize
```

三个独立输出根避免完成跳过使并发测速失真。比较实际有效数量/搜索秒、总墙钟、并发吞吐、CPU/RAM/显存峰值及数据读取/门禁开销；`nvidia-smi`、`free -h`、`df -h` 同时观察。调线程优先根据 CPU 争用、内存和吞吐实测，不能为填满显存盲目扩大批大小。

每任务保存 `state.json`、独立 `attempt-NNNN/`、请求、日志、候选、精英、权重、resolved_config、预算曲线、验证账户、冻结选择、测试账户与结果。达到有效预算、提案次数限制、时间限制或无合格候选如实结束。模型权重明确不是完整断点；重试从随机初始化重做整个失败任务，成功任务只在身份与产物哈希均一致时跳过。修改代码/数据/预算/配置生成新身份，不能混称同一次任务恢复。

单任务或失败重跑（使用同一 BASE 参数和输出根，task ID 从预览取完整值或唯一前缀）：

```bash
python scripts/run_cloud_search.py "${BASE[@]}" --output "$DATA_ROOT/runs/bench-4" \
  --task 填写唯一任务ID --max-concurrent 1 --gpus 0 --retry
python scripts/run_cloud_search.py "${BASE[@]}" --output "$DATA_ROOT/runs/bench-4" \
  --rerun-failed --max-concurrent 4 --gpus 0,1,2,3
```

超时会终止子进程组，已写进度保留；失败记录包含每次尝试。输出非空却无任务状态、完成文件被修改、活动锁、重复身份均拒绝复用。无合格候选完成为 `no_qualified_candidate`，写原因且没有新策略文件、没有测试账户；不会读取仓库或以前任务的全局策略/精英/模型。

## 7. 测速后才决定第一轮与扩大搜索

以下是目标规模命令，不是本轮结果。先用上节实测的单组耗时、并发吞吐和服务器每小时费用计算总成本；时间上限 86,400 秒只是例子，必须结合测速调整，超时不伪造完成预算。默认 60 组（4 方法 × 3 种子 × 5 窗口），每组 10,000 次，固定每 5 日目标/执行周期：

```bash
ROUND=(--primary "$PRIMARY" --v2 "$V2" --device cuda --threads 8 \
       --methods random,gp,tpe,rl --seeds 42,43,44 --horizon 5 \
       --budget 10000 --time-limit 86400 --output "$DATA_ROOT/runs/round-1")
python scripts/run_cloud_search.py "${ROUND[@]}" --plan > "$DATA_ROOT/round-1-plan.json"
# 用户按实测成本决定后执行：
python scripts/run_cloud_search.py "${ROUND[@]}" --max-concurrent 4 --gpus 0,1,2,3
python scripts/run_cloud_search.py --output "$DATA_ROOT/runs/round-1" --summarize
# 扩大预算也必须另行估算费用；参数身份不同，使用另一输出根：
# python scripts/run_cloud_search.py "${ROUND[@]}" --budget 50000 \
#   --output "$DATA_ROOT/runs/round-50000" --max-concurrent 4 --gpus 0,1,2,3
```

每组最多保留 15 个合格候选，方向来自训练区间内部验证。真实后续验证区间用 5 万元账户扣费超额筛选，固定公式、方向和执行参数，然后测试 1 万 / 5 万 / 10 万元。测试从现金开始，各日及区间内跨年连续记账；不同滚动窗口独立起始本金，不宣称把五窗口自动拼成一条实盘账户。验证账户与测试账户相互独立。最低交易额分别 500 / 1,000 / 2,000 元，持仓 5 / 10 / 10 只，权重上限 18% / 9% / 9%，保留费用空间；高股价或整手不足仍可能导致无法成交，并记录原因。

## 8. 可选：三个预注册频率在验证段选择后冻结

只有日频、每 5 日、每 10 日三种，持仓数与阈值不再展开。此方案增加三倍搜索预算，不能与默认 60 组成本混算，也不能在已看过测试结果后决定切换到它。先注册好这项实验，再执行验证专用任务；它们只冻结候选，不评估测试：

```bash
for HORIZON in 1 5 10; do
  python scripts/run_cloud_search.py --primary "$PRIMARY" --v2 "$V2" \
    --output "$DATA_ROOT/runs/frequency-study" --device cuda --threads 8 \
    --methods random,gp,tpe,rl --seeds 42,43,44 --budget 10000 --time-limit 86400 \
    --horizon "$HORIZON" --validation-only --max-concurrent 4 --gpus 0,1,2,3
done
# 必须三种频率的验证任务全部完成、文件未改；失败先用原参数 --rerun-failed 修复。
python scripts/run_cloud_search.py --output "$DATA_ROOT/runs/frequency-study" \
  --select-frequency --frequencies 1,5,10 --plan > "$DATA_ROOT/frequency-test-plan.json"
python scripts/run_cloud_search.py --output "$DATA_ROOT/runs/frequency-study" \
  --select-frequency --frequencies 1,5,10 --max-concurrent 4 --gpus 0,1,2,3
python scripts/run_cloud_search.py --output "$DATA_ROOT/runs/frequency-study" --summarize
```

按方法/种子/窗口成组，以 5 万元验证扣费超额选频率，同分选较短周期，三种频率无合格候选就不测试。预测目标 horizon 与调仓周期始终同值。最终测试任务含冻结选择、三个来源结果哈希及三组总搜索成本，身份不同于来源训练任务。不会根据测试收益重选。主比较的 `comparisons` 用同身份只变化方法的组；频率选择可能导致方法最终周期不同，因此它的最终账户比较应同时参考冻结来源及 `frequency_search_cost`，不能仅按胜出任务的单组预算宣传优势。

## 9. 汇总与解释边界

`summary.json` 保留完整公式、方向、复杂度、请求/完成预算、提案/重复/无效数、停止原因、预算/时间曲线、资金规模账户和配置。`summary.csv` 每任务每资金规模一行，包含扣费收益、基准、超额、回撤、实际换手、交易数和费用；无交易列区分 signal_missing / execution_or_price_constraints / normal_no_trade，配置数学不可行会训练前失败。共同预算/共同搜索时间的最佳验证分数只比较真正配对的身份，缺少四方法会标 `all_four_present=false`；失败或超时取持久化进度，不补齐未完成预算。账户策略还必须检查估值完整性及实际可成交性。

基准沿用连续账户审计中的 PIT 股票池理论等权收益，明确不是可复制的小账户扣费基准。结果不是实盘盈利承诺；当前历史费用设置固定，调整后形成不同实验身份。复权财富/股数口径沿用已有执行契约，没有新增券商级逐笔分红税/公司行动现金账。已有 searcher_bench 的小窗口速度不能代替这里的冻结后账户评估。

RL 曲线按批完成时间提交，其他方法按单次有效评估完成提交；时间比较保留这一分辨率差异。共同时间之前没有已提交观察时返回 null，含义是没有该时点观察，不能当作零分或零计算量。

干净代码检出的完整测试不要求附带旧大型研究档案：档案专用测试缺档时明确 skip，提供对应文件后仍执行原有冻结哈希与内容断言；合成核心行为测试继续执行。具体通过/跳过数见 `reports/cloud_20261008/test_summary.json`。

本地已验证：四路入口、任务隔离与完整任务恢复、无候选、资金配置拒绝、历史完整数据身份/门禁，以及相关测试。云端待验证：Ubuntu/Python 3.12 全依赖安装、驱动与四张 4090 的真实 CUDA 执行、真实数据完整因子构建的显存/RAM/磁盘峰值、1/2/4 并发吞吐、1,000 次有效预算是否能在上限内完成、完整历史样本外账户表现、网络上传耗时及最终每轮费用。本文所有吞吐/预算数字明确为目标或本地合成功能观察，没有云端实测结果。
