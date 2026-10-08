from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cloud_search as cloud
from make_cloud_smoke_fixture import create_fixture


def task(tmp_path):
    return {"task_id": "a" * 64, "identity": {"method": "random", "seed": 42},
        "output": str(tmp_path / "job"), "settings": {"threads": 1, "time_limit": 30}}


def test_retry_preserves_failed_attempt_and_completed_skips(tmp_path, monkeypatch):
    request = task(tmp_path)
    attempts = []
    class FakeProcess:
        def __init__(self, command, **kwargs):
            path = Path(command[-1])
            attempts.append(path)
            self.path = path
        def wait(self, timeout=None):
            if len(attempts) == 1:
                return 1
            cloud.write_json(self.path.parent / "result.json", {"outcome": "no_qualified_candidate"})
            return 0
    monkeypatch.setattr(cloud.subprocess, "Popen", FakeProcess)
    assert cloud.run_task(request, None)["status"] == "failed"
    assert cloud.run_task(request, None)["status"] == "retry_required"
    assert cloud.run_task(request, None, retry=True)["status"] == "completed"
    assert (Path(request["output"]) / "attempt-0001/request.json").exists()
    assert cloud.run_task(request, None)["status"] == "skipped_completed"
    assert len(attempts) == 2


def test_modified_completion_is_refused(tmp_path, monkeypatch):
    request = task(tmp_path)
    output = Path(request["output"])
    cloud.write_json(output / "attempt-0001/result.json", {})
    cloud.write_json(output / "state.json", {"identity": request["identity"], "status": "completed",
        "attempts": [], "latest_attempt": "attempt-0001", "artifacts": {"result.json": "wrong"}})
    with pytest.raises(ValueError, match="changed"):
        cloud.run_task(request, None)


def test_identity_conflict_and_live_lock_refused(tmp_path):
    request = task(tmp_path)
    output = Path(request["output"])
    cloud.write_json(output / "state.json", {"identity": {"seed": 7}, "status": "completed"})
    with pytest.raises(ValueError, match="identity conflict"):
        cloud.state_for(request)
    with cloud.claim_task(output):
        with pytest.raises(ValueError, match="already running"):
            with cloud.claim_task(output):
                pass


def test_unowned_outputs_and_timeout_preserve_partial_budget(tmp_path, monkeypatch):
    request = task(tmp_path)
    output = Path(request["output"])
    output.mkdir()
    (output / "old_strategy.json").write_text("{}")
    with pytest.raises(ValueError, match="unowned"):
        cloud.state_for(request)
    (output / "old_strategy.json").unlink()
    request["identity"].update(budget=8, seed=42, window={"name": "small"})
    class SlowProcess:
        def __init__(self, command, **kwargs):
            self.path = Path(command[-1])
            cloud.write_json(self.path.parent / "search_progress.json", {"search": {
                "requested_budget": 8, "consumed_budget": 2, "proposal_count": 5,
                "invalid_proposals": 1, "semantic_duplicates": 2}, "curve": [], "search_seconds": 20})
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("worker", timeout)
    monkeypatch.setattr(cloud.subprocess, "Popen", SlowProcess)
    monkeypatch.setattr(cloud, "kill_worker", lambda _: None)
    assert cloud.run_task(request, None)["status"] == "timeout"
    rows = cloud.summarize(tmp_path)["rows"]
    assert rows[0]["search"]["consumed_budget"] == 2
    assert rows[0]["search"]["stop_reason"] == "timeout"
    assert (output / "attempt-0001/request.json").exists()


def test_plan_identity_tracks_effective_inputs_but_not_output(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "code_identity", lambda: {"commit": "a" * 40})
    (tmp_path / "a.db").write_bytes(b"primary")
    (tmp_path / "b.db").write_bytes(b"v2")
    config = dict(primary=str(tmp_path / "a.db"), v2=str(tmp_path / "b.db"), output=str(tmp_path / "out"),
        methods=["random"], seeds=[42], budget=4, windows=[{"name": "a"}])
    first = cloud.make_plan(config)[0]["task_id"]
    assert cloud.make_plan({**config, "output": str(tmp_path / "elsewhere")})[0]["task_id"] == first
    for key, value in (("budget", 5), ("seeds", [7]), ("windows", [{"name": "b"}])):
        assert cloud.make_plan({**config, key: value})[0]["task_id"] != first
    (tmp_path / "a.db").write_bytes(b"different")
    assert cloud.make_plan(config)[0]["task_id"] != first


@pytest.fixture(scope="module")
def fixture_config(tmp_path_factory):
    return create_fixture(tmp_path_factory.mktemp("cloud-input-parent") / "input")


def cli(*arguments):
    return subprocess.run([sys.executable, str(ROOT / "scripts/run_cloud_search.py"), *map(str, arguments)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        env={**__import__("os").environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1"}, timeout=240)


def test_real_cli_four_methods_isolated_and_resume(fixture_config, tmp_path):
    output = tmp_path / "runs"
    args = ["--config", fixture_config, "--output", output, "--budget", "4", "--max-concurrent", "2"]
    preview = cli(*args, "--plan")
    assert preview.returncode == 0, preview.stderr
    assert json.loads(preview.stdout)["task_count"] == 4
    assert not output.exists()
    execution = cli(*args)
    assert execution.returncode == 0, execution.stdout + execution.stderr
    results = list(output.glob("*/attempt-0001/result.json"))
    assert len(results) == 4
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in results]
    assert {p["method"] for p in payloads} == cloud.METHODS
    for payload, path in zip(payloads, results):
        assert 0 <= payload["search"]["consumed_budget"] <= 4
        assert (path.parent / "candidates.json").exists()
        assert (path.parent / "elite_archive.json").exists()
        assert (path.parent / "model.pt").exists()
        if payload["outcome"] == "no_qualified_candidate":
            assert not (path.parent / "strategy.json").exists()
        else:
            assert {r["capital"] for r in payload["test_accounts"]} == {10000, 50000, 100000}
            assert payload["selected"]["frozen_before_test"]
    repeated = cli(*args)
    assert repeated.returncode == 0, repeated.stderr
    assert all(r["status"] == "skipped_completed" for r in json.loads(repeated.stdout))


def test_no_eligible_candidate_never_reads_stale_strategy(fixture_config, tmp_path):
    settings = yaml.safe_load(fixture_config.read_text(encoding="utf-8"))
    settings["reward"]["min_val_reward"] = 1e9
    settings["methods"] = ["random"]
    config = tmp_path / "reject.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    output = tmp_path / "runs"
    # A global old strategy must not influence the task.
    (tmp_path / "best_ashare_strategy.json").write_text('{"old":true}')
    execution = cli("--config", config, "--output", output, "--budget", "3")
    assert execution.returncode == 0, execution.stdout + execution.stderr
    result_path = next(output.glob("*/attempt-0001/result.json"))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["outcome"] == "no_qualified_candidate"
    assert payload["test_accounts"] == []
    assert payload["no_candidate_reason"]
    assert not (result_path.parent / "strategy.json").exists()


def test_mismatched_data_and_infeasible_account_refuse(fixture_config, tmp_path):
    from ashare_model.cloud_search import account_config
    settings = yaml.safe_load(fixture_config.read_text(encoding="utf-8"))
    settings["accounts"]["10000"]["min_trade_amount"] = 5000
    with pytest.raises(ValueError, match="configuration_infeasible"):
        account_config(settings, 10000, settings["windows"][0])
    settings["dataset_id"] = "wrong"
    config = tmp_path / "wrong.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    execution = cli("--config", config, "--output", tmp_path / "runs", "--methods", "random", "--budget", "1")
    assert execution.returncode == 1
    log = next((tmp_path / "runs").glob("*/attempt-0001/run.log")).read_text(encoding="utf-8")
    assert "dataset identity mismatch" in log
    assert not list((tmp_path / "runs").glob("*/attempt-0001/model.pt"))


def test_frequency_is_selected_before_any_test(fixture_config, tmp_path):
    output = tmp_path / "runs"
    for horizon in (1, 5, 10):
        execution = cli("--config", fixture_config, "--output", output, "--methods", "random",
            "--budget", "4", "--horizon", horizon, "--validation-only")
        assert execution.returncode == 0, execution.stdout + execution.stderr
    sources = [json.loads(p.read_text(encoding="utf-8")) for p in output.glob("*/attempt-0001/result.json")]
    assert len(sources) == 3
    assert all(r["test_accounts"] == [] for r in sources)
    jobs = cloud.frequency_tasks(output, [1, 5, 10])
    assert len(jobs) == 1
    chosen = max((r for r in sources if r["selected"]),
        key=lambda r: (r["selected"]["validation_account"]["net_excess_return"], -r["identity"]["config"]["horizon"]))
    assert jobs[0]["identity"]["frequency_selection"]["selected_task_id"] == chosen["task_id"]
    selected = cli("--output", output, "--select-frequency", "--frequencies", "1,5,10")
    assert selected.returncode == 0, selected.stdout + selected.stderr
    result = json.loads((Path(jobs[0]["output"]) / "attempt-0001/result.json").read_text(encoding="utf-8"))
    assert result["outcome"] == "frozen_frequency_test_completed"
    assert len(result["test_accounts"]) == 3
    assert result["frequency_search_cost"]["requested_budget"] == 12
    assert result["selected"]["target_horizon"] == chosen["selected"]["target_horizon"]


def test_rl_duplicates_do_not_complete_budget_and_sampling_is_bounded(fixture_config, tmp_path):
    settings = yaml.safe_load(fixture_config.read_text(encoding="utf-8"))
    settings["methods"] = ["rl"]
    settings["feature_names"] = ["RET_1"]
    # Two tokens permit only this feature plus one unary operator, a formula
    # space smaller than 64 classes. Sampling must continue through repeats.
    settings["model"]["max_formula_len"] = 2
    settings["attempt_multiplier"] = 2
    config = tmp_path / "duplicates.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    output = tmp_path / "runs"
    execution = cli("--config", config, "--output", output, "--budget", "64")
    assert execution.returncode == 0, execution.stdout + execution.stderr
    result = json.loads(next(output.glob("*/attempt-0001/result.json")).read_text(encoding="utf-8"))
    search = result["search"]
    assert search["consumed_budget"] < 64
    assert search["proposal_count"] == 128
    assert search["semantic_duplicates"] > 0
    assert search["proposal_count"] == (search["consumed_budget"] +
        search["semantic_duplicates"] + search["invalid_proposals"])
    assert search["stop_reason"] != "budget_exhausted"
