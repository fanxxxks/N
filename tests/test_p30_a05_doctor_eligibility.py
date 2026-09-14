"""P30 A05: the layered doctor readiness report (RED first).

Executable projection of ``docs/p30_a05_doctor_eligibility_draft.md``
(§2 invariants, §3 schema, §4 refusal matrix R1–R12).  Approval: user
direct instruction 2026-09-14 「完成A05」(session record; the t6 delegation
does NOT cover p30).  Authority chain: every readiness check composes an
existing authority (G1–G8 runner, ``confirm_composite_snapshot``, D7
evidence replay, lifecycle replay, ``bound_paper_account_id``) — the
tests pin the four-status semantics and the bindings, never a second
judgement implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashare_model import lifecycle, readiness
from ashare_model.identity import content_hash
from ashare_model.run_store import RunStore

from conftest import make_test_spec
from test_run_store import LIFECYCLE_IDEA, _clean_data_report, _spec


# -- fixtures ------------------------------------------------------------------


def _pass_config() -> dict:
    return {
        "status": readiness.STATUS_PASS,
        "path": "config/ashare_config.yaml",
        "sha256": "a" * 64,
        "detail": "config loaded",
    }


def _pass_input_paths() -> dict:
    return {
        "status": readiness.STATUS_PASS,
        "detail": "primary and v2 input databases present",
        "missing": [],
    }


def _pass_store_root() -> dict:
    return {
        "root": "D:/tmp/data",
        "source": "config",
        "detail": "runtime paths resolved (audit mode)",
    }


def _pass_snapshot() -> dict:
    return {
        "status": readiness.STATUS_PASS,
        "detail": "composite snapshot confirmed (publication 7)",
        "composite_id": "c" * 64,
        "publication_id": 7,
        "primary_dataset_id": "p" * 64,
        "v2_dataset_id": "v" * 64,
    }


def _pass_gates() -> dict:
    return {
        "status": readiness.STATUS_PASS,
        "checks": [
            {"name": f"G{i} check", "ok": True, "detail": "ok"}
            for i in range(1, 9)
        ],
    }


def _pass_lifecycle_edges() -> dict:
    return {
        "status": readiness.STATUS_PASS,
        "detail": "all lifecycle edges activated",
    }


def _pass_fairness() -> dict:
    return {
        "status": readiness.STATUS_PASS,
        "detail": "probes passed; baseline recorded",
    }


def _green_paper() -> dict:
    return {
        "binding": {
            "status": readiness.STATUS_NOT_APPLICABLE,
            "detail": "no lifecycle-bound paper observation exists",
            "bound": [],
        },
        "sim_state": {
            "status": readiness.STATUS_NOT_APPLICABLE,
            "detail": "no paper state file and no binding evidence",
            "state_path": None,
            "state_exists": False,
        },
    }


def _green_dependencies() -> dict:
    return {"torch": "2.0", "numpy": "1.24", "pandas": "2.0", "deap": "1.4.4"}


def _green_inputs(**overrides) -> dict:
    """All-pass readiness inputs (no bound run, no artifacts)."""

    inputs = dict(
        now="2026-09-14T14:00:00+00:00",
        evaluation_day="20260914",
        config_status=_pass_config(),
        input_paths=_pass_input_paths(),
        dependencies=_green_dependencies(),
        model_searcher="gp",
        store_root=_pass_store_root(),
        snapshot=_pass_snapshot(),
        gates=_pass_gates(),
        artifacts=[],
        artifact_conflicts=[],
        fairness=_pass_fairness(),
        lifecycle_edges=_pass_lifecycle_edges(),
        bound_run=None,
        paper=_green_paper(),
    )
    inputs.update(overrides)
    return inputs


def _report(**overrides) -> dict:
    return readiness.build_readiness_report(**_green_inputs(**overrides))


def _dim(report: dict, name: str) -> dict:
    return report["dimensions"][name]


# --- R6: the frozen four-status aggregation rule ------------------------------


def test_r6_aggregation_rule_is_frozen():
    agg = readiness.aggregate_status
    assert agg(["pass", "pass"]) == readiness.STATUS_PASS
    assert agg(["pass", readiness.STATUS_NOT_APPLICABLE]) == readiness.STATUS_PASS
    assert agg([readiness.STATUS_NOT_APPLICABLE]) == (
        readiness.STATUS_NOT_APPLICABLE
    )
    assert agg(["pass", readiness.STATUS_UNKNOWN]) == readiness.STATUS_UNKNOWN
    assert agg(["pass", readiness.STATUS_FAIL]) == readiness.STATUS_FAIL
    assert agg(
        [readiness.STATUS_UNKNOWN, readiness.STATUS_FAIL]
    ) == readiness.STATUS_FAIL
    assert agg([]) == readiness.STATUS_NOT_APPLICABLE
    with pytest.raises(ValueError):
        agg(["green"])  # an unregistered status is fail-closed rejected


def test_r6_not_applicable_dimension_is_never_pass():
    """R6/R2: a dimension whose only check is not_applicable stays
    not_applicable — N/A can never be aggregated into a pass."""

    report = _report(artifacts=[])
    promotion = _dim(report, "promotion_ready")
    assert promotion["status"] == readiness.STATUS_NOT_APPLICABLE


# --- R1: dependencies fine, manifest broken → layered, not global -------------


def test_r1_snapshot_drift_fails_its_dimension_only():
    snapshot = {
        "status": readiness.STATUS_FAIL,
        "detail": (
            "the primary database no longer matches the composite "
            "generation 7: drifted: daily_bar; publish a new composite "
            "snapshot generation before a formal run"
        ),
        "composite_id": None,
        "publication_id": 7,
        "primary_dataset_id": "p" * 64,
        "v2_dataset_id": "v" * 64,
    }
    report = _report(snapshot=snapshot)
    assert _dim(report, "runtime")["status"] == readiness.STATUS_PASS
    assert _dim(report, "snapshot_integrity")["status"] == readiness.STATUS_FAIL
    assert _dim(report, "research_ready")["status"] == readiness.STATUS_FAIL
    blocking = report["summary"]["blocking"]
    assert any(
        entry["dimension"] == "snapshot_integrity" for entry in blocking
    )
    # The blocker names the release condition, not just a failure.
    assert any("publish a new composite" in entry["detail"] for entry in blocking)


def test_r1_unreadable_database_is_unknown_not_pass():
    snapshot = {
        "status": readiness.STATUS_UNKNOWN,
        "detail": "not computable: primary db unreadable at D:/x/ashare.duckdb",
        "composite_id": None,
        "publication_id": None,
        "primary_dataset_id": None,
        "v2_dataset_id": None,
    }
    report = _report(snapshot=snapshot)
    assert (
        _dim(report, "snapshot_integrity")["status"] == readiness.STATUS_UNKNOWN
    )


# --- R2: no account must never claim resumable --------------------------------


def test_r2_paper_state_without_binding_evidence_is_fail():
    paper = {
        "binding": {
            "status": readiness.STATUS_NOT_APPLICABLE,
            "detail": "no lifecycle-bound paper observation exists",
            "bound": [],
        },
        "sim_state": {
            "status": readiness.STATUS_FAIL,
            "detail": (
                "paper state file exists at paper/sim_portfolio_state.json "
                "but no PAPER_OBSERVING run binds an account — resuming "
                "would be an unbound continuation"
            ),
            "state_path": "paper/sim_portfolio_state.json",
            "state_exists": True,
        },
    }
    report = _report(paper=paper)
    assert _dim(report, "paper_ready")["status"] == readiness.STATUS_FAIL
    assert any(
        entry["dimension"] == "paper_ready"
        for entry in report["summary"]["blocking"]
    )


def test_r2_paper_without_state_and_without_binding_is_not_applicable():
    report = _report(paper=_green_paper())
    assert _dim(report, "paper_ready")["status"] == (
        readiness.STATUS_NOT_APPLICABLE
    )
    assert report["summary"]["by_status"]["pass"] >= 0  # counts present


# --- R3: config error but main-db gates fine → runtime fail, rest unknown -----


def test_r3_config_error_never_whitewashes_runtime():
    """R3: with a config error the doctor cannot compute gates at all —
    the data dimension must read ``unknown`` (never a stale pass), while
    the runtime dimension carries the concrete config failure.  The
    fact that the main-db gates WOULD pass lives outside the report; a
    stale gate pass must never whitewash a runtime failure."""

    report = _report(
        config_status={
            "status": readiness.STATUS_FAIL,
            "path": "config/ashare_config.yaml",
            "sha256": None,
            "detail": "config error: invalid mapping at line 3",
        },
        input_paths={
            "status": readiness.STATUS_UNKNOWN,
            "detail": "not computable: config error",
            "missing": [],
        },
        store_root={
            "root": None,
            "source": "unresolved",
            "detail": "not computable: config error",
        },
        model_searcher=None,
        gates=None,
        snapshot=None,
        fairness=None,
        lifecycle_edges={
            "status": readiness.STATUS_UNKNOWN,
            "detail": "not computable: config error",
        },
    )
    assert _dim(report, "runtime")["status"] == readiness.STATUS_FAIL
    assert _dim(report, "data_eligibility")["status"] == readiness.STATUS_UNKNOWN
    assert _dim(report, "snapshot_integrity")["status"] == readiness.STATUS_UNKNOWN
    assert _dim(report, "research_ready")["status"] == readiness.STATUS_FAIL
    blocking = report["summary"]["blocking"]
    config_blockers = [
        entry for entry in blocking if entry["check"] == "config_loaded"
    ]
    assert config_blockers and config_blockers[0]["release_condition"]


# --- R4: legacy-only artifacts are auditable, never promotable ----------------


def test_r4_legacy_only_artifacts_are_not_applicable():
    artifacts = [
        {
            "name": "strategy",
            "path": "/s",
            "exists": True,
            "legacy": True,
            "legacy_reasons": ["reward_version 9 != current 15"],
            "fields": {},
        }
    ]
    report = _report(artifacts=artifacts)
    promotion = _dim(report, "promotion_ready")
    assert promotion["status"] == readiness.STATUS_NOT_APPLICABLE
    assert "audit" in promotion["checks"][0]["detail"].lower() or (
        "never promotable" in promotion["checks"][0]["detail"].lower()
    )


def test_r4_unstamped_conflicting_artifact_fails_promotion():
    artifacts = [
        {
            "name": "strategy",
            "path": "/s",
            "exists": True,
            "legacy": False,
            "legacy_reasons": [],
            "fields": {"dataset_id": "d1"},
        }
    ]
    conflicts = [
        "strategy artifact reward_version 10 != current 15 and the "
        "artifact is not marked legacy"
    ]
    report = _report(artifacts=artifacts, artifact_conflicts=conflicts)
    assert _dim(report, "promotion_ready")["status"] == readiness.STATUS_FAIL
    assert any(
        "not marked legacy" in entry["detail"]
        for entry in report["summary"]["blocking"]
    )


# --- R5: frozen snapshot vs online freshness are separate ---------------------


def _bound_paper() -> dict:
    """A paper observation exists (the carried freshness applies)."""

    return {
        "binding": {
            "status": readiness.STATUS_PASS,
            "detail": "1 lifecycle-bound paper observation replayed",
            "bound": [
                {
                    "spec_id": "s",
                    "run_id": "r",
                    "account_id": "acc-0001",
                    "state": "PAPER_OBSERVING",
                }
            ],
        },
        "sim_state": {
            "status": readiness.STATUS_PASS,
            "detail": "paper state file present and consistent with the "
            "binding evidence",
            "state_path": "paper/sim_portfolio_state.json",
            "state_exists": True,
        },
    }


def test_r5_frozen_snapshot_pass_coexists_with_stale_online_freshness():
    gates = {
        "status": readiness.STATUS_FAIL,
        "checks": [
            {"name": f"G{i} check", "ok": True, "detail": "ok"}
            for i in range(1, 8)
        ]
        + [
            {
                "name": "G8 daily bars fresh within the open-session tolerance",
                "ok": False,
                "detail": "lag 9 > tolerance 8 sessions (today=20260914)",
            }
        ],
    }
    report = _report(gates=gates, paper=_bound_paper())
    assert _dim(report, "snapshot_integrity")["status"] == readiness.STATUS_PASS
    paper_checks = _dim(report, "paper_ready")["checks"]
    carried = [c for c in paper_checks if c["name"] == "online_data_freshness"]
    assert carried and carried[0]["status"] == readiness.STATUS_FAIL
    assert carried[0].get("carried_from") == "data_eligibility.production_gates"


def test_r5_freshness_carry_is_not_applicable_without_an_observation():
    """The G8 carry answers 'is the LIVE observation's data fresh?' —
    with no observation it is not_applicable, never a silent pass."""

    report = _report(paper=_green_paper())
    paper_checks = _dim(report, "paper_ready")["checks"]
    carried = [c for c in paper_checks if c["name"] == "online_data_freshness"]
    assert carried
    assert carried[0]["status"] == readiness.STATUS_NOT_APPLICABLE


# --- R7: the bindings block is mandatory --------------------------------------


def test_r7_bindings_are_mandatory_and_audit_mode():
    report = _report()
    bindings = report["bindings"]
    for key in (
        "readiness_version",
        "action_mode",
        "evaluated_at",
        "evaluation_day",
        "config_path",
        "config_sha256",
        "git_commit",
        "repo_root",
        "store_root",
        "store_root_source",
        "bound_run",
        "snapshot",
    ):
        assert key in bindings, f"missing binding {key}"
    assert bindings["action_mode"] == "audit"
    assert bindings["readiness_version"] == readiness.READINESS_REPORT_VERSION
    assert bindings["evaluated_at"] == "2026-09-14T14:00:00+00:00"
    assert bindings["evaluation_day"] == "20260914"
    assert bindings["bound_run"] is None
    # The report must be JSON-serializable (doctor --output contract).
    json.dumps(report)


# --- R8: the paper scan replays real runs (tmp RunStore) ----------------------


def _paper_observing_run(store: RunStore, spec, *, account_id: str) -> str:
    """A real lifecycle chain driven to ``PAPER_OBSERVING`` with a
    registered binding evidence artifact (the p8_09 machine path)."""

    handle = store.open_run(spec)
    writer = lifecycle.LifecycleWriter(handle)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    writer.qualify_data(_clean_data_report(spec))
    writer.qualify_factor_set(
        {
            "report_schema_version": 1,
            "spec_id": spec.spec_id,
            "research_domain": spec.research_domain,
            "frequency": spec.frequency,
            "horizon": int(spec.horizon),
            "feature_count": 1,
            "features": [
                {
                    "name": "RET_1",
                    "family": "price",
                    "pit_level": "EVENT",
                    "data_tier": "A",
                    "promotion_allowed": True,
                    "domain_id": "unified",
                    "domain_consistent": True,
                    "in_domain_execution": True,
                    "coverage": 0.95,
                }
            ],
            "domain_execution_legal": True,
            "weakest_data_tier": "A",
            "promotion_input_eligible": True,
            "data_eligibility": {"ok": True, "reasons": [], "checks": {}},
            "baseline_evidence": {
                "completed": True,
                "result_hash": content_hash("baseline", {"seed": 7}),
            },
            "ablation_evidence": {
                "completed": True,
                "result_hash": content_hash("ablation", {"seed": 7}),
            },
            "completed": True,
        }
    )
    writer.admit_search_plan(lifecycle.build_search_admission_evidence(spec))
    writer.qualify_oos(lifecycle.build_oos_qualification_evidence(spec))
    writer.start_paper_observation(
        lifecycle.build_paper_account_binding(
            spec,
            handle.run_id,
            account_id=account_id,
            candidate_id="cand-0001",
            observation_lineage_hash="b" * 64,
        )
    )
    run_id = handle.run_id
    handle.close()
    return run_id


def test_r8_paper_scan_finds_real_binding(tmp_path):
    store = RunStore(tmp_path / "store")
    spec = _spec()
    run_id = _paper_observing_run(store, spec, account_id="acc-0001")
    gathered = readiness.gather_paper(store.root, sim_state_path=None)
    assert gathered["binding"]["status"] == readiness.STATUS_PASS
    assert gathered["binding"]["bound"] == [
        {
            "spec_id": spec.spec_id,
            "run_id": run_id,
            "account_id": "acc-0001",
            "state": "PAPER_OBSERVING",
        }
    ]
    # A retired run is not paper-eligible: retire the run and rescan.
    handle = store.attach_run(spec, run_id)
    try:
        lifecycle.LifecycleWriter(handle).retire(reason="superseded")
    finally:
        handle.close()
    gathered_after = readiness.gather_paper(store.root, sim_state_path=None)
    assert gathered_after["binding"]["status"] == (
        readiness.STATUS_NOT_APPLICABLE
    )


def test_r8_paper_scan_without_runs_is_not_applicable(tmp_path):
    store = RunStore(tmp_path / "store")
    gathered = readiness.gather_paper(store.root, sim_state_path=None)
    assert gathered["binding"]["status"] == readiness.STATUS_NOT_APPLICABLE
    assert "no lifecycle-bound paper" in gathered["binding"]["detail"]


# --- R9: the hardcoded "current" false positive is gone -----------------------


def test_r9_lifecycle_checks_drop_the_current_hardcode(tmp_path):
    from ashare_model.research_doctor import gather_lifecycle_checks

    result = gather_lifecycle_checks(tmp_path, None)
    names = [check["name"] for check in result["checks"]]
    assert "paper_account_writable" not in names
    assert "lifecycle_edges_activated" in names
    # The vacuous pass is gone from the module's behaviour: the only
    # writable-account check lives in readiness, driven by real evidence.
    with pytest.raises(lifecycle.LifecycleError):
        lifecycle.assert_paper_account_writable("v1")


# --- R10: O5 — get_run_detail carries root provenance --------------------------


def test_r10_run_detail_provenance_matches_list_runs(tmp_path, monkeypatch):
    import ashare_model.run_store as run_store
    import webapi.service as service

    store = RunStore(tmp_path)
    spec = make_test_spec()
    handle = store.open_run(spec)
    handle.close()

    monkeypatch.setattr(
        service, "_run_store_root", lambda root=None: (tmp_path, "explicit")
    )
    listed = service.list_runs()
    detail = service.get_run_detail(None, spec.spec_id, handle.run_id)
    assert detail["runs_dir_source"] == listed["runs_dir_source"]
    assert detail["runs_dir_exists"] == listed["runs_dir_exists"]
    assert detail["runs_dir_source"] == "explicit"
    assert detail["runs_dir_exists"] is True
    assert str(tmp_path) in str(detail.get("runs_dir", "")) or (
        detail.get("runs_dir") is None
    )


# --- R11: the doctor CLI binds a run only with both identities -----------------


def test_r11_cli_requires_paired_spec_and_run():
    from ashare_model import research_doctor

    with pytest.raises(SystemExit):
        research_doctor.main(["--spec-id", "abc"])
    with pytest.raises(SystemExit):
        research_doctor.main(["--run-id", "def"])


# --- R12: the webapi readiness endpoint degrades loudly, never healthy --------


def test_r12_webapi_readiness_degrades_defensively(monkeypatch):
    import webapi.service as service

    def _broken():
        return None, None, None, None

    monkeypatch.setattr(service, "_get_configs", _broken)
    payload = service.get_readiness()
    assert payload["available"] is False
    assert payload["reason"]
    # An unavailable report is never presented as a healthy one.
    assert payload.get("readiness") is None


def test_r12_webapi_readiness_reports_layered_dimensions(monkeypatch, tmp_path):
    import webapi.service as service

    class _Cfg:
        duckdb_path = tmp_path / "missing.duckdb"
        v2_duckdb_path = str(tmp_path / "missing_v2.duckdb")
        data_dir = tmp_path / "data"

    class _Model:
        searcher = "gp"

    class _Sim:
        state_path = tmp_path / "paper" / "sim_portfolio_state.json"

    monkeypatch.setattr(
        service, "_get_configs", lambda: (_Cfg(), _Model(), None, _Sim())
    )
    monkeypatch.setattr(
        service,
        "_resolved_paths",
        lambda root=None: type(
            "_RP",
            (),
            {
                "data_dir": tmp_path / "data",
                "effective": lambda self: {"data_dir": str(tmp_path / "data")},
            },
        )(),
    )
    payload = service.get_readiness()
    assert payload["available"] is True
    dims = payload["readiness"]["dimensions"]
    assert set(dims) == {
        "runtime",
        "snapshot_integrity",
        "data_eligibility",
        "research_ready",
        "promotion_ready",
        "paper_ready",
    }
    # The heavy authorities are explicitly not computed on-demand:
    assert (
        dims["snapshot_integrity"]["status"] == readiness.STATUS_UNKNOWN
    )
    assert (
        dims["data_eligibility"]["status"] == readiness.STATUS_UNKNOWN
    )
    # The primary input database does not exist in this tmp environment:
    assert dims["runtime"]["status"] == readiness.STATUS_FAIL
