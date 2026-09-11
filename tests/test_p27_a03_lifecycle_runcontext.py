"""P27 A03: the formal-entry lifecycle RunContext contract (RED first).

Executable projection of ``docs/p27_a03_lifecycle_runcontext_draft.md``
(§3 invariants I1–I13, §6 refusal matrix R1–R16).  Authority chain: user
delegation → captain approval (``reports/alphagpt-ab/
00_authority_exception_record.md`` §11, ruling 1) → user confirmation
(§4ter item 4).  Scenario tags in each test name map to the §6 matrix.

CLI/service-boundary refusal tests also pin the I6 ORDER invariant: the
refusal must happen before the loader/search/account-write side effects,
observed through a fail-loud stub on the loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashare_model import lifecycle
from ashare_model.identity import content_hash
from ashare_model.run_store import RunSpecMismatchError, RunStore, RunStoreError

from conftest import make_test_spec, write_current_strategy
from test_run_store import LIFECYCLE_IDEA, _clean_data_report, _spec


def _genesis_run(store: RunStore, spec=None, run_id: str | None = None):
    """A preregistered run: genesis recorded, state ``SPEC_LOCKED``."""

    spec = spec or _spec()
    handle = store.open_run(spec, run_id=run_id)
    writer = lifecycle.LifecycleWriter(handle)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    return handle, writer, spec


def _factor_report(spec):
    """A complete, passing §5.2 FactorQualificationReport payload."""

    return {
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


# --- R1/R3/R4/R5/R10/R15: the formal gate refuses without real identity -----


def test_r1_formal_mode_refuses_missing_identity(tmp_path):
    """R1/I2: formal mode with no spec/run identity must refuse — the
    pre-A03 ``return None`` vacuous pass is gone; ``None`` only exists in
    the explicitly declared non-formal modes."""

    data_dir = tmp_path / "store-root"
    data_dir.mkdir()
    for entry in ("train", "evaluation", "paper_sim", "backtest"):
        with pytest.raises(lifecycle.LifecycleError, match="identity"):
            lifecycle.assert_formal_run_lifecycle(entry, data_dir)
        with pytest.raises(lifecycle.LifecycleError):
            lifecycle.assert_formal_run_lifecycle(entry, data_dir, None, None)
    # The sanctioned None path: explicit non-formal declaration.
    assert (
        lifecycle.assert_formal_run_lifecycle(
            "train", data_dir, action_mode="engineering"
        )
        is None
    )
    assert (
        lifecycle.assert_formal_run_lifecycle(
            "train", data_dir, action_mode="audit"
        )
        is None
    )


def test_r10_unknown_action_mode_refuses(tmp_path):
    """R10/I3: an unregistered action mode is fail-closed rejected."""

    with pytest.raises(lifecycle.LifecycleError, match="action mode"):
        lifecycle.assert_formal_run_lifecycle(
            "train", tmp_path, "s" * 8, "r" * 8, action_mode="yolo"
        )


def test_r3_formal_mode_refuses_missing_event_log(tmp_path):
    """R3/I2/I10: a run directory whose lifecycle chain is empty is
    refused in formal mode; non-formal modes keep the explicit legacy
    passthrough (the only sanctioned ``None``)."""

    store = RunStore(tmp_path)
    spec = _spec()
    handle = store.open_run(spec)  # layout created, genesis NOT recorded
    run_id = handle.run_id
    handle.close()
    with pytest.raises(lifecycle.LifecycleError, match="event"):
        lifecycle.assert_formal_run_lifecycle(
            "train", tmp_path, spec.spec_id, run_id
        )
    with pytest.raises(lifecycle.LifecycleError):
        lifecycle.assert_formal_run_lifecycle(
            "evaluation", tmp_path, spec.spec_id, run_id
        )
    assert (
        lifecycle.assert_formal_run_lifecycle(
            "train", tmp_path, spec.spec_id, run_id,
            action_mode="engineering",
        )
        is None
    )


def test_r4_evaluation_entry_refuses_below_search_plan(tmp_path):
    """R4/I4/I5: an evaluation entry against a run at ``IDEA``/
    ``SPEC_LOCKED`` is refused (the protocol consumes a fully admitted
    search plan).  The train floor is separately fixed at ``SPEC_LOCKED``
    (p27 §3/I5) — a run at ``IDEA`` is below it."""

    store = RunStore(tmp_path)
    handle, writer, spec = _genesis_run(store)  # SPEC_LOCKED
    handle.close()
    with pytest.raises(lifecycle.LifecycleError, match="SEARCH_PLAN_ADMITTED"):
        lifecycle.assert_formal_run_lifecycle(
            "evaluation", tmp_path, spec.spec_id, handle.run_id
        )
    # train's floor: SPEC_LOCKED is enough at establishment.
    state = lifecycle.assert_formal_run_lifecycle(
        "train", tmp_path, spec.spec_id, handle.run_id
    )
    assert state == "SPEC_LOCKED"
    # IDEA-only run: below every formal floor.
    handle2 = store.open_run(_spec())
    writer2 = lifecycle.LifecycleWriter(handle2)
    writer2.record_idea(LIFECYCLE_IDEA)
    handle2.close()
    with pytest.raises(lifecycle.LifecycleError):
        lifecycle.assert_formal_run_lifecycle(
            "train", tmp_path, spec.spec_id, handle2.run_id
        )


def test_r5_terminal_run_refused_at_every_entry(tmp_path):
    """R5/I4: a run in ``FAILED``/``RETIRED`` has no out edges — no formal
    entry may continue it."""

    store = RunStore(tmp_path)
    for terminal in ("failed", "retired"):
        handle, writer, spec = _genesis_run(store)
        if terminal == "failed":
            writer.fail(reason="loader crashed")
        else:
            writer.retire(reason="superseded")
        handle.close()
        for entry in ("train", "evaluation", "paper_sim", "backtest"):
            with pytest.raises(lifecycle.LifecycleError, match="terminal"):
                lifecycle.assert_formal_run_lifecycle(
                    entry, tmp_path, spec.spec_id, handle.run_id
                )


def test_r15_paper_sim_requires_exactly_paper_observing(tmp_path):
    """R15/I4: the paper entry keeps its exact-state semantics."""

    store = RunStore(tmp_path)
    handle, writer, spec = _genesis_run(store)  # SPEC_LOCKED ≠ observing
    handle.close()
    with pytest.raises(lifecycle.LifecycleError, match="PAPER_OBSERVING"):
        lifecycle.assert_formal_run_lifecycle(
            "paper_sim", tmp_path, spec.spec_id, handle.run_id
        )


# --- RunContext: the single identity object formal entries receive (I1) -----


def test_run_context_attach_is_the_single_formal_path(tmp_path):
    """I1/I12: formal entries receive the same RunContext; attach replays
    the chain, refuses spec mismatch (R2), and exposes the live state."""

    from ashare_model.lifecycle import RunContext

    store = RunStore(tmp_path)
    handle, writer, spec = _genesis_run(store)
    run_id = handle.run_id
    handle.close()
    context = RunContext.attach(
        store,
        spec_id=spec.spec_id,
        run_id=run_id,
        entry_type="train",
        action_mode="research",
        spec=spec,
    )
    try:
        assert context.state == "SPEC_LOCKED"
        assert context.spec.spec_id == spec.spec_id
        assert context.run_id == run_id
        assert context.entry_type == "train"
        assert context.action_mode == "research"
    finally:
        context.close()


def test_r2_attach_refuses_spec_identity_mismatch(tmp_path):
    """R2/I12/I13: a context whose resolved spec does not match the
    stored run identity is refused (no silent re-derivation)."""

    from ashare_model.lifecycle import RunContext

    store = RunStore(tmp_path)
    handle, writer, spec = _genesis_run(store)
    run_id = handle.run_id
    handle.close()
    other = make_test_spec(dataset_id="ds-other")
    with pytest.raises(
        (RunSpecMismatchError, RunStoreError, lifecycle.LifecycleError)
    ):
        RunContext.attach(
            store,
            spec_id=other.spec_id,
            run_id=run_id,
            entry_type="train",
            action_mode="research",
            spec=other,
        )


def test_r3_context_attach_refuses_empty_chain(tmp_path):
    from ashare_model.lifecycle import RunContext

    store = RunStore(tmp_path)
    spec = _spec()
    handle = store.open_run(spec)
    run_id = handle.run_id
    handle.close()
    with pytest.raises(lifecycle.LifecycleError, match="event"):
        RunContext.attach(
            store,
            spec_id=spec.spec_id,
            run_id=run_id,
            entry_type="train",
            action_mode="research",
            spec=spec,
        )


def test_r10_context_attach_refuses_unknown_mode(tmp_path):
    from ashare_model.lifecycle import RunContext

    store = RunStore(tmp_path)
    handle, writer, spec = _genesis_run(store)
    run_id = handle.run_id
    handle.close()
    with pytest.raises(lifecycle.LifecycleError, match="action mode"):
        RunContext.attach(
            store,
            spec_id=spec.spec_id,
            run_id=run_id,
            entry_type="train",
            action_mode="informal",
            spec=spec,
        )


def test_i5_search_action_gate_requires_full_admission(tmp_path):
    """I5: the search action is a different action from loading — it
    requires the run to have consumed data + factor + search-plan
    qualification (≥ ``SEARCH_PLAN_ADMITTED``) at action time."""

    from ashare_model.lifecycle import RunContext

    store = RunStore(tmp_path)
    handle, writer, spec = _genesis_run(store)  # SPEC_LOCKED
    run_id = handle.run_id
    handle.close()  # release the writer lock before attaching the context
    context = RunContext.attach(
        store,
        spec_id=spec.spec_id,
        run_id=run_id,
        entry_type="train",
        action_mode="research",
        spec=spec,
    )
    try:
        with pytest.raises(
            lifecycle.LifecycleError, match="SEARCH_PLAN_ADMITTED"
        ):
            context.assert_search_ready()
    finally:
        context.close()

    # A run driven through the full qualification chain passes the gate.
    handle2 = store.attach_run(spec, run_id)
    writer2 = lifecycle.LifecycleWriter(handle2)
    writer2.qualify_data(_clean_data_report(spec))
    writer2.qualify_factor_set(_factor_report(spec))
    writer2.admit_search_plan(lifecycle.build_search_admission_evidence(spec))
    handle2.close()
    context2 = RunContext.attach(
        store,
        spec_id=spec.spec_id,
        run_id=run_id,
        entry_type="train",
        action_mode="research",
        spec=spec,
    )
    try:
        context2.assert_search_ready()  # no raise
        assert context2.state == "SEARCH_PLAN_ADMITTED"
    finally:
        context2.close()


# --- R8/I8: protected OOS isolation gains the action-mode dimension ---------


def test_r8_protected_oos_refuses_non_formal_modes_and_low_states():
    from ashare_model.lifecycle import assert_protected_oos_access

    # research/promotion at or after admission: the legal window.
    assert_protected_oos_access("SEARCH_PLAN_ADMITTED", "research")
    # Engineering/audit may never read the locked window, in any state.
    with pytest.raises(lifecycle.LifecycleError):
        assert_protected_oos_access(
            "PROMOTED", "research", action_mode="engineering"
        )
    with pytest.raises(lifecycle.LifecycleError):
        assert_protected_oos_access(
            "PROMOTED", "research", action_mode="audit"
        )
    # Pre-admission states stay refused (existing p8 §6 rule).
    with pytest.raises(lifecycle.LifecycleError):
        assert_protected_oos_access("SPEC_LOCKED", "research")


# --- R6/R16/I9: paper account identity is consumed, never hardcoded ---------


def test_r6_paper_account_mismatch_refused(tmp_path):
    from ashare_model.identity import candidate_id
    from ashare_model.lifecycle import RunContext

    store = RunStore(tmp_path)
    handle, writer, spec = _genesis_run(store)
    writer.qualify_data(_clean_data_report(spec))
    writer.qualify_factor_set(_factor_report(spec))
    writer.admit_search_plan(lifecycle.build_search_admission_evidence(spec))
    writer.qualify_oos(lifecycle.build_oos_qualification_evidence(spec))
    writer.start_paper_observation(
        lifecycle.build_paper_account_binding(
            spec,
            handle.run_id,
            account_id="acct-bound",
            candidate_id=candidate_id(spec.spec_id, [3, 1, 2], 1),
            observation_lineage_hash="e" * 64,
        )
    )
    run_id = handle.run_id
    handle.close()
    context = RunContext.attach(
        store,
        spec_id=spec.spec_id,
        run_id=run_id,
        entry_type="paper_sim",
        action_mode="paper",
        spec=spec,
    )
    try:
        assert context.state == "PAPER_OBSERVING"
        assert lifecycle.bound_paper_account_id(context.handle) == "acct-bound"
        context.assert_paper_account("acct-bound")  # no raise
        with pytest.raises(lifecycle.LifecycleError, match="account"):
            context.assert_paper_account("acct-other")
    finally:
        context.close()


def test_r16_run_sim_has_no_hardcoded_current_account():
    """R16/I9: the run_sim entry consumes the real account identity; the
    literal ``assert_paper_account_writable("current")`` call is gone."""

    source = (
        Path(__file__).resolve().parents[1] / "ashare_trading" / "run_sim.py"
    ).read_text(encoding="utf-8")
    assert 'assert_paper_account_writable("current")' not in source
    assert "assert_paper_account_writable('current')" not in source


# --- R7/I10: current schema is not lifecycle qualification (backtest) -------


def test_r7_backtest_formal_refuses_current_schema_without_chain(tmp_path):
    """R7/I10: a schema-v2 strategy whose source run has no lifecycle
    evidence is refused as a formal backtest source (schema ≠ qualification)."""

    from ashare_model.lifecycle import RunContext

    store = RunStore(tmp_path)
    spec = _spec()
    handle = store.open_run(spec)  # no genesis: no qualification chain
    run_id = handle.run_id
    handle.close()
    with pytest.raises(lifecycle.LifecycleError):
        RunContext.attach(
            store,
            spec_id=spec.spec_id,
            run_id=run_id,
            entry_type="backtest",
            action_mode="research",
            spec=spec,
        )


# --- R9/R12/R14: entry-boundary order, refusal residue, selfcheck mode ------


class _ExpectRefusal:
    """Assert an entry refuses with SystemExit, optionally message-matched."""

    def __init__(self, message_part: str | None = None):
        self.message_part = message_part

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError("expected the entry to refuse")
        if not issubclass(exc_type, SystemExit):
            return False  # propagate unexpected exceptions
        if self.message_part is not None:
            assert self.message_part in str(exc), (
                f"refusal message missing {self.message_part!r}: {exc!r}"
            )
        return True  # swallow the expected SystemExit


def _populated_config(tmp_path, populated_db):
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(populated_db.data_dir),
                "duckdb_path": str(populated_db.duckdb_path),
                "parquet_dir": str(populated_db.parquet_dir),
                "index_codes": ["000300.SH"],
                "min_listed_sessions": 1,
                "model": {"max_formula_len": 6},
                "protocol": {
                    "folds": [
                        {"train_end": "2024-01-10", "test_end": "2024-01-25"}
                    ],
                    "seeds": [42],
                    "screening": {"steps": 1, "batch_size": 8},
                    "random_samples": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return cfg_path


def test_r9_evaluation_formal_refuses_before_loader(
    monkeypatch, tmp_path, populated_db, production_gate_asof
):
    """R9/I6: the formal evaluation entry without real identity refuses
    BEFORE ``AshareDataLoader.load_data`` is ever called."""

    from ashare_model import evaluation
    from ashare_model.data_loader import AshareDataLoader

    def _forbidden(self, *a, **k):
        raise AssertionError("load_data ran before the lifecycle refusal")

    monkeypatch.setattr(AshareDataLoader, "load_data", _forbidden)
    cfg_path = _populated_config(tmp_path, populated_db)
    with _ExpectRefusal():
        evaluation.main(
            [
                "--config", str(cfg_path),
                "--output", str(tmp_path / "out.json"),
                "--min-eligible", "3",
                "--ledger", str(tmp_path / "ledger.jsonl"),
                "--regime", str(tmp_path / "holdout_registry.json"),
            ]
        )


def test_r9_train_formal_refuses_before_loader(
    monkeypatch, tmp_path, populated_db, production_gate_asof
):
    """R9/I6: same order invariant for the formal train entry."""

    from ashare_model import train as train_module
    from ashare_model.data_loader import AshareDataLoader

    def _forbidden(self, *a, **k):
        raise AssertionError("load_data ran before the lifecycle refusal")

    monkeypatch.setattr(AshareDataLoader, "load_data", _forbidden)
    cfg_path = _populated_config(tmp_path, populated_db)
    monkeypatch.setattr(
        "sys.argv", ["train", "--config", str(cfg_path), "--min-eligible", "3"]
    )
    with _ExpectRefusal():
        train_module.main()


def test_r12_refused_formal_run_leaves_no_business_artifacts(
    tmp_path, populated_db, production_gate_asof
):
    """R12/I11: a refused formal run may keep its (pre-existing)
    lifecycle chain but gains no successful business artifact."""

    from ashare_model import evaluation

    store_root = populated_db.data_dir
    store = RunStore(store_root)
    handle, writer, spec = _genesis_run(store)  # SPEC_LOCKED: below floor
    run_id = handle.run_id
    handle.close()

    cfg_path = _populated_config(tmp_path, populated_db)
    out_path = tmp_path / "protocol_result.json"
    with _ExpectRefusal("SEARCH_PLAN_ADMITTED"):
        evaluation.main(
            [
                "--config", str(cfg_path),
                "--output", str(out_path),
                "--min-eligible", "3",
                "--ledger", str(tmp_path / "ledger.jsonl"),
                "--regime", str(tmp_path / "holdout_registry.json"),
                "--spec-id", spec.spec_id,
                "--run-id", run_id,
            ]
        )
    assert not out_path.exists()
    index = json.loads(
        (handle.run_dir / "index.json").read_text(encoding="utf-8")
    )
    assert index["artifacts"] == []


def test_r14_selfcheck_requires_engineering_declaration(
    monkeypatch, tmp_path, populated_db, production_gate_asof
):
    """R14/I5: the ``--selfcheck`` dry run is a different action — it must
    be declared engineering, never inherit formal research identity."""

    from ashare_model import evaluation
    from ashare_model.data_loader import AshareDataLoader

    def _forbidden(self, *a, **k):
        raise AssertionError("selfcheck ran without its mode declaration")

    monkeypatch.setattr(AshareDataLoader, "load_data", _forbidden)
    cfg_path = _populated_config(tmp_path, populated_db)
    with _ExpectRefusal("engineering"):
        evaluation.main(
            [
                "--selfcheck",
                "--config", str(cfg_path),
                "--output", str(tmp_path / "out.json"),
                "--min-eligible", "3",
                "--ledger", str(tmp_path / "ledger.jsonl"),
                "--regime", str(tmp_path / "holdout_registry.json"),
            ]
        )


# --- R11/I7: artifact identity comes from the execution-time context -------


def test_r11_formal_artifact_write_uses_context_identity(tmp_path):
    """I7: the formal writer stamps the open context's identity and
    refuses a payload that contradicts it; no second run is minted."""

    from ashare_model.artifact_schemas import StrategyArtifact
    from ashare_model.artifact_writer import write_boundary_artifact
    from ashare_model.identity import candidate_id
    from ashare_model.lifecycle import RunContext

    store = RunStore(tmp_path / "ctx-store")
    handle, writer, spec = _genesis_run(store)
    run_id = handle.run_id
    handle.close()  # release the writer lock before attaching the context
    context = RunContext.attach(
        store,
        spec_id=spec.spec_id,
        run_id=run_id,
        entry_type="train",
        action_mode="research",
        spec=spec,
    )
    try:
        # A real, current-schema strategy payload produced through the
        # sanctioned fixture writer, stripped of the identity keys the
        # formal writer stamps from the open context.
        persisted = write_current_strategy(
            tmp_path / "mirror",
            dataset_id=spec.dataset_id,
            spec=spec,
        )
        payload = {
            key: value
            for key, value in persisted.items()
            if key
            not in {"spec_id", "run_id", "candidate_id", "artifact_id"}
        }
        tokens = list(payload["formula"])
        cand = candidate_id(spec.spec_id, tokens, int(payload["direction"]))
        record = write_boundary_artifact(
            context.handle,
            artifact_type="strategy",
            model_cls=StrategyArtifact,
            payload=payload,
            candidate_id=cand,
        )
        index = context.handle.load_index()
        # The genesis evidence artifacts precede it; the context write is
        # the newest registration in the SAME run (no second run minted).
        assert index["artifacts"][-1]["artifact_id"] == record.artifact_id
        assert record.artifact_id in [
            a["artifact_id"] for a in index["artifacts"]
        ]
        assert len(index["artifacts"]) == 3  # idea + spec-lock + strategy
        # A payload carrying a foreign identity is refused (I7/R11).
        conflicting = dict(payload, run_id="f" * 8)
        with pytest.raises(Exception, match="conflicts"):
            write_boundary_artifact(
                context.handle,
                artifact_type="strategy",
                model_cls=StrategyArtifact,
                payload=conflicting,
                candidate_id=cand,
            )
    finally:
        context.close()


def test_r13_single_runcontext_type_across_entries():
    """R13/I1: the wired entries share the one RunContext type — the p27
    §6 R13 import-surface guard, plus the runtime type check the trainer
    performs on a passed context."""

    root = Path(__file__).resolve().parents[1]
    for rel in (
        "ashare_model/train.py",
        "ashare_model/evaluation.py",
        "ashare_model/backtest.py",
        "ashare_trading/run_sim.py",
        "ashare_model/train_artifacts.py",
    ):
        source = (root / rel).read_text(encoding="utf-8")
        assert "RunContext" in source, f"{rel} does not consume RunContext"

    import inspect

    from ashare_model.lifecycle import RunContext
    from ashare_model.train import AshareTrainer

    assert (
        "run_context" in inspect.signature(AshareTrainer.__init__).parameters
    )
    assert RunContext is lifecycle.RunContext
