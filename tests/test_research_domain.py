"""P6 contract tests: research-domain split by prediction horizon.

Assertion source: ``docs/p6_research_domain_contract.md`` sections 1-4.
These tests pin the domain registry, the exhaustive feature partition, the
legal execution points, the monthly rebalance cadence, the per-domain
config defaults, the search-space restriction and the artifact provenance.
Expected values are derived from the contract, never from implementation
output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ashare_data.config import (
    make_backtest_config,
    make_protocol_config,
    make_reward_config,
)
from ashare_model.alphagpt import build_action_mask
from ashare_model.evaluation import Fold, run_protocol, search_window_id
from ashare_model.research_domain import (
    RESEARCH_DOMAIN_VERSION,
    RESEARCH_DOMAINS,
    UNIFIED_DOMAIN_ID,
    domain_of_feature,
    feature_token_ids,
    restrict_tensor,
)
from ashare_model.time_contract import FoldTimeContract
from ashare_model.train import sample_random_formulas
from ashare_model.vocab import FEATURE_NAMES, FORMULA_VOCAB
from ashare_portfolio.rebalance import RebalancePolicy

DOMAIN_IDS = ("short_price_volume", "medium_cross_section", "slow_fundamental")
TEST_SPEC_ID = "a" * 64
TEST_RUN_ID = "b" * 32


def test_domains_partition_vocabulary_exhaustively():
    """§1.1: every live vocabulary feature belongs to exactly one domain;
    domains are mutually exclusive; deprecated NORTHBOUND_CHG is owned by
    no domain."""
    assert UNIFIED_DOMAIN_ID not in RESEARCH_DOMAINS
    domains = list(RESEARCH_DOMAINS.values())
    assert len(domains) == 3
    assert [d.id for d in domains] == list(DOMAIN_IDS)
    owned: set[str] = set()
    for domain in domains:
        assert len(domain.features) == len(set(domain.features))
        assert domain.id != UNIFIED_DOMAIN_ID
        unknown = [name for name in domain.features if name not in FEATURE_NAMES]
        assert unknown == []
        overlap = owned & set(domain.features)
        assert overlap == set()
        owned |= set(domain.features)
    live = {name for name in FEATURE_NAMES if name != "NORTHBOUND_CHG"}
    assert owned == live
    # P9 §5 (whitelist §10.1 case 2, contract APPROVED): 62 + 11 new
    # features minus the deprecated neutral member.  P13 §5.5 (whitelist
    # §10.1 case 2, docs/p13_fundamental_fields_contract.md APPROVED):
    # family ⑤ appends four more -> 76.
    assert len(live) == 76
    with pytest.raises(ValueError, match="NORTHBOUND_CHG"):
        domain_of_feature("NORTHBOUND_CHG")


def test_domain_defaults_are_legal():
    """§1/§1.2: default execution point passes legality; default baselines
    stay inside the domain; per-domain reward/turnover parameters are
    positive and declared."""
    for domain in RESEARCH_DOMAINS.values():
        assert domain.is_legal_execution(
            domain.default_frequency, domain.default_horizon
        )
        frequency, horizon = domain.default_execution()
        assert (frequency, horizon) == (
            domain.default_frequency,
            domain.default_horizon,
        )
        assert set(domain.baseline_signals) <= set(domain.features)
        assert domain.baseline_signals
        assert domain.turnover_budget > 0.0
        assert domain.cost_weight > 0.0


def test_legal_executions_respect_non_overlap():
    """§1.2: legal execution points are exactly the contract table; every
    legal point is constructible under the P3 non-overlap rule."""
    short = RESEARCH_DOMAINS["short_price_volume"]
    medium = RESEARCH_DOMAINS["medium_cross_section"]
    slow = RESEARCH_DOMAINS["slow_fundamental"]

    assert short.is_legal_execution("daily", 1)
    assert not short.is_legal_execution("daily", 2)
    assert short.is_legal_execution("every_5_days", 1)
    assert short.is_legal_execution("every_5_days", 5)
    assert not short.is_legal_execution("every_5_days", 6)

    assert medium.is_legal_execution("every_5_days", 5)
    assert medium.is_legal_execution("every_10_days", 5)
    assert medium.is_legal_execution("every_10_days", 10)
    assert not medium.is_legal_execution("every_10_days", 4)
    assert not medium.is_legal_execution("every_10_days", 11)
    assert not medium.is_legal_execution("every_5_days", 6)
    assert not medium.is_legal_execution("daily", 1)

    assert slow.is_legal_execution("every_20_days", 20)
    assert not slow.is_legal_execution("every_20_days", 21)
    # monthly legality is calendar-dependent (runtime check on the date
    # axis); the static pair is accepted.
    assert slow.is_legal_execution("monthly", 20)
    assert not slow.is_legal_execution("monthly", 19)
    assert not slow.is_legal_execution("daily", 1)

    # Property: a legal execution is always constructible by the policy.
    for domain in RESEARCH_DOMAINS.values():
        for frequency in domain.frequencies:
            for horizon in range(1, 31):
                if domain.is_legal_execution(frequency, horizon):
                    RebalancePolicy(frequency, horizon)


def test_domain_of_feature_resolution():
    """§1.1: feature-to-domain resolution follows the contract table."""
    assert domain_of_feature("RET_1").id == "short_price_volume"
    assert domain_of_feature("LIMIT_UP_CNT_20").id == "short_price_volume"
    assert domain_of_feature("MOMENTUM_20").id == "medium_cross_section"
    assert domain_of_feature("IND_REL_RET_20").id == "medium_cross_section"
    assert domain_of_feature("MARGIN_BALANCE_CHG").id == "medium_cross_section"
    assert domain_of_feature("ROE").id == "slow_fundamental"
    assert domain_of_feature("MARKET_CAP").id == "slow_fundamental"
    with pytest.raises(ValueError, match="NOT_A_FEATURE"):
        domain_of_feature("NOT_A_FEATURE")


def test_restrict_tensor_zeroes_out_of_domain_rows():
    """§4.1: the domain-restricted tensor keeps its shape and in-domain
    rows, zeroes out-of-domain rows, and is the identity for unified."""
    n_features = len(FEATURE_NAMES)
    tensor = np.full((n_features, 2, 3), 7.0, dtype=np.float32)
    short = RESEARCH_DOMAINS["short_price_volume"]

    out = restrict_tensor(tensor, short.id)
    assert out.shape == tensor.shape
    for index, name in enumerate(FEATURE_NAMES):
        if name in short.features:
            assert (out[index] == 7.0).all()
        else:
            assert (out[index] == 0.0).all()

    identity = restrict_tensor(tensor, UNIFIED_DOMAIN_ID)
    assert identity is not tensor
    assert (identity == tensor).all()

    with pytest.raises(ValueError, match="features"):
        restrict_tensor(np.zeros((3, 2, 2), dtype=np.float32), short.id)
    with pytest.raises(ValueError, match="domain"):
        restrict_tensor(tensor, "no_such_domain")


def test_feature_token_ids_follow_global_vocab():
    """§4.2: domain feature token ids are global vocabulary ids; unified
    resolves to None (no restriction)."""
    vocab = FORMULA_VOCAB
    slow = RESEARCH_DOMAINS["slow_fundamental"]
    ids = feature_token_ids(slow.id)
    assert ids is not None
    assert len(ids) == len(slow.features)
    expected = {
        vocab.feature_offset + vocab.feature_names.index(name)
        for name in slow.features
    }
    assert set(ids) == expected
    assert all(
        vocab.feature_offset <= token < vocab.operator_offset for token in ids
    )
    assert feature_token_ids(UNIFIED_DOMAIN_ID) is None


def test_protocol_config_applies_domain_defaults():
    """§3: domain mode resolves frequency/horizon/baseline defaults from
    the registry; explicit values must be legal and inside the domain;
    unified keeps today's behavior exactly."""
    medium = RESEARCH_DOMAINS["medium_cross_section"]
    cfg = make_protocol_config({"protocol": {"domain": medium.id}})
    assert cfg.domain == medium.id
    assert cfg.frequency == medium.default_frequency == "every_10_days"
    assert cfg.horizon == medium.default_horizon == 10
    assert cfg.baseline_signals == list(medium.baseline_signals)

    cfg = make_protocol_config(
        {
            "protocol": {
                "domain": "short_price_volume",
                "frequency": "every_5_days",
                "horizon": 3,
            }
        }
    )
    assert cfg.frequency == "every_5_days"
    assert cfg.horizon == 3

    slow = RESEARCH_DOMAINS["slow_fundamental"]
    cfg = make_protocol_config({"protocol": {"domain": slow.id}})
    assert (cfg.frequency, cfg.horizon) == ("every_20_days", 20)
    assert cfg.baseline_signals == list(slow.baseline_signals)

    with pytest.raises(ValueError, match="short_price_volume"):
        make_protocol_config(
            {
                "protocol": {
                    "domain": "short_price_volume",
                    "frequency": "daily",
                    "horizon": 3,
                }
            }
        )
    with pytest.raises(ValueError, match="ROE"):
        make_protocol_config(
            {
                "protocol": {
                    "domain": "short_price_volume",
                    "baseline_signals": ["ROE"],
                }
            }
        )
    with pytest.raises(ValueError, match="domain"):
        make_protocol_config({"protocol": {"domain": "no_such_domain"}})

    # No domain / explicit unified: pre-P6 behavior unchanged.
    legacy = make_protocol_config({})
    assert legacy.domain == UNIFIED_DOMAIN_ID
    assert legacy.frequency == "daily"
    assert legacy.horizon == 1
    assert legacy.baseline_signals == [
        "REVERSAL_5",
        "RSQ_60",
        "ILLIQ_20",
        "OVERNIGHT_RET",
        "MOMENTUM_20",
        "ROE",
        "TURNOVER",
    ]
    explicit = make_protocol_config({"protocol": {"domain": UNIFIED_DOMAIN_ID}})
    assert explicit.frequency == "daily"
    assert explicit.horizon == 1


def test_backtest_and_reward_defaults_follow_domain():
    """§3.3/§3.4: turnover_budget and cost_weight default to the domain's
    declared values; explicit overrides win; unified keeps defaults."""
    slow = RESEARCH_DOMAINS["slow_fundamental"]
    raw = {"protocol": {"domain": slow.id}}
    assert make_backtest_config(raw).turnover_budget == pytest.approx(
        slow.turnover_budget
    )
    assert make_reward_config(raw).cost_weight == pytest.approx(
        slow.cost_weight
    )

    medium = RESEARCH_DOMAINS["medium_cross_section"]
    assert make_backtest_config(
        {"protocol": {"domain": medium.id}}
    ).turnover_budget == pytest.approx(medium.turnover_budget)

    overridden = make_backtest_config(
        {
            "protocol": {"domain": slow.id},
            "backtest": {"turnover_budget": 0.5},
        }
    )
    assert overridden.turnover_budget == pytest.approx(0.5)

    # Unified: nothing injected (BacktestConfig's own default is None).
    assert make_backtest_config({}).turnover_budget is None
    assert make_reward_config({}).cost_weight == pytest.approx(1.0)


def test_build_action_mask_restricts_feature_ids():
    """§4.2: with feature_ids only those feature tokens open; without them
    the mask matches the pre-P6 behavior (every feature open)."""
    vocab = FORMULA_VOCAB
    stack = torch.zeros(2, dtype=torch.long)
    stack_types = torch.zeros(2, 8, dtype=torch.long)
    done = torch.zeros(2, dtype=torch.bool)
    ids = [
        vocab.feature_offset + vocab.feature_names.index("RET_1"),
        vocab.feature_offset + vocab.feature_names.index("ROE"),
    ]

    mask = build_action_mask(
        stack,
        done,
        0,
        8,
        vocab,
        feature_ids=ids,
        stack_types=stack_types,
    )
    feature_tokens = [
        token
        for token in range(vocab.size)
        if mask[0, token] == 0.0
        and vocab.feature_offset <= token < vocab.operator_offset
    ]
    assert set(feature_tokens) == set(ids)

    mask_all = build_action_mask(
        stack, done, 0, 8, vocab, stack_types=stack_types
    )
    feature_tokens_all = [
        token
        for token in range(vocab.size)
        if mask_all[0, token] == 0.0
        and vocab.feature_offset <= token < vocab.operator_offset
    ]
    # P9 §4.2 (whitelist §10.1 case 2, contract APPROVED): the deprecated
    # features leave the sampling space, so the unrestricted mask opens
    # exactly the live features.
    from ashare_model.vocab import DEPRECATED_FEATURE_NAMES

    live_count = len(vocab.feature_names) - len(DEPRECATED_FEATURE_NAMES)
    assert len(feature_tokens_all) == live_count


def test_random_sampling_stays_inside_domain():
    """§4.2: random sampling with feature_ids never emits an out-of-domain
    feature token."""
    vocab = FORMULA_VOCAB
    slow = RESEARCH_DOMAINS["slow_fundamental"]
    ids = feature_token_ids(slow.id)
    sequences = sample_random_formulas(42, vocab, 12, 300, feature_ids=ids)
    assert sequences
    for sequence in sequences:
        for token in sequence:
            if vocab.feature_offset <= token < vocab.operator_offset:
                assert token in ids, (
                    f"sampled out-of-domain feature token {token} "
                    f"({vocab.token_names[token]})"
                )


def test_window_id_carries_domain():
    """§4.3: the domain-mode window id carries a domain component; the
    default (unified) id stays domain-free."""
    dates = [
        "20240102",
        "20240103",
        "20240105",
        "20240108",
        "20240109",
        "20240112",
        "20240115",
        "20240116",
        "20240117",
        "20240118",
        "20240119",
        "20240122",
        "20240123",
        "20240124",
        "20240125",
        "20240126",
        "20240129",
        "20240130",
        "20240131",
        "20240201",
        "20240202",
        "20240205",
        "20240206",
        "20240207",
        "20240208",
    ]
    contract = FoldTimeContract.resolve(dates, "2024-01-15", "2024-02-08", horizon=1)
    fold = Fold(contract, frequency="daily")
    assert "domain:slow_fundamental" in search_window_id(
        fold, 7, domain_id="slow_fundamental"
    )
    assert "domain:" not in search_window_id(fold, 7)
    assert "domain:" not in search_window_id(fold, 7, domain_id=None)


def test_protocol_artifact_records_research_domain():
    """§4.4: protocol artifacts record the research domain and its
    registry version."""
    from ashare_data.config import ProtocolConfig, TierConfig
    from ashare_model.evaluation import build_result

    proto = ProtocolConfig(
        domain="slow_fundamental", frequency="every_20_days", horizon=20
    )
    result = build_result(
        proto,
        "screening",
        TierConfig(),
        rows=[],
        spec_id=TEST_SPEC_ID,
        run_id=TEST_RUN_ID,
        data_end_date="20240201",
    )
    assert result["research_domain"] == "slow_fundamental"
    assert result["research_domain_version"] == RESEARCH_DOMAIN_VERSION
    assert result["frequency"] == "every_20_days"
    assert result["horizon"] == 20

    unified = build_result(
        ProtocolConfig(),
        "screening",
        TierConfig(),
        rows=[],
        spec_id=TEST_SPEC_ID,
        run_id=TEST_RUN_ID,
        data_end_date="20240201",
    )
    assert unified["research_domain"] == UNIFIED_DOMAIN_ID


def test_run_protocol_domain_restricts_tensor_and_rows(
    populated_db, monkeypatch
):
    """§4.1/§4.4: a domain run zeroes the out-of-domain tensor rows, uses
    the domain's baseline ladder and records the domain in the artifact."""
    import ashare_model.evaluation as evaluation
    from ashare_data.config import BacktestConfig, FoldConfig, ModelConfig, ProtocolConfig
    from tests.test_evaluation import _FakeTrainer, _loader

    loader = _loader(populated_db)
    original = loader.factor_tensor.numpy().copy()
    short = RESEARCH_DOMAINS["short_price_volume"]
    proto = ProtocolConfig(
        domain="short_price_volume",
        baseline_signals=list(short.baseline_signals),
        folds=[FoldConfig("2024-01-10", "2024-01-25")],
        seeds=[42],
        random_samples=0,
        random_match_budget=False,
        gp_enabled=False,
        tpe_enabled=False,
    )
    monkeypatch.setattr(
        evaluation, "_build_trainer", lambda *a, **k: _FakeTrainer([1])
    )
    result = run_protocol(
        loader,
        populated_db,
        ModelConfig(),
        BacktestConfig(),
        None,
        proto,
        "screening",
        spec_id=TEST_SPEC_ID,
        run_id=TEST_RUN_ID,
    )
    assert result["research_domain"] == "short_price_volume"
    rows = result["rows"]
    # 1 benchmark + 5 short-domain baselines + 1 trained seed.
    assert len(rows) == 7
    baselines = [r for r in rows if r["candidate"].startswith("baseline:")]
    assert {r["formula_text"] for r in baselines} == set(
        RESEARCH_DOMAINS["short_price_volume"].baseline_signals
    )
    # The loader tensor is restricted: out-of-domain rows are zero and
    # in-domain rows keep their exact pre-restriction values.
    tensor = loader.factor_tensor.numpy()
    short = RESEARCH_DOMAINS["short_price_volume"]
    for index, name in enumerate(FEATURE_NAMES):
        if name in short.features:
            np.testing.assert_array_equal(tensor[index], original[index])
        else:
            assert not tensor[index].any()


def test_run_protocol_rejects_illegal_domain_execution(
    populated_db, monkeypatch
):
    """§3: a directly-constructed protocol with an illegal (domain,
    frequency, horizon) triple fails fast at the protocol entry."""
    import ashare_model.evaluation as evaluation
    from ashare_data.config import BacktestConfig, ModelConfig, ProtocolConfig
    from tests.test_evaluation import _FakeTrainer, _loader

    loader = _loader(populated_db)
    proto = ProtocolConfig(
        domain="short_price_volume", frequency="daily", horizon=3
    )
    monkeypatch.setattr(
        evaluation, "_build_trainer", lambda *a, **k: _FakeTrainer([1])
    )
    with pytest.raises(ValueError, match="short_price_volume"):
        run_protocol(
            loader,
            populated_db,
            ModelConfig(),
            BacktestConfig(),
            None,
            proto,
            "screening",
            spec_id=TEST_SPEC_ID,
            run_id=TEST_RUN_ID,
        )


def test_p9_research_domain_version_and_single_domain_assignments():
    """P9 §5: the v2 domain registry assigns every new feature to exactly
    one research domain (LIQ/volume + limit-event -> short; residualized
    momentum, PV divergence, crowding -> medium).  v3 (P13 §5.5,
    whitelist §10.1 case 2, docs/p13_fundamental_fields_contract.md
    APPROVED): the four family-⑤ features join slow_fundamental."""
    assert RESEARCH_DOMAIN_VERSION == 3
    from ashare_model.research_domain import domain_of_feature

    expectations = {
        "IND_REL_RET_60": "medium_cross_section",
        "IND_REL_RET_120": "medium_cross_section",
        "PV_DIV_20": "medium_cross_section",
        "CROWD_TURNOVER_60": "medium_cross_section",
        "CROWD_AMOUNT_60": "medium_cross_section",
        "MARGIN_CROWD_60": "medium_cross_section",
        "LIQ_SHOCK_20": "short_price_volume",
        "VOLUME_SHRINK_5_20": "short_price_volume",
        "LIMIT_UP_CNT_5": "short_price_volume",
        "LIMIT_DOWN_STREAK": "short_price_volume",
        "LIMIT_BREAK_5": "short_price_volume",
    }
    for name, domain_id in expectations.items():
        assert domain_of_feature(name).id == domain_id, name
