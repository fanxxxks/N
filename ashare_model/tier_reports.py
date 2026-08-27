"""Tier-grouped diagnostics and ablation reports (P2-05).

The free-data credibility tiers (A / B / C, see :mod:`ashare_model.data_tier`)
are reported in three tier sets — **A**, **A+B**, **all** — so research
can see how the factor stack behaves when Tier C (industry snapshot /
placeholder) or Tier B (fundamentals / margin) is dropped:

* :func:`run_tier_diagnostics` — the full diagnostics chain (coverage /
  rank-IC / ICIR / correlation matrix) restricted to one tier set, via
  :func:`ashare_model.diagnostics.factor_report`;
* :func:`run_tier_ablation` — trains the formula generator on each tier
  set with the same seed/steps/batch (all = baseline, AB = ablate C,
  A = ablate B+C) and records each best formula together with its
  data-tier trace, so every report formula is traceable back to the data
  it depends on;
* :func:`assemble_tier_report` — the versioned ``data/tier_report.json``
  payload (``TIER_REPORT_VERSION``).

The training glue mirrors ``scripts/ablate_families.py`` exactly (same
trainer path, same seed semantics), so the two ablation families stay
comparable; no new dependencies, all sources free.
"""

from __future__ import annotations

from .data_tier import (
    DATA_TIER_VERSION,
    DataTier,
    formula_data_tier_report,
    tier_features,
)

# Bump when the report schema, the tier sets or the ablation semantics
# change.
TIER_REPORT_VERSION = 1

# The three reported tier sets (research ladder: A, A+B, all).
TIER_SETS: tuple[tuple[str, ...], ...] = (
    ("A",),
    ("A", "B"),
    ("A", "B", "C"),
)


def tier_set_features(tier_set: tuple[str, ...]) -> list[str]:
    """Vocabulary features belonging to one tier set, in vocab order."""
    return tier_features(tuple(DataTier(t) for t in tier_set))


def build_tier_ablation_plan() -> list[dict]:
    """The three ablation runs: all (baseline), AB (ablate C), A (ablate
    B+C).  Each entry names its included tiers and the excluded features;
    exclusion is the complement of the included set."""

    plan = []
    all_features = set(tier_features((DataTier.A, DataTier.B, DataTier.C)))
    for label, tier_set in (("all", ("A", "B", "C")), ("AB", ("A", "B")),
                            ("A", ("A",))):
        included = set(tier_set_features(tier_set))
        plan.append(
            {
                "label": label,
                "included_tiers": list(tier_set),
                # Exclusion is the complement of the included set: dropping
                # every feature outside the tier set is exactly "train on
                # this tier set only".
                "excluded_features": sorted(all_features - included),
            }
        )
    return plan


def run_tier_diagnostics(loader, train_end_date: str, tier_set: tuple[str, ...]):
    """Diagnostics for one tier set (coverage / rank-IC / correlations)."""

    from .diagnostics import factor_report

    return factor_report(loader, train_end_date, tiers=tuple(tier_set))


def _train_one(loader, tensor, data_config, model_config, backtest_config,
               steps: int, batch_size: int, seed: int):
    """One formula-generator training run on the given factor tensor.

    Mirrors ``scripts/ablate_families.py._train_one``: same trainer path,
    artifacts disabled.  Returns ``(best_reward, best_formula_text,
    best_tokens)``.
    """

    import torch

    from .train import AshareTrainer

    loader.factor_tensor = torch.tensor(tensor, dtype=torch.float32)
    trainer = AshareTrainer(
        data_config, model_config, backtest_config, loader=loader
    )
    trainer.train(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        save_artifacts=False,
    )
    return float(trainer.best_reward), trainer.best_formula, trainer.best_tokens


def run_tier_ablation(
    loader,
    data_config,
    model_config,
    backtest_config,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict:
    """Train the generator on every tier set (same seed/steps/batch).

    Returns one run per plan entry with the best reward / formula and the
    formula's data-tier trace; the delta is measured against the all-tier
    baseline so the report is self-contained.
    """

    from .factors import ablate_factors

    full = loader.factor_tensor.numpy()
    runs: dict[str, dict] = {}
    baseline_reward: float | None = None
    for entry in build_tier_ablation_plan():
        excluded = entry["excluded_features"]
        tensor = ablate_factors(full, excluded) if excluded else full
        reward, formula_text, tokens = _train_one(
            loader, tensor, data_config, model_config, backtest_config,
            steps, batch_size, seed,
        )
        if baseline_reward is None:
            baseline_reward = reward
        runs[entry["label"]] = {
            "included_tiers": entry["included_tiers"],
            "excluded_features": excluded,
            "best_reward": reward,
            "best_formula": formula_text,
            "formula_data_tier": formula_data_tier_report(tokens=tokens),
            "delta_vs_baseline": round(reward - baseline_reward, 4),
        }
    return runs


def assemble_tier_report(
    *,
    diagnostics: dict,
    ablation: dict,
    dataset_id: str | None,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict:
    """The versioned tier-report payload (``data/tier_report.json``)."""

    return {
        "tier_report_version": TIER_REPORT_VERSION,
        "data_tier_version": DATA_TIER_VERSION,
        "dataset_id": dataset_id,
        "tier_sets": [list(s) for s in TIER_SETS],
        "diagnostics": diagnostics,
        "ablation": {
            "budget": {"steps": steps, "batch_size": batch_size, "seed": seed},
            "runs": ablation,
        },
    }
