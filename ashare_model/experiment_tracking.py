"""Optional MLflow tracking bridge (T1-01).

The experiment archive (``scripts/archive_run.py``, domain JSON under
``experiments/``) remains the primary, offline-first experiment record.
MLflow is an **additive, opt-in** channel: when ``mlflow`` is installed
and a tracking URI is configured (``MLFLOW_TRACKING_URI`` or the
``tracking_uri`` argument), a run is registered with its parameters,
metrics, tags and artifact paths; otherwise the call returns a structured
:class:`TrackingOutcome` instead of raising, so an offline or
MLflow-less environment never breaks an archive.

The explicit rejection policy: tracking is disabled (``disabled``) when
``ALPHAGPT_DISABLE_TRACKING`` is set, refused with ``mlflow_not_installed``
when the package is absent, refused with ``tracking_uri_unset`` when no
URI is configured, and reported as ``tracking_failed`` (with the exception
detail) when the MLflow call itself errors.  Every outcome is a value, not
an exception, so callers log it and continue.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class TrackingOutcome:
    """Structured result of one tracking attempt (never raises)."""

    tracked: bool
    run_id: str | None
    reason: str
    detail: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def _disabled() -> bool:
    return os.environ.get("ALPHAGPT_DISABLE_TRACKING", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _load_mlflow():
    try:
        return importlib.import_module("mlflow")
    except ImportError:
        return None


def log_run(
    *,
    experiment_name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
    artifacts: Iterable[str] | None = None,
    tracking_uri: str | None = None,
) -> TrackingOutcome:
    """Register one run with MLflow, or return a structured refusal.

    ``tracking_uri`` overrides ``MLFLOW_TRACKING_URI``.  ``artifacts`` are
    local file paths logged via ``mlflow.log_artifact``.  Missing mlflow,
    missing URI and MLflow errors are all structured outcomes — this
    function never raises.
    """

    params = dict(params or {})
    metrics = {key: float(value) for key, value in (metrics or {}).items()}
    if _disabled():
        return TrackingOutcome(False, None, "disabled", params=params, metrics=metrics)

    mlflow = _load_mlflow()
    if mlflow is None:
        return TrackingOutcome(
            False, None, "mlflow_not_installed", params=params, metrics=metrics
        )

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return TrackingOutcome(
            False, None, "tracking_uri_unset", params=params, metrics=metrics
        )

    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        for key, value in params.items():
            mlflow.log_params({key: value})
        if metrics:
            mlflow.log_metrics(metrics)
        for key, value in (tags or {}).items():
            mlflow.set_tag(key, value)
        for path in artifacts or ():
            mlflow.log_artifact(str(path))
        run = mlflow.active_run()
        run_id = str(run.info.run_id) if run is not None else None
        mlflow.end_run()
        return TrackingOutcome(
            True, run_id, "ok", params=params, metrics=metrics
        )
    except Exception as exc:  # noqa: BLE001 - structured, never fatal
        return TrackingOutcome(
            False,
            None,
            "tracking_failed",
            detail=f"{type(exc).__name__}: {exc}",
            params=params,
            metrics=metrics,
        )
