"""The method-revision conditional-trigger protocol guards (WP17 W6).

Frozen spec: ``reports/wp17/FROZEN_17.md`` §6.  This module is
DOCUMENT-PLUS-GUARD only: it runs no calibration and compares no method
versions.  What it enforces, fail-closed and with an EMPTY bypass
surface (§6.5 — no allow-list, no opt-out flag, no variable lookup):

* a method-version comparison or acceptance calibration cannot start
  without an OPEN trigger record (G1 — the default refusal);
* at most TWO method versions per episode (G2);
* exactly ONE independent-seed acceptance per episode — a failed or
  insufficient acceptance keeps 'no reliable inference', no re-runs and
  no additional seeds (G3);
* the wall clock cannot exceed the frozen budget (G4);
* the acceptance seed block must be fresh and disjoint from the p40
  tuning blocks (10000-50000), the p48 validation block (900000+) and
  every prior episode block, with a floor at 1,000,000 (G5, p48 B3).

The registry (``method_revision_registry.json`` beside this module)
lives in-package on purpose: it evolves with the guard code and never
touches the read-only ``data/`` tree.  Episodes are append-only; the
identity/plan fields of an episode (episode_id, opened_utc, artifact,
blocking_reason_codes, target_operating_points, frozen_budget_seconds,
pilot_seeds) are written once at open and have no update path;
``versions`` grows append-only up to two; ``acceptance`` is written
once; ``status`` moves monotonically open -> terminal.  The guard
re-reads and schema-validates the registry on every call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

#: The in-package registry (FROZEN §6 preamble).
DEFAULT_REGISTRY_PATH = Path(__file__).with_name(
    "method_revision_registry.json"
)

# --- the frozen refusal strings (FROZEN §6.5, verbatim) -------------------------

G1_REFUSED = (
    "method revision refused: no open trigger record exists for this "
    "work - register the candidate worth formal testing (artifact "
    "identity + blocking reason codes) before any method-version "
    "comparison or acceptance calibration starts (wp17 W6)"
)

G2_REFUSED_TEMPLATE = (
    "method revision refused: episode {episode_id} already lists {n} "
    "method versions; at most two may be compared per episode "
    "(wp17 W6)"
)

G3_REFUSED_TEMPLATE = (
    "method revision refused: episode {episode_id} has already run its "
    "single independent-seed acceptance calibration; a failed or "
    "insufficient acceptance keeps 'no reliable inference' - no re-runs "
    "and no additional seeds (wp17 W6)"
)

G4_REFUSED_TEMPLATE = (
    "method revision refused: episode {episode_id} exceeded its frozen "
    "budget of {budget_seconds}s; the honest state remains 'no reliable "
    "inference' (wp17 W6)"
)

G5_REFUSED_TEMPLATE = (
    "method revision refused: the requested seed block [{start},{span}) "
    "overlaps a used block (tuning 10000-50000, validation 900000+, or "
    "a prior episode); acceptance requires a fresh disjoint block "
    "(wp17 W6, p48 B3)"
)

#: The version attested forward by the evidence registration (FROZEN
#: §1.3): a method constant equal to this needs no episode; any OTHER
#: value requires a terminal ``accepted`` episode behind it (§6.5 item 2).
ATTESTED_METHOD_VERSION = 1

MAX_VERSIONS_PER_EPISODE = 2
ACCEPTANCE_SEED_FLOOR = 1_000_000

#: [start, end) ranges the acceptance block must not touch (FROZEN §6.4:
#: the p40 tuning seeds 10000-50000 participated in method adjustment;
#: the p48 validation block is 900000 + 100000).
RESERVED_BLOCKS: tuple[tuple[int, int], ...] = (
    (10_000, 50_001),
    (900_000, 1_000_000),
)

#: The two registered episode purposes (§6.5 closing note): the same
#: discipline in two registration classes — scenario_extension is not
#: bound by the two-version cap but still demands a trigger record, a
#: frozen budget and a fresh seed block.
PURPOSES = ("method_revision", "scenario_extension")

STATUSES = ("open", "accepted", "rejected", "budget_exhausted")

#: The wp17 W2 qualification reason codes (FROZEN §1.2) — the only
#: legal ``blocking_reason_codes`` entries (trigger condition (c): the
#: barrier between the candidate and promotion must be statistical
#: qualification, not data/identity/content/numeric failure).
KNOWN_REASON_CODES = frozenset(
    {
        "EVIDENCE_NOT_REGISTERED",
        "EVIDENCE_HASH_MISMATCH",
        "EVIDENCE_IDENTITY_MISMATCH",
        "N_DATES_OUT_OF_CALIBRATED_DOMAIN",
        "K_OUT_OF_CALIBRATED_DOMAIN",
        "MISSING_FOLD_POLICY_UNREGISTERED",
        "SCENARIO_UNCALIBRATED",
        "SCENARIO_INSUFFICIENT_SAMPLES",
        "SCENARIO_UNRELIABLE_AT_NOMINAL",
    }
)

_EPISODE_REQUIRED_FIELDS = (
    "episode_id", "purpose", "opened_utc", "artifact",
    "blocking_reason_codes", "target_operating_points", "versions",
    "frozen_budget_seconds", "pilot_seeds", "acceptance", "status",
)

_POINT_INT_FIELDS = ("n_dates", "k_trials", "n_perms", "block_size")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_episode(episode: dict) -> None:
    """Schema validation (FROZEN §6.5 item 3), re-run on every read."""

    if not isinstance(episode, dict):
        raise ValueError("episode must be an object (wp17 W6)")
    missing = [f for f in _EPISODE_REQUIRED_FIELDS if f not in episode]
    if missing:
        raise ValueError(
            f"episode lacks required field(s): {missing} (wp17 W6 §6.1)"
        )
    if not isinstance(episode["episode_id"], str) or not episode["episode_id"]:
        raise ValueError("episode_id must be a non-empty string (wp17 W6)")
    if episode["purpose"] not in PURPOSES:
        raise ValueError(
            f"episode purpose {episode['purpose']!r} not in {PURPOSES} "
            "(wp17 W6)"
        )
    if episode["status"] not in STATUSES:
        raise ValueError(
            f"episode status {episode['status']!r} not in {STATUSES} "
            "(wp17 W6)"
        )
    artifact = episode["artifact"]
    if not isinstance(artifact, dict) or any(
        not artifact.get(field)
        for field in ("run_id", "spec_id", "candidate", "artifact_sha256")
    ) or "seed" not in artifact:
        raise ValueError(
            "episode artifact identity must carry run_id/spec_id/"
            "candidate/seed/artifact_sha256 (wp17 W6 §6.1 (a))"
        )
    codes = episode["blocking_reason_codes"]
    if (
        not isinstance(codes, list)
        or not codes
        or not all(code in KNOWN_REASON_CODES for code in codes)
    ):
        raise ValueError(
            "blocking_reason_codes must be a non-empty list of wp17 W2 "
            "qualification reason codes (wp17 W6 §6.1 (c))"
        )
    points = episode["target_operating_points"]
    if not isinstance(points, list) or not points:
        raise ValueError(
            "target_operating_points must be a non-empty list (wp17 W6 "
            "§6.1 (d))"
        )
    for point in points:
        if not isinstance(point, dict) or not all(
            _is_int(point.get(field)) for field in _POINT_INT_FIELDS
        ):
            raise ValueError(
                "each target operating point needs integer n_dates/"
                "k_trials/n_perms/block_size (wp17 W6 §6.1 (d))"
            )
        forms = point.get("correlation_forms")
        if not isinstance(forms, list) or not forms:
            raise ValueError(
                "each target operating point needs a non-empty "
                "correlation_forms list — n=250, K=8 evidence is not "
                "extrapolable to other correlation shapes (wp17 W6 §6.1 (d))"
            )
    versions = episode["versions"]
    if not isinstance(versions, list) or not all(
        isinstance(v, str) and v for v in versions
    ):
        raise ValueError("versions must be a list of strings (wp17 W6)")
    if len(versions) > MAX_VERSIONS_PER_EPISODE:
        raise ValueError(
            G2_REFUSED_TEMPLATE.format(
                episode_id=episode["episode_id"], n=len(versions)
            )
        )
    budget = episode["frozen_budget_seconds"]
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) \
            or budget <= 0:
        raise ValueError(
            "frozen_budget_seconds must be a positive number (wp17 W6 §6.3)"
        )
    pilots = episode["pilot_seeds"]
    if not isinstance(pilots, list) or not all(
        _is_int(seed) for seed in pilots
    ):
        raise ValueError(
            "pilot_seeds must be a list of integers, declared as "
            "tuning-participating and never reused for acceptance "
            "(wp17 W6 §6.1)"
        )
    acceptance = episode["acceptance"]
    if acceptance is not None:
        if not isinstance(acceptance, dict) or not all(
            field in acceptance
            for field in (
                "version_id", "seed_block_start", "span", "reps",
                "verdict", "closed_utc",
            )
        ):
            raise ValueError(
                "acceptance must carry version_id/seed_block_start/span/"
                "reps/verdict/closed_utc (wp17 W6 §6.1)"
            )


def _blocks_overlap(a_start: int, a_span: int, b_start: int, b_span: int):
    return a_start < b_start + b_span and b_start < a_start + a_span


def _assert_seed_block_fresh(
    start: int, span: int, episodes: list[dict], own_id: str
) -> None:
    if not _is_int(start) or not _is_int(span) or start < 0 or span <= 0:
        raise ValueError(
            "seed block must be non-negative integers with span >= 1 "
            "(wp17 W6)"
        )
    if start < ACCEPTANCE_SEED_FLOOR:
        raise ValueError(G5_REFUSED_TEMPLATE.format(start=start, span=span))
    for reserved_start, reserved_end in RESERVED_BLOCKS:
        if start < reserved_end and reserved_start < start + span:
            raise ValueError(
                G5_REFUSED_TEMPLATE.format(start=start, span=span)
            )
    for episode in episodes:
        acceptance = episode.get("acceptance")
        if not acceptance or episode.get("episode_id") == own_id:
            continue
        if _blocks_overlap(
            start, span, acceptance["seed_block_start"], acceptance["span"]
        ):
            raise ValueError(
                G5_REFUSED_TEMPLATE.format(start=start, span=span)
            )


def _verify_mutual_disjointness(episodes: list[dict]) -> None:
    """Re-verify seed-block disjointness on every read (§6.5 item 3)."""

    written: list[tuple[str, int, int]] = []
    for episode in episodes:
        acceptance = episode.get("acceptance")
        if not acceptance:
            continue
        start = acceptance["seed_block_start"]
        span = acceptance["span"]
        for other_id, other_start, other_span in written:
            if _blocks_overlap(start, span, other_start, other_span):
                raise ValueError(
                    G5_REFUSED_TEMPLATE.format(start=start, span=span)
                )
        written.append((episode["episode_id"], start, span))
        for reserved_start, reserved_end in RESERVED_BLOCKS:
            if start < reserved_end and reserved_start < start + span:
                raise ValueError(
                    G5_REFUSED_TEMPLATE.format(start=start, span=span)
                )


def _load(path=None) -> list[dict]:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not registry_path.exists():
        return []
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    episodes = (payload or {}).get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(
            "method revision registry must carry an 'episodes' list "
            "(wp17 W6)"
        )
    seen: set[str] = set()
    for episode in episodes:
        _validate_episode(episode)
        if episode["episode_id"] in seen:
            raise ValueError(
                f"duplicate episode_id {episode['episode_id']!r}: the "
                "registry is append-only (wp17 W6)"
            )
        seen.add(episode["episode_id"])
    _verify_mutual_disjointness(episodes)
    return episodes


def _save(path, episodes: list[dict]) -> None:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    registry_path.write_text(
        json.dumps({"episodes": episodes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --- the guards -----------------------------------------------------------------


def require_open_episode(episode_token, *, path=None) -> dict:
    """Guard挂点 1 (FROZEN §6.5): ``calibrate_scenario`` and every other
    calibration entry point must pass their ``episode_token`` through
    here.  ``None`` (the default) refuses — fail-closed."""

    episodes = _load(path)
    if not isinstance(episode_token, str) or not episode_token:
        raise ValueError(G1_REFUSED)
    for episode in episodes:
        if episode["episode_id"] == episode_token:
            if episode["status"] == "open":
                return episode
            break
    raise ValueError(G1_REFUSED)


def open_episode(record: dict, *, path=None) -> dict:
    """Register one trigger record (FROZEN §6.1): the four trigger
    conditions are validated field-by-field — (a) the artifact identity
    five-tuple, (c) the blocking codes are W2 qualification codes, (d)
    the target operating points cover the actual correlation shapes.
    Episodes are append-only; a duplicate id is refused."""

    episode = dict(record)
    episode.setdefault(
        "opened_utc",
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    episode.setdefault("status", "open")
    episode.setdefault("versions", [])
    episode.setdefault("acceptance", None)
    _validate_episode(episode)
    episodes = _load(path)
    if any(e["episode_id"] == episode["episode_id"] for e in episodes):
        raise ValueError(
            f"episode {episode['episode_id']!r} already exists: the "
            "registry is append-only (wp17 W6)"
        )
    episodes.append(episode)
    _save(path, episodes)
    return episode


def register_method_version(
    episode_id: str, version_id: str, *, path=None
) -> list[str]:
    """Append one method version to an OPEN episode, refusing the third
    (G2).  Development compares at most two versions per episode."""

    episodes = _load(path)
    for episode in episodes:
        if episode["episode_id"] == episode_id:
            if episode["status"] != "open":
                raise ValueError(G1_REFUSED)
            if version_id in episode["versions"]:
                raise ValueError(
                    f"method version {version_id!r} is already listed on "
                    f"episode {episode_id!r} (wp17 W6)"
                )
            if len(episode["versions"]) + 1 > MAX_VERSIONS_PER_EPISODE:
                raise ValueError(
                    G2_REFUSED_TEMPLATE.format(
                        episode_id=episode_id, n=len(episode["versions"])
                    )
                )
            episode["versions"] = [*episode["versions"], version_id]
            _validate_episode(episode)
            _save(path, episodes)
            return list(episode["versions"])
    raise ValueError(G1_REFUSED)


def check_budget(episode_id: str, elapsed_seconds: float, *, path=None):
    """Guard挂点 for the frozen wall-clock budget (FROZEN §6.3): at or
    under the budget passes; over it refuses (G4) — the honest state
    remains 'no reliable inference'."""

    episodes = _load(path)
    for episode in episodes:
        if episode["episode_id"] == episode_id:
            if elapsed_seconds > episode["frozen_budget_seconds"]:
                raise ValueError(
                    G4_REFUSED_TEMPLATE.format(
                        episode_id=episode_id,
                        budget_seconds=episode["frozen_budget_seconds"],
                    )
                )
            return None
    raise ValueError(G1_REFUSED)


def record_acceptance(
    episode_id: str,
    *,
    version_id: str,
    seed_block_start: int,
    span: int,
    reps: int,
    verdict: str,
    path=None,
) -> dict:
    """Record the ONE independent-seed acceptance (FROZEN §6.4): a
    second request refuses (G3); the seed block must be fresh and
    disjoint (G5); the episode then moves to its terminal status."""

    episodes = _load(path)
    for episode in episodes:
        if episode["episode_id"] == episode_id:
            if episode.get("acceptance") is not None:
                raise ValueError(
                    G3_REFUSED_TEMPLATE.format(episode_id=episode_id)
                )
            if episode["status"] != "open":
                raise ValueError(G1_REFUSED)
            if version_id not in episode["versions"]:
                raise ValueError(
                    f"method version {version_id!r} is not listed on "
                    f"episode {episode_id!r} (wp17 W6)"
                )
            _assert_seed_block_fresh(
                seed_block_start, span, episodes, episode_id
            )
            acceptance = {
                "version_id": version_id,
                "seed_block_start": int(seed_block_start),
                "span": int(span),
                "reps": int(reps),
                "verdict": verdict,
                "closed_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
            episode["acceptance"] = acceptance
            episode["status"] = (
                "accepted"
                if verdict in ("calibrated", "conservative")
                else "rejected"
            )
            _validate_episode(episode)
            _save(path, episodes)
            return acceptance
    raise ValueError(G1_REFUSED)


def assert_method_version_endorsed(version: int, *, path=None):
    """Guard挂点 2 (FROZEN §6.5): a method constant equal to the
    attested version needs no episode; any OTHER value requires a
    terminal ``accepted`` episode behind it — a method-constant
    turnover without an endorsement refuses."""

    if version == ATTESTED_METHOD_VERSION:
        return None
    episodes = _load(path)
    if not any(episode["status"] == "accepted" for episode in episodes):
        raise ValueError(
            f"method revision refused: max_t_method_version={version} has "
            "no accepted episode endorsing it - a method-constant "
            "turnover requires a terminal accepted episode (wp17 W6)"
        )
    return None
