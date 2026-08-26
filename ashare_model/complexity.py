"""AST complexity billing (T1-03).

Complexity is billed from the formula AST — the single source of truth for
formula semantics — along four auditable axes:

* ``node_count`` — every operator/feature node;
* ``max_depth`` — the deepest nesting;
* ``longest_window`` — the longest trailing window any operator looks back
  (the memory footprint of a signal);
* ``operation_cost`` — the summed per-operator cost (arithmetic = 1,
  windowed = 2, cross-sectional/event = 3), reflecting compute and data
  requirements.

``complexity_bill`` combines the four axes into one monotone bill
(``>= 1`` for any formula, exactly ``1`` for a bare feature).  The
candidate scorer charges ``complexity_penalty * bill`` against the reward
and rejects formulas whose bill exceeds ``max_complexity``, so the policy
cannot game rewards with arbitrarily large or long-windowed formulas.

The window and cost tables are explicit and versioned (never parsed from
operator names), so a vocabulary change cannot silently rebill formulas.
"""

from __future__ import annotations

from typing import Union

from .ir import Binary, Feature, FormulaIR, Ternary, Unary

# Longest look-back (in sessions) of each operator that looks back at all.
WINDOW_BY_OPERATOR: dict[str, int] = {
    "JUMP": 60,
    "DECAY": 2,
    "DELAY1": 1,
    "MAX3": 2,
    "DELTA5": 5,
    "MA20": 20,
    "STD20": 20,
    "TS_RANK20": 20,
    "CORR20": 20,
    "DOWNVOL20": 20,
    "MA5": 5,
    "MA10": 10,
    "MA60": 60,
    "STD5": 5,
    "STD10": 10,
    "STD60": 60,
    "TS_RANK5": 5,
    "TS_RANK10": 10,
    "TS_RANK60": 60,
    "CORR5": 5,
    "CORR10": 10,
    "CORR60": 60,
    "DOWNVOL5": 5,
    "DOWNVOL10": 10,
    "DOWNVOL60": 60,
    "DELTA10": 10,
    "DELTA20": 20,
}

# Per-operator compute/data cost units: arithmetic = 1, windowed = 2
# (per-date windows), cross-sectional / event / gate = 3.
OPERATION_COST: dict[str, int] = {
    "ADD": 1,
    "SUB": 1,
    "MUL": 1,
    "DIV": 1,
    "NEG": 1,
    "ABS": 1,
    "SIGN": 1,
    "GATE": 3,
    "JUMP": 3,
    "DECAY": 2,
    "DELAY1": 1,
    "MAX3": 2,
    "CS_RANK": 3,
    "CS_ZSCORE": 3,
    "CS_DEMEAN": 3,
    "CS_NEUTRALIZE": 3,
}
for _name in WINDOW_BY_OPERATOR:
    OPERATION_COST.setdefault(_name, 2)

# Coefficients of the monotone bill (documented, tested):
#   bill = 1 + 0.25*(nodes-1) + 0.10*depth + 0.05*(window/60) + 0.10*(opcost-1)
# A typical 3-node windowed formula bills ~1 unit; a 12-node monster with
# MA60 bills ~5 units.  Bare features bill exactly 1.0.
_NODE_WEIGHT = 0.25
_DEPTH_WEIGHT = 0.10
_WINDOW_WEIGHT = 0.05
_OPCOST_WEIGHT = 0.10


def _children(ir: FormulaIR) -> tuple[FormulaIR, ...]:
    if isinstance(ir, Unary):
        return (ir.operand,)
    if isinstance(ir, Binary):
        return (ir.left, ir.right)
    if isinstance(ir, Ternary):
        return (ir.a, ir.b, ir.c)
    return ()


def node_count(ir: FormulaIR) -> int:
    """Number of nodes (features and operators) in the AST."""

    return 1 + sum(node_count(child) for child in _children(ir))


def max_depth(ir: FormulaIR) -> int:
    """Longest root-to-leaf path length."""

    children = _children(ir)
    if not children:
        return 1
    return 1 + max(max_depth(child) for child in children)


def longest_window(ir: FormulaIR) -> int:
    """Longest look-back window of any operator in the AST (0 = none)."""

    if isinstance(ir, Feature):
        return 0
    own = WINDOW_BY_OPERATOR.get(ir.op, 0)
    return max([own] + [longest_window(child) for child in _children(ir)])


def operation_cost(ir: FormulaIR) -> int:
    """Summed per-operator cost units (features cost nothing)."""

    if isinstance(ir, Feature):
        return 0
    own = OPERATION_COST.get(ir.op, 1)
    return own + sum(operation_cost(child) for child in _children(ir))


def complexity_bill(ir: FormulaIR) -> float:
    """Monotone complexity bill combining nodes, depth, window and cost.

    ``>= 1.0`` for every formula; exactly ``1.0`` for a bare feature.
    """

    nodes = node_count(ir)
    depth = max_depth(ir)
    window = longest_window(ir)
    opcost = operation_cost(ir)
    bill = (
        1.0
        + _NODE_WEIGHT * (nodes - 1)
        + _DEPTH_WEIGHT * depth
        + _WINDOW_WEIGHT * (window / 60.0)
        + _OPCOST_WEIGHT * (opcost - 1)
    )
    return max(1.0, float(bill))
