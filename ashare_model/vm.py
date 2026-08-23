"""StackVM interpreter for A-share factor formulas.

Output contract: every executed formula returns its signal **cross-sectionally
z-scored per date** (:func:`ashare_model.ops.cross_sectional_zscore`), applied
as the final step of :meth:`StackVM.execute`.  Stacked arithmetic (MUL/DIV/ADD
chains) can drift the raw scale by orders of magnitude; the terminal
standardization keeps the scale meaningful for threshold semantics (GATE/JUMP
compare against 0) and gives every formula a common scale for future signal
combination.  It is a monotone per-date transformation, so it changes neither
the rank IC nor the top-n selection of a signal.

The VM also carries execution-context tensors (``[stock, date]``):

* ``industry_codes`` — optional discrete group ids consumed by the
  ``CS_NEUTRALIZE`` operator; without it the operator degrades to the
  full-market demean.
* ``universe_mask`` — the mandatory bool PIT eligibility mask consumed by
  every cross-sectional operator (``CS_RANK``/``CS_ZSCORE``/``CS_DEMEAN``/
  ``CS_NEUTRALIZE``) and by the terminal z-score.  The reference set of each
  date column is the eligible & finite cells only, so stocks outside the
  universe can never move the statistics of eligible stocks; the final
  signal keeps NaN (non-participating) at ineligible cells, so they cannot
  enter a downstream sort as zeros.  A date with no eligible cell collapses
  to the stable neutral 0.

Both tensors are (re)assigned by the trainer/evaluator when the factor
tensor moves to the compute device, exactly like any other VM data.
"""

from __future__ import annotations

from typing import Iterable

import torch

from .ops import (
    OPS_CONFIG,
    _cs_demean,
    _cs_neutralize,
    _cs_rank,
    cross_sectional_zscore,
)
from .vocab import FORMULA_VOCAB


class StackVM:
    def __init__(
        self,
        vocab=None,
        industry_codes: torch.Tensor | None = None,
        *,
        universe_mask: torch.Tensor,
    ):
        self.vocab = vocab or FORMULA_VOCAB
        self.feature_offset = 1
        self.operator_offset = self.vocab.operator_offset
        self.op_map = {
            self.operator_offset + i: cfg[1] for i, cfg in enumerate(OPS_CONFIG)
        }
        self.arity_map = {
            self.operator_offset + i: cfg[2] for i, cfg in enumerate(OPS_CONFIG)
        }
        # Execution-context data for the CS_NEUTRALIZE operator: discrete
        # industry group ids aligned with the [stock, date] factor columns.
        # The trainer (re)assigns this attribute when it moves the factor
        # tensor to the compute device.
        self.industry_codes = industry_codes
        # Execution-context data for the cross-sectional operators: the
        # [stock, date] bool PIT eligibility mask.  Reference statistics
        # (and the terminal z-score) use only eligible & finite cells; the
        # trainer (re)assigns this attribute when it moves the factor
        # tensor to the compute device.
        self.universe_mask = universe_mask
        self._cs_token_names = {
            self.operator_offset + i: name
            for i, (name, _, _) in enumerate(OPS_CONFIG)
            if name.startswith("CS_")
        }

    def execute(
        self,
        formula_tokens: Iterable[int],
        factor_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        """Execute a postfix formula over ``[Feature, Stock, Time]`` input.

        The resulting ``[Stock, Time]`` signal is cross-sectionally z-scored
        per date before it is returned (see
        :func:`ashare_model.ops.cross_sectional_zscore`), so all consumers —
        training reward, protocol scoring, backtest and the simulation
        runner — see one standardized formula semantics.  The z-score's
        reference set is the eligible & finite cells of the VM's PIT mask
        and the signal keeps NaN at ineligible cells, so ineligible stocks
        can never be sorted into a position.
        """

        stack: list[torch.Tensor] = []
        try:
            for raw_token in formula_tokens:
                token = int(raw_token)
                if token == self.vocab.pad_token_id:
                    continue
                if token < self.operator_offset:
                    feature_idx = token - self.feature_offset
                    if feature_idx < 0 or feature_idx >= factor_tensor.shape[0]:
                        return None
                    stack.append(factor_tensor[feature_idx])
                    continue
                if token not in self.op_map:
                    return None
                arity = self.arity_map[token]
                if len(stack) < arity:
                    return None
                args = [stack.pop() for _ in range(arity)]
                args.reverse()
                name = self._cs_token_names.get(token)
                if name is not None:
                    # Cross-sectional operators consume the PIT eligibility
                    # mask carried on the VM (CS_NEUTRALIZE also consumes
                    # the industry groups).
                    if name == "CS_NEUTRALIZE":
                        result = _cs_neutralize(
                            args[0], self.industry_codes, eligible=self.universe_mask
                        )
                    elif name == "CS_RANK":
                        result = _cs_rank(args[0], self.universe_mask)
                    elif name == "CS_ZSCORE":
                        result = cross_sectional_zscore(
                            args[0], self.universe_mask
                        )
                    else:  # CS_DEMEAN
                        result = _cs_demean(args[0], self.universe_mask)
                else:
                    result = self.op_map[token](*args)
                # Non-finite results are clamped to the neutral value 0 so a
                # single degenerate operator cannot fabricate extreme signals.
                result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                stack.append(result)
            if len(stack) != 1:
                return None
            return cross_sectional_zscore(stack[-1], self.universe_mask)
        except Exception:
            return None


def formula_decode(tokens: Iterable[int], vocab=None) -> str:
    """Convert postfix tokens to a human-readable infix formula."""

    vocab = vocab or FORMULA_VOCAB
    names = vocab.token_names
    arity = {vocab.operator_offset + i: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
    op_names = {vocab.operator_offset + i: cfg[0] for i, cfg in enumerate(OPS_CONFIG)}
    stack: list[str] = []

    for raw_token in tokens:
        token = int(raw_token)
        if token == vocab.pad_token_id:
            continue
        if token < vocab.operator_offset:
            feature_idx = token - 1
            if feature_idx < 0 or feature_idx >= len(vocab.feature_names):
                return "INVALID"
            stack.append(vocab.feature_names[feature_idx])
            continue
        if token not in arity:
            return "INVALID"
        a = arity[token]
        if len(stack) < a:
            return "INVALID"
        args = [stack.pop() for _ in range(a)]
        args.reverse()
        op = op_names[token]
        if a == 1:
            stack.append(f"{op}({args[0]})")
        elif a == 2:
            stack.append(f"({args[0]} {op} {args[1]})")
        else:
            stack.append(f"{op}({', '.join(args)})")

    return stack[-1] if len(stack) == 1 else "INVALID"
