"""StackVM interpreter for A-share factor formulas.

Output contract: every executed formula returns its signal **cross-sectionally
z-scored per date** (:func:`ashare_model.ops.cross_sectional_zscore`), applied
as the final step of :meth:`StackVM.execute`.  Stacked arithmetic (MUL/DIV/ADD
chains) can drift the raw scale by orders of magnitude; the terminal
standardization keeps the scale meaningful for threshold semantics (GATE/JUMP
compare against 0) and gives every formula a common scale for future signal
combination.  It is a monotone per-date transformation, so it changes neither
the rank IC nor the top-n selection of a signal.

The VM also carries an optional ``industry_codes`` tensor (``[stock, date]``
discrete group ids) consumed by the ``CS_NEUTRALIZE`` operator; without it
the operator degrades to the full-market demean.
"""

from __future__ import annotations

from typing import Iterable

import torch

from .ops import OPS_CONFIG, _cs_neutralize, cross_sectional_zscore
from .vocab import FORMULA_VOCAB


class StackVM:
    def __init__(self, vocab=None, industry_codes: torch.Tensor | None = None):
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
        self._neutralize_token = self.operator_offset + next(
            i for i, (name, _, _) in enumerate(OPS_CONFIG) if name == "CS_NEUTRALIZE"
        )

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
        runner — see one standardized formula semantics.
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
                if token == self._neutralize_token:
                    result = _cs_neutralize(args[0], self.industry_codes)
                else:
                    result = self.op_map[token](*args)
                # Non-finite results are clamped to the neutral value 0 so a
                # single degenerate operator cannot fabricate extreme signals.
                result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                stack.append(result)
            if len(stack) != 1:
                return None
            return cross_sectional_zscore(stack[-1])
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
