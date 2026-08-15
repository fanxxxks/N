"""StackVM interpreter for A-share factor formulas."""

from __future__ import annotations

from typing import Iterable

import torch

from .ops import OP_ARITY, OPS_CONFIG
from .vocab import FORMULA_VOCAB


class StackVM:
    def __init__(self, vocab=None):
        self.vocab = vocab or FORMULA_VOCAB
        self.feature_offset = 1
        self.operator_offset = self.vocab.operator_offset
        self.op_map = {
            self.operator_offset + i: cfg[1] for i, cfg in enumerate(OPS_CONFIG)
        }
        self.arity_map = {
            self.operator_offset + i: cfg[2] for i, cfg in enumerate(OPS_CONFIG)
        }

    def execute(
        self,
        formula_tokens: Iterable[int],
        factor_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        """Execute a postfix formula over ``[Feature, Stock, Time]`` input."""

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
                result = self.op_map[token](*args)
                # Non-finite results are clamped to the neutral value 0 so a
                # single degenerate operator cannot fabricate extreme signals.
                result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                stack.append(result)
            return stack[-1] if len(stack) == 1 else None
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
