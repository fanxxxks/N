"""Supervised next-token imitation over baseline elite formulas."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .elite_archive import EliteArchive

IMITATION_VERSION = 1
_ALLOWED_SOURCES = frozenset({"gp", "tpe", "random"})


@dataclass(frozen=True)
class ImitationResult:
    version: int
    epochs: int
    sample_count: int
    token_count: int
    initial_loss: float
    final_loss: float
    initial_token_accuracy: float
    final_token_accuracy: float
    epoch_losses: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "epochs": self.epochs,
            "sample_count": self.sample_count,
            "token_count": self.token_count,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "initial_token_accuracy": self.initial_token_accuracy,
            "final_token_accuracy": self.final_token_accuracy,
            "epoch_losses": list(self.epoch_losses),
        }


def _fixed_length_tokens(entry_tokens, *, vocab, max_formula_len: int) -> list[int]:
    tokens = [int(token) for token in entry_tokens]
    eos = vocab.eos_token_id
    pad = int(vocab.pad_token_id)
    while tokens and tokens[-1] == pad:
        tokens.pop()
    if eos is not None and eos in tokens:
        tokens = tokens[: tokens.index(eos) + 1]
    elif eos is not None:
        tokens.append(int(eos))
    if len(tokens) > max_formula_len:
        raise ValueError(
            f"elite formula length {len(tokens)} exceeds max_formula_len "
            f"{max_formula_len}"
        )
    if any(token < 0 or token >= vocab.size for token in tokens):
        raise ValueError("elite archive contains a token outside the model vocabulary")
    return tokens + [pad] * (max_formula_len - len(tokens))


def _teacher_logits(model, targets: torch.Tensor) -> torch.Tensor:
    """All next-token logits under the model's existing PAD/BOS convention."""

    batch, length = targets.shape
    prefix = torch.full(
        (batch, 1),
        int(model.vocab.pad_token_id),
        dtype=torch.long,
        device=targets.device,
    )
    logits: list[torch.Tensor] = []
    for position in range(length):
        next_logits, _ = model(prefix)
        logits.append(next_logits)
        prefix = torch.cat([prefix, targets[:, position : position + 1]], dim=1)
    return torch.stack(logits, dim=1)


def _metrics(model, targets: torch.Tensor) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = _teacher_logits(model, targets)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            accuracy = (logits.argmax(dim=-1) == targets).float().mean()
            return float(loss), float(accuracy)
    finally:
        model.train(was_training)


def pretrain_on_elites(
    model,
    archive: EliteArchive,
    *,
    max_formula_len: int,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> ImitationResult:
    """Teacher-force a policy on GP/TPE/Random elite token sequences."""

    sources = set(archive.source_backends)
    if not sources.issubset(_ALLOWED_SOURCES):
        raise ValueError("imitation archive sources must be gp, tpe or random")
    if not archive.entries:
        raise ValueError("imitation requires a non-empty elite archive")
    epochs = int(epochs)
    batch_size = int(batch_size)
    max_formula_len = int(max_formula_len)
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if max_formula_len < 2:
        raise ValueError("max_formula_len must be at least 2")

    device = next(model.parameters()).device
    targets = torch.tensor(
        [
            _fixed_length_tokens(
                entry.tokens,
                vocab=model.vocab,
                max_formula_len=max_formula_len,
            )
            for entry in archive.entries
        ],
        dtype=torch.long,
        device=device,
    )
    torch.manual_seed(int(seed))
    initial_loss, initial_accuracy = _metrics(model, targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    epoch_losses: list[float] = []
    was_training = model.training
    model.train()
    try:
        for _ in range(epochs):
            order = torch.randperm(targets.shape[0], generator=generator)
            total_loss = 0.0
            total_tokens = 0
            for start in range(0, targets.shape[0], batch_size):
                indices = order[start : start + batch_size].to(device)
                batch_targets = targets.index_select(0, indices)
                logits = _teacher_logits(model, batch_targets)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch_targets.reshape(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count = int(batch_targets.numel())
                total_loss += float(loss.detach()) * count
                total_tokens += count
            epoch_losses.append(total_loss / total_tokens)
    finally:
        model.train(was_training)
    final_loss, final_accuracy = _metrics(model, targets)
    return ImitationResult(
        version=IMITATION_VERSION,
        epochs=epochs,
        sample_count=int(targets.shape[0]),
        token_count=int(targets.numel()),
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_token_accuracy=initial_accuracy,
        final_token_accuracy=final_accuracy,
        epoch_losses=tuple(epoch_losses),
    )
