"""
LLM Sampling Strategies

Implements the full taxonomy of token sampling algorithms used in LLM inference:

  1. Greedy decoding — argmax over logits
  2. Temperature scaling — logits / T before softmax
  3. Top-k filtering — keep only the top-k most probable tokens
  4. Top-p (nucleus) sampling — keep tokens summing to probability mass ≥ p
  5. Min-p filtering — relative threshold based on max probability
  6. Beam search — maintain K candidate sequences
  7. Repetition penalty — discourage repeating already-generated tokens
  8. Frequency / presence penalties (OpenAI-compatible)

All functions operate on raw logit tensors (torch.Tensor, shape [vocab_size]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingConfig:
    """Configuration for a single sampling call."""

    temperature: float = 1.0
    top_k: int = -1          # -1 = disabled
    top_p: float = 1.0       # 1.0 = disabled
    min_p: float = 0.0       # 0.0 = disabled
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # Beam search specific.
    beam_width: int = 1
    length_penalty: float = 1.0
    # Token IDs already generated (for repetition/frequency penalties).
    generated_token_ids: list[int] = field(default_factory=list)


# ── Core sampling functions ────────────────────────────────────────────────

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by 1/temperature. temperature=0 → greedy (returns argmax)."""
    if temperature == 0.0:
        return logits
    if temperature < 1e-6:
        temperature = 1e-6
    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """
    Apply repetition penalty: logit[id] /= penalty if id was generated before
    (for positive logits, /penalty reduces probability; for negative, *penalty).
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    unique_ids = torch.tensor(list(set(generated_ids)), dtype=torch.long, device=logits.device)
    score = logits[unique_ids]
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits = logits.clone()
    logits[unique_ids] = score
    return logits


def apply_frequency_presence_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    """
    Apply OpenAI-style frequency and presence penalties.
    frequency_penalty penalises tokens proportional to how often they appear.
    presence_penalty applies a flat penalty to any token that has appeared.
    """
    if frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits
    logits = logits.clone()
    freq: dict[int, int] = {}
    for tid in generated_ids:
        freq[tid] = freq.get(tid, 0) + 1
    for tid, count in freq.items():
        logits[tid] -= frequency_penalty * count + presence_penalty
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: keep the smallest set of tokens whose cumulative
    probability mass is ≥ p, zero-out the rest.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens whose cumulative probability exceeds p (shift right by one
    # so we include the token that crosses the threshold).
    sorted_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= p
    sorted_logits[sorted_mask] = float("-inf")
    logits = logits.clone()
    logits.scatter_(0, sorted_indices, sorted_logits)
    return logits


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """
    Min-p sampling (vLLM 0.4+): a token is considered only if its probability
    is ≥ min_p * max_probability. This adapts the threshold to the distribution.
    """
    if min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max()
    threshold = min_p * max_prob
    return logits.masked_fill(probs < threshold, float("-inf"))


# ── Main sampling function ─────────────────────────────────────────────────

def sample(logits: torch.Tensor, config: SamplingConfig) -> int:
    """
    Sample the next token from logits using the configured strategy.

    Returns a single token ID (int).
    """
    # Apply penalties.
    if config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(
            logits, config.generated_token_ids, config.repetition_penalty
        )
    if config.frequency_penalty != 0.0 or config.presence_penalty != 0.0:
        logits = apply_frequency_presence_penalty(
            logits, config.generated_token_ids,
            config.frequency_penalty, config.presence_penalty,
        )

    # Greedy shortcut.
    if config.temperature == 0.0:
        return int(logits.argmax().item())

    logits = apply_temperature(logits, config.temperature)
    logits = top_k_filter(logits, config.top_k)
    logits = top_p_filter(logits, config.top_p)
    logits = min_p_filter(logits, config.min_p)

    # Sample from the filtered distribution.
    probs = F.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()
    return int(token_id)


def greedy_decode(logits: torch.Tensor) -> int:
    """Return the argmax token (greedy decoding)."""
    return int(logits.argmax().item())


# ── Beam search ───────────────────────────────────────────────────────────

@dataclass
class BeamHypothesis:
    token_ids: list[int]
    log_prob: float = 0.0

    def score(self, length_penalty: float) -> float:
        """Length-normalized beam score."""
        length = max(len(self.token_ids), 1)
        return self.log_prob / (length ** length_penalty)


def beam_search_step(
    hypotheses: list[BeamHypothesis],
    logits_batch: torch.Tensor,
    beam_width: int,
    length_penalty: float = 1.0,
    eos_token_id: int = 2,
) -> tuple[list[BeamHypothesis], list[BeamHypothesis]]:
    """
    Execute one beam search expansion step.

    Args:
        hypotheses: Current live beam hypotheses.
        logits_batch: Logits tensor [num_beams, vocab_size].
        beam_width: Number of beams to keep.
        length_penalty: Exponent for length normalization.
        eos_token_id: Token ID that terminates a hypothesis.

    Returns:
        (live_hypotheses, completed_hypotheses)
    """
    log_probs = F.log_softmax(logits_batch, dim=-1)  # [num_beams, vocab_size]
    vocab_size = log_probs.shape[-1]

    candidates: list[BeamHypothesis] = []
    completed: list[BeamHypothesis] = []

    for i, hyp in enumerate(hypotheses):
        # Get top-beam_width extensions for this hypothesis.
        top_log_probs, top_ids = torch.topk(log_probs[i], beam_width)
        for log_p, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
            new_ids = hyp.token_ids + [token_id]
            new_log_prob = hyp.log_prob + log_p
            new_hyp = BeamHypothesis(token_ids=new_ids, log_prob=new_log_prob)
            if token_id == eos_token_id:
                completed.append(new_hyp)
            else:
                candidates.append(new_hyp)

    # Keep top beam_width candidates by score.
    candidates.sort(key=lambda h: h.score(length_penalty), reverse=True)
    return candidates[:beam_width], completed
