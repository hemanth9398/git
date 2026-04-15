"""
Speculative Decoding

Implements speculative decoding (also known as speculative sampling or
draft-model decoding). A small, fast "draft" model proposes K tokens in
a single forward pass; the large "target" model then verifies all K tokens
in a single forward pass (much cheaper than K serial passes).

Algorithm (Chen et al., 2023 — "Accelerating Large Language Model Decoding
with Speculative Sampling"):

  1. Draft model generates K candidate tokens autoregressively.
  2. Target model runs a single forward pass on the original context +
     the K draft tokens, producing K+1 probability distributions.
  3. For each draft token i:
     - Let p = target distribution at position i
     - Let q = draft distribution at position i
     - Accept token x_i with probability min(1, p(x_i) / q(x_i)).
     - If rejected, sample the "corrected" token from the residual:
       max(0, p - q) / sum(max(0, p - q))
  4. The output is identical in distribution to sampling from the target
     model alone, but typically 2-4x faster due to fewer serial passes.

References:
  - https://arxiv.org/abs/2302.01318
  - https://arxiv.org/abs/2211.17192 (Leviathan et al., "Fast Inference")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .sampler import SamplingConfig, apply_temperature, top_p_filter, top_k_filter

logger = logging.getLogger("speculative-decoding")

# Type alias for a model forward function:
# Callable[[token_ids: list[int]], logits: torch.Tensor]
# where logits shape is [len(token_ids), vocab_size].
ModelFn = Callable[[list[int]], torch.Tensor]


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    # Number of draft tokens to propose per iteration.
    num_draft_tokens: int = 5
    # Sampling config applied to both draft and target models.
    sampling: SamplingConfig = None

    def __post_init__(self) -> None:
        if self.sampling is None:
            self.sampling = SamplingConfig()


def speculative_decode(
    context_ids: list[int],
    draft_model: ModelFn,
    target_model: ModelFn,
    config: SpeculativeConfig,
    eos_token_id: int = 2,
) -> list[int]:
    """
    Generate tokens using speculative decoding.

    Returns the list of accepted token IDs (may be fewer than num_draft_tokens
    if some draft tokens are rejected or EOS is reached).

    Args:
        context_ids:   Input token IDs (prompt + already-generated tokens).
        draft_model:   Small model that generates draft proposals.
        target_model:  Large target model that verifies proposals.
        config:        Speculative decoding configuration.
        eos_token_id:  Token ID that terminates generation.
    """
    K = config.num_draft_tokens
    cfg = config.sampling

    # ── Step 1: Draft phase ────────────────────────────────────────────────
    # Autoregressively generate K draft tokens using the draft model.
    draft_token_ids: list[int] = []
    draft_probs: list[torch.Tensor] = []  # draft distribution at each step
    current_ids = list(context_ids)

    for _ in range(K):
        logits = draft_model(current_ids)
        last_logits = logits[-1]  # [vocab_size]
        last_logits = apply_temperature(last_logits, cfg.temperature)
        last_logits = top_k_filter(last_logits, cfg.top_k)
        last_logits = top_p_filter(last_logits, cfg.top_p)
        q = F.softmax(last_logits, dim=-1)
        draft_probs.append(q)

        token_id = int(torch.multinomial(q, num_samples=1).item())
        draft_token_ids.append(token_id)
        current_ids.append(token_id)

        if token_id == eos_token_id:
            break

    # ── Step 2: Target verification ────────────────────────────────────────
    # Single forward pass of the target model on context + draft tokens.
    verify_ids = list(context_ids) + draft_token_ids
    target_logits = target_model(verify_ids)  # [len(verify_ids), vocab_size]

    # We only need the logits at positions len(context_ids)-1 through
    # len(context_ids) + len(draft_token_ids) - 1.
    offset = len(context_ids) - 1

    accepted: list[int] = []
    for i, draft_id in enumerate(draft_token_ids):
        t_logits = target_logits[offset + i]
        t_logits = apply_temperature(t_logits, cfg.temperature)
        t_logits = top_k_filter(t_logits, cfg.top_k)
        t_logits = top_p_filter(t_logits, cfg.top_p)
        p = F.softmax(t_logits, dim=-1)  # target distribution
        q = draft_probs[i]               # draft distribution

        # Acceptance probability: min(1, p(x) / q(x)).
        p_x = p[draft_id].item()
        q_x = max(q[draft_id].item(), 1e-10)
        accept_prob = min(1.0, p_x / q_x)

        if torch.rand(1).item() < accept_prob:
            accepted.append(draft_id)
            if draft_id == eos_token_id:
                return accepted
        else:
            # Rejection: sample a corrected token from the residual distribution.
            residual = torch.clamp(p - q, min=0.0)
            residual_sum = residual.sum()
            if residual_sum > 0:
                corrected_id = int(torch.multinomial(residual / residual_sum, 1).item())
            else:
                corrected_id = int(p.argmax().item())
            accepted.append(corrected_id)
            logger.debug(
                "Rejected draft token %d at position %d, corrected to %d "
                "(p=%.4f, q=%.4f, accept_prob=%.4f)",
                draft_id, i, corrected_id, p_x, q_x, accept_prob,
            )
            return accepted

    # All draft tokens accepted — also sample one bonus token from target.
    bonus_logits = target_logits[offset + len(draft_token_ids)]
    bonus_logits = apply_temperature(bonus_logits, cfg.temperature)
    bonus_logits = top_k_filter(bonus_logits, cfg.top_k)
    bonus_logits = top_p_filter(bonus_logits, cfg.top_p)
    bonus_p = F.softmax(bonus_logits, dim=-1)
    bonus_id = int(torch.multinomial(bonus_p, num_samples=1).item())
    accepted.append(bonus_id)

    return accepted
