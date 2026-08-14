"""
Implements draft-target speculation pipelines, parallel target verification,
modified rejection sampling, and token acceptance rate analytics.
"""

from __future__ import annotations

import time
import dataclasses
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class SpeculativeBatchResult:
    """Holds execution result metrics for a speculative step."""

    accepted_tokens: torch.Tensor  # [batch_size, num_accepted]
    num_accepted: int
    draft_tokens_generated: int
    acceptance_rate: float
    draft_latency_ms: float
    target_latency_ms: float


class SpeculativeEngine:
    """High-throughput Speculative Decoding Engine for CUDAForge."""

    def __init__(
        self,
        draft_runner: Any,
        target_runner: Any,
        num_speculative_tokens: int = 5,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> None:
        """Initialize Speculative Engine.

        Args:
            draft_runner: Model runner for fast draft token proposals.
            target_runner: Main model runner for batch verification.
            num_speculative_tokens: K draft tokens to generate per speculation step.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
        """
        self.draft_runner = draft_runner
        self.target_runner = target_runner
        self.k = num_speculative_tokens
        self.temperature = max(temperature, 1e-5)
        self.top_p = top_p

        # Analytics tracking
        self.total_draft_tokens: int = 0
        self.total_accepted_tokens: int = 0

    @torch.no_grad()
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Applies temperature scaling, top-p filtering, and samples next token."""
        logits = logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above top_p threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        kv_cache_handles: Optional[Dict[str, Any]] = None,
    ) -> SpeculativeBatchResult:
        """Executes one round of speculative draft generation, target verification, and rejection sampling.

        Args:
            input_ids: Current prefix tokens [batch_size, seq_len]
            kv_cache_handles: Optional pointers to paged KV-cache state

        Returns:
            SpeculativeBatchResult with accepted sequence and performance metrics.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # ------------------------------------------------------------------
        # Phase 1: Draft Model Auto-Regression (Generate K tokens)
        # ------------------------------------------------------------------
        t_draft_start = time.perf_counter()

        draft_input_ids = input_ids.clone()
        draft_tokens_list: List[torch.Tensor] = []
        draft_probs_list: List[torch.Tensor] = []

        for _ in range(self.k):
            # Run forward pass on small draft model
            draft_logits = self.draft_runner.forward(draft_input_ids, kv_cache_handles)
            next_logits = draft_logits[:, -1, :]

            probs = F.softmax(next_logits / self.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            draft_tokens_list.append(next_token)
            draft_probs_list.append(probs)
            draft_input_ids = torch.cat([draft_input_ids, next_token], dim=-1)

        t_draft_ms = (time.perf_counter() - t_draft_start) * 1000.0

        # Stack draft outputs
        draft_tokens = torch.cat(draft_tokens_list, dim=-1)  # [batch_size, K]
        draft_probs = torch.stack(
            draft_probs_list, dim=1
        )  # [batch_size, K, vocab_size]

        # ------------------------------------------------------------------
        # Phase 2: Target Model Parallel Verification (K + 1 tokens)
        # ------------------------------------------------------------------
        t_target_start = time.perf_counter()

        verification_input_ids = torch.cat([input_ids, draft_tokens], dim=-1)
        target_logits = self.target_runner.forward(
            verification_input_ids, kv_cache_handles
        )

        # Extract target probabilities over candidate draft positions
        target_candidate_logits = target_logits[:, -(self.k + 1) :, :]
        target_probs = F.softmax(target_candidate_logits / self.temperature, dim=-1)

        t_target_ms = (time.perf_counter() - t_target_start) * 1000.0

        # ------------------------------------------------------------------
        # Phase 3: Vectorized Rejection Sampling
        # ------------------------------------------------------------------
        accepted_tokens_list: List[torch.Tensor] = []
        num_accepted = 0

        for i in range(self.k):
            candidate_token = draft_tokens[:, i : i + 1]  # [batch_size, 1]

            p_draft = draft_probs[:, i, :].gather(dim=-1, index=candidate_token)
            q_target = target_probs[:, i, :].gather(dim=-1, index=candidate_token)

            r = torch.rand_like(p_draft)
            acceptance_ratio = torch.minimum(
                torch.ones_like(q_target), q_target / (p_draft + 1e-8)
            )

            if (r < acceptance_ratio).all():
                accepted_tokens_list.append(candidate_token)
                num_accepted += 1
            else:
                # Reject token: sample replacement from modified distribution max(0, q - p)
                modified_dist = torch.clamp(
                    target_probs[:, i, :] - draft_probs[:, i, :], min=0.0
                )
                sum_dist = modified_dist.sum(dim=-1, keepdim=True)

                # Fallback to pure target distribution if normalized difference is zero
                fallback_mask = sum_dist <= 1e-6
                modified_dist = torch.where(
                    fallback_mask, target_probs[:, i, :], modified_dist
                )

                resampled_token = torch.multinomial(modified_dist, num_samples=1)
                accepted_tokens_list.append(resampled_token)
                num_accepted += 1
                break  # Stop speculation upon first rejection

        # If all K tokens were accepted, sample additional token from final target position
        if num_accepted == self.k:
            final_token = self._sample_next_token(
                target_candidate_logits[:, -1, :],
                temperature=self.temperature,
                top_p=self.top_p,
            )
            accepted_tokens_list.append(final_token)
            num_accepted += 1

        accepted_tokens = torch.cat(accepted_tokens_list, dim=-1)

        # Update lifetime telemetry statistics
        self.total_draft_tokens += self.k
        self.total_accepted_tokens += num_accepted
        acceptance_rate = num_accepted / self.k

        return SpeculativeBatchResult(
            accepted_tokens=accepted_tokens,
            num_accepted=num_accepted,
            draft_tokens_generated=self.k,
            acceptance_rate=acceptance_rate,
            draft_latency_ms=t_draft_ms,
            target_latency_ms=t_target_ms,
        )

    def get_aggregate_acceptance_rate(self) -> float:
        """Returns overall token acceptance rate ratio."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens
