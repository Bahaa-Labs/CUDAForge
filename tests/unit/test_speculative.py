import pytest
import torch
from cudaforge import SpeculativeConfig, SpeculativeEngine


def test_greedy_speculative_verification():
    config = SpeculativeConfig()
    config.gamma = 3
    engine = SpeculativeEngine(config)

    draft_tokens = [10, 20, 30]

    # Target logits matching draft candidates for first 2 tokens, but differing at 3rd
    target_probs = [
        [0.1, 0.0, 0.0] + [0.0] * 7 + [0.9],  # Argmax: 9 -> Wait, argmax 9 = index 9
        [0.0] * 10,
        [0.0] * 10,
        [0.0] * 10,
    ]
    
    # Vocabulary size = 5
    vocab_size = 5
    draft_tokens = [1, 2, 3]

    # Target distribution perfectly matching draft for indices 0 and 1, but predicting 4 for index 2
    target_probs = [
        [0.0, 1.0, 0.0, 0.0, 0.0],  # Argmax = 1 (Matches draft_tokens[0])
        [0.0, 0.0, 1.0, 0.0, 0.0],  # Argmax = 2 (Matches draft_tokens[1])
        [0.0, 0.0, 0.0, 0.0, 1.0],  # Argmax = 4 (Mismatches draft_tokens[2] = 3)
        [1.0, 0.0, 0.0, 0.0, 0.0],  # Bonus distribution
    ]

    result = engine.verify_candidates_greedy(draft_tokens, target_probs, eos_token_id=0)

    assert result.num_accepted == 2
    assert result.accepted_tokens == [1, 2]
    assert result.final_token == 4


def test_full_acceptance_speculative_verification():
    config = SpeculativeConfig()
    config.gamma = 2
    engine = SpeculativeEngine(config)

    draft_tokens = [1, 2]
    target_probs = [
        [0.0, 1.0, 0.0],  # Argmax = 1
        [0.0, 0.0, 1.0],  # Argmax = 2
        [1.0, 0.0, 0.0],  # Bonus token = 0 (EOS)
    ]

    result = engine.verify_candidates_greedy(draft_tokens, target_probs, eos_token_id=0)

    assert result.num_accepted == 2
    assert result.accepted_tokens == [1, 2]
    assert result.final_token == 0
    assert result.end_of_sequence