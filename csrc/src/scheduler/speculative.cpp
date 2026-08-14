#include "cudaforge/scheduler/speculative.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cudaforge::scheduler {

SpeculativeEngine::SpeculativeEngine(const SpeculativeConfig& config)
    : config_(config), rng_(config.rng_seed) {}

int32_t SpeculativeEngine::sample_from_distribution(const std::vector<float>& probs) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cumulative = 0.0f;

    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (r <= cumulative) {
            return static_cast<int32_t>(i);
        }
    }
    return static_cast<int32_t>(probs.size() - 1);
}

int32_t SpeculativeEngine::sample_adjusted_distribution(const std::vector<float>& target_p,
                                                     const std::vector<float>& draft_q) {
    size_t vocab_size = target_p.size();
    std::vector<float> adjusted(vocab_size, 0.0f);
    float sum = 0.0f;

    for (size_t i = 0; i < vocab_size; ++i) {
        adjusted[i] = std::max(0.0f, target_p[i] - draft_q[i]);
        sum += adjusted[i];
    }

    if (sum <= 1e-6f) {
        // Fallback to target distribution if difference distribution is zero
        return sample_from_distribution(target_p);
    }

    // Normalize adjusted distribution
    for (size_t i = 0; i < vocab_size; ++i) {
        adjusted[i] /= sum;
    }

    return sample_from_distribution(adjusted);
}

SpeculativeVerificationResult SpeculativeEngine::verify_candidates(
    const std::vector<int32_t>& draft_tokens,
    const std::vector<std::vector<float>>& draft_probs,
    const std::vector<std::vector<float>>& target_probs,
    int32_t eos_token_id) {

    std::lock_guard<std::mutex> lock(engine_mutex_);
    SpeculativeVerificationResult result;

    size_t K = draft_tokens.size();
    if (draft_probs.size() != K || target_probs.size() < K + 1) {
        throw std::invalid_argument("Incompatible draft and target tensor dimensions in SpeculativeEngine.");
    }

    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

    for (size_t i = 0; i < K; ++i) {
        int32_t candidate_token = draft_tokens[i];
        float q = draft_probs[i][candidate_token];   // Draft model prob
        float p = target_probs[i][candidate_token];  // Target model prob

        float acceptance_ratio = (q > 0.0f) ? std::min(1.0f, p / q) : 0.0f;
        float r = uniform_dist(rng_);

        if (r <= acceptance_ratio) {
            // Token Accepted
            result.accepted_tokens.push_back(candidate_token);
            result.num_accepted++;

            if (candidate_token == eos_token_id) {
                result.end_of_sequence = true;
                result.final_token = eos_token_id;
                return result;
            }
        } else {
            // Token Rejected at index i: Sample replacement from p'(x) = max(0, p(x) - q(x))
            result.final_token = sample_adjusted_distribution(target_probs[i], draft_probs[i]);
            if (result.final_token == eos_token_id) {
                result.end_of_sequence = true;
            }
            return result;
        }
    }

    // All K draft tokens accepted: Sample bonus token from target_probs[K]
    result.final_token = sample_from_distribution(target_probs[K]);
    if (result.final_token == eos_token_id) {
        result.end_of_sequence = true;
    }

    return result;
}

SpeculativeVerificationResult SpeculativeEngine::verify_candidates_greedy(
    const std::vector<int32_t>& draft_tokens,
    const std::vector<std::vector<float>>& target_probs,
    int32_t eos_token_id) {

    std::lock_guard<std::mutex> lock(engine_mutex_);
    SpeculativeVerificationResult result;
    size_t K = draft_tokens.size();

    for (size_t i = 0; i < K; ++i) {
        int32_t candidate_token = draft_tokens[i];

        // Find argmax of target model logits at position i
        auto max_it = std::max_element(target_probs[i].begin(), target_probs[i].end());
        int32_t target_argmax = static_cast<int32_t>(std::distance(target_probs[i].begin(), max_it));

        if (candidate_token == target_argmax) {
            result.accepted_tokens.push_back(candidate_token);
            result.num_accepted++;

            if (candidate_token == eos_token_id) {
                result.end_of_sequence = true;
                result.final_token = eos_token_id;
                return result;
            }
        } else {
            result.final_token = target_argmax;
            if (result.final_token == eos_token_id) {
                result.end_of_sequence = true;
            }
            return result;
        }
    }

    // All accepted, sample bonus argmax token
    auto bonus_it = std::max_element(target_probs[K].begin(), target_probs[K].end());
    result.final_token = static_cast<int32_t>(std::distance(target_probs[K].begin(), bonus_it));
    if (result.final_token == eos_token_id) {
        result.end_of_sequence = true;
    }

    return result;
}

} // namespace cudaforge::scheduler