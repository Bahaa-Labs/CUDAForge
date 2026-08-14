#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <vector>

#include "cudaforge/scheduler/continuous_batcher.h"

namespace cudaforge::engine {

/*
    Thread-safe struct representing a single generated token event.
 */
struct StreamOutput {
  uint64_t request_id;
  int32_t token_id;
  bool is_final;

  StreamOutput() = default;
  StreamOutput(uint64_t req_id, int32_t tok_id, bool final_flag)
      : request_id(req_id), token_id(tok_id), is_final(final_flag) {}
};

/**
 * @brief Thread-safe Token-by-Token Streaming Buffer.
 * Allows C++ inference loop to push generated tokens with zero python GIL
 * contention, while Python async clients drain tokens in real time.
 */
class TokenStreamBuffer {
public:
  TokenStreamBuffer() = default;

  void push(const StreamOutput &output);
  void push_batch(const std::vector<StreamOutput> &outputs);
  std::vector<StreamOutput> pop_all();
  bool pop_timeout(StreamOutput &output, int timeout_ms);
  size_t size() const;
  bool empty() const;

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<StreamOutput> queue_;
};

/**
 * @brief High-Performance Model Runner managing continuous batching execution,
 * mid-flight request cancellation, and asynchronous streaming output.
 */
class ModelRunner {
public:
  ModelRunner(std::shared_ptr<scheduler::ContinuousBatcher> batcher,
              size_t vocab_size, int32_t eos_token_id);

  ~ModelRunner() = default;

  // Trigger mid-flight cancellation for a specific request ID
  void cancel_request(uint64_t request_id);

  // Query if a request has been cancelled mid-flight
  bool is_cancelled(uint64_t request_id) const;

  // Execute a single step of the autoregressive continuous batching loop
  size_t step();

  // Access the shared token streaming buffer
  std::shared_ptr<TokenStreamBuffer> get_stream_buffer() const {
    return stream_buffer_;
  }

private:
  std::shared_ptr<scheduler::ContinuousBatcher> batcher_;
  std::shared_ptr<TokenStreamBuffer> stream_buffer_;
  size_t vocab_size_;
  int32_t eos_token_id_;

  mutable std::mutex cancel_mutex_;
  std::unordered_set<uint64_t> cancelled_requests_;

  void filter_cancelled_requests(scheduler::BatchStepResult &batch_result);
};

} // namespace cudaforge::engine