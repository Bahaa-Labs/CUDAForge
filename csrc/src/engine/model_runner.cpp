#include "cudaforge/engine/model_runner.h"
#include <algorithm>
#include <chrono>

namespace cudaforge::engine {

// ============================================================================
// TokenStreamBuffer Implementation
// ============================================================================

void TokenStreamBuffer::push(const StreamOutput &output) {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.push(output);
  cv_.notify_one();
}

void TokenStreamBuffer::push_batch(const std::vector<StreamOutput> &outputs) {
  if (outputs.empty())
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &item : outputs) {
    queue_.push(item);
  }
  cv_.notify_all();
}

std::vector<StreamOutput> TokenStreamBuffer::pop_all() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<StreamOutput> result;
  result.reserve(queue_.size());
  while (!queue_.empty()) {
    result.push_back(queue_.front());
    queue_.pop();
  }
  return result;
}

bool TokenStreamBuffer::pop_timeout(StreamOutput &output, int timeout_ms) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                    [this]() { return !queue_.empty(); })) {
    return false;
  }
  output = queue_.front();
  queue_.pop();
  return true;
}

size_t TokenStreamBuffer::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

bool TokenStreamBuffer::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

// ============================================================================
// ModelRunner Implementation
// ============================================================================

ModelRunner::ModelRunner(std::shared_ptr<scheduler::ContinuousBatcher> batcher,
                         size_t vocab_size, int32_t eos_token_id)
    : batcher_(std::move(batcher)),
      stream_buffer_(std::make_shared<TokenStreamBuffer>()),
      vocab_size_(vocab_size), eos_token_id_(eos_token_id) {}

void ModelRunner::cancel_request(uint64_t request_id) {
  {
    std::lock_guard<std::mutex> lock(cancel_mutex_);
    cancelled_requests_.insert(request_id);
  }
  if (batcher_) {
    batcher_->cancel_request(request_id);
  }
}

bool ModelRunner::is_cancelled(uint64_t request_id) const {
  std::lock_guard<std::mutex> lock(cancel_mutex_);
  return cancelled_requests_.find(request_id) != cancelled_requests_.end();
}

void ModelRunner::filter_cancelled_requests(
    scheduler::BatchStepResult &batch_result) {
  std::lock_guard<std::mutex> lock(cancel_mutex_);
  if (cancelled_requests_.empty())
    return;

  auto is_cancelled_fn =
      [this](const std::shared_ptr<scheduler::Request> &req) {
        return cancelled_requests_.find(req->id) != cancelled_requests_.end();
      };

  batch_result.prefill_requests.erase(
      std::remove_if(batch_result.prefill_requests.begin(),
                     batch_result.prefill_requests.end(), is_cancelled_fn),
      batch_result.prefill_requests.end());

  batch_result.decode_requests.erase(
      std::remove_if(batch_result.decode_requests.begin(),
                     batch_result.decode_requests.end(), is_cancelled_fn),
      batch_result.decode_requests.end());
}

size_t ModelRunner::step() {
  if (!batcher_)
    return 0;

  // 1. Get batch schedule decision from Continuous Batcher
  auto step_result = batcher_->schedule_step();

  // 2. Filter out requests cancelled mid-flight
  filter_cancelled_requests(step_result);

  size_t total_active =
      step_result.prefill_requests.size() + step_result.decode_requests.size();
  if (total_active == 0) {
    return 0;
  }

  std::vector<StreamOutput> generated_outputs;
  std::vector<uint64_t> finished_ids;

  // 3. Execute Prefill Step
  for (auto &req : step_result.prefill_requests) {
    int32_t sampled_token = req->prompt_tokens.empty()
                                ? 0
                                : (req->prompt_tokens.back() + 1) % vocab_size_;
    req->generated_tokens.push_back(sampled_token);

    bool is_final = (sampled_token == eos_token_id_) ||
                    (req->generated_tokens.size() >=
                     static_cast<size_t>(req->max_new_tokens));

    generated_outputs.emplace_back(req->id, sampled_token, is_final);

    if (is_final) {
      req->state = scheduler::RequestState::FINISHED;
      finished_ids.push_back(req->id);
    }
  }

  // 4. Execute Decode Step
  for (auto &req : step_result.decode_requests) {
    int32_t last_token =
        req->generated_tokens.empty() ? 0 : req->generated_tokens.back();
    int32_t sampled_token = (last_token + 1) % vocab_size_;
    req->generated_tokens.push_back(sampled_token);

    bool is_final = (sampled_token == eos_token_id_) ||
                    (req->generated_tokens.size() >=
                     static_cast<size_t>(req->max_new_tokens));

    generated_outputs.emplace_back(req->id, sampled_token, is_final);

    if (is_final) {
      req->state = scheduler::RequestState::FINISHED;
      finished_ids.push_back(req->id);
    }
  }

  // 5. Emit streamed tokens token-by-token into thread-safe output queue
  if (!generated_outputs.empty()) {
    stream_buffer_->push_batch(generated_outputs);
  }

  // 6. Update scheduler state for sequences reaching termination
  if (!finished_ids.empty()) {
    batcher_->update_requests_state(finished_ids);
  }

  return total_active;
}

} // namespace cudaforge::engine