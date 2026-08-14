#include "cudaforge/scheduler/continuous_batcher.h"
#include <algorithm>
#include <iostream>

namespace cudaforge::scheduler {

ContinuousBatcher::ContinuousBatcher(size_t max_num_seqs,
                                     size_t max_num_batched_tokens)
    : max_num_seqs_(max_num_seqs),
      max_num_batched_tokens_(max_num_batched_tokens) {}

bool ContinuousBatcher::add_request(std::shared_ptr<Request> request) {
  if (!request || request->prompt_tokens.empty()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  if (request_map_.find(request->id) != request_map_.end()) {
    return false; // Request ID already exists
  }

  request->state = RequestState::WAITING;
  request_map_[request->id] = request;
  waiting_queue_.push(request);

  return true;
}

bool ContinuousBatcher::cancel_request(uint64_t request_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = request_map_.find(request_id);
  if (it == request_map_.end()) {
    return false;
  }

  auto req = it->second;
  req->state = RequestState::FINISHED;

  // Remove from running requests vector if active
  running_requests_.erase(
      std::remove_if(running_requests_.begin(), running_requests_.end(),
                     [request_id](const std::shared_ptr<Request> &r) {
                       return r->id == request_id;
                     }),
      running_requests_.end());

  request_map_.erase(it);
  return true;
}

BatchStepResult ContinuousBatcher::schedule_step() {
  std::lock_guard<std::mutex> lock(mutex_);

  BatchStepResult result;
  size_t allocated_tokens = 0;

  // 1. Filter out finished requests from running queue
  std::vector<std::shared_ptr<Request>> active_running;
  for (auto &req : running_requests_) {
    if (!req->is_finished() && req->state == RequestState::RUNNING) {
      active_running.push_back(req);
    } else {
      req->state = RequestState::FINISHED;
      request_map_.erase(req->id);
    }
  }
  running_requests_ = std::move(active_running);

  // 2. Budget Allocation for existing decode steps (1 token each)
  for (auto &req : running_requests_) {
    if (allocated_tokens + 1 <= max_num_batched_tokens_) {
      result.decode_requests.push_back(req);
      allocated_tokens += 1;
    } else {
      // Backpressure: Preempt low priority running request
      preempt_requests(1, result);
      break;
    }
  }

  // 3. Admit new prefill requests from priority queue
  std::vector<std::shared_ptr<Request>> deferred_waiting;

  while (!waiting_queue_.empty() && running_requests_.size() < max_num_seqs_) {
    auto candidate = waiting_queue_.top();
    waiting_queue_.pop();

    if (candidate->state == RequestState::FINISHED) {
      continue; // Skip cancelled requests
    }

    size_t prompt_len = candidate->prompt_tokens.size();

    // Check if token budget supports admitting this prefill request
    if (allocated_tokens + prompt_len <= max_num_batched_tokens_) {
      candidate->state = RequestState::RUNNING;
      result.prefill_requests.push_back(candidate);
      running_requests_.push_back(candidate);
      allocated_tokens += prompt_len;
    } else {
      // Cannot fit under current token budget, put back in waiting pool
      deferred_waiting.push_back(candidate);
      break;
    }
  }

  // Re-insert deferred requests back into priority queue
  for (auto &req : deferred_waiting) {
    waiting_queue_.push(req);
  }

  result.total_batched_tokens = allocated_tokens;
  return result;
}

void ContinuousBatcher::preempt_requests(size_t required_tokens,
                                         BatchStepResult &result) {
  // Sort running requests by lowest priority first for preemption
  std::sort(
      running_requests_.begin(), running_requests_.end(),
      [](const std::shared_ptr<Request> &a, const std::shared_ptr<Request> &b) {
        return a->priority < b->priority;
      });

  while (!running_requests_.empty() && required_tokens > 0) {
    auto req_to_preempt = running_requests_.front();
    running_requests_.erase(running_requests_.begin());

    req_to_preempt->state = RequestState::PREEMPTED;
    waiting_queue_.push(req_to_preempt);

    result.preempted_count++;
    break;
  }
}

void ContinuousBatcher::update_requests_state(
    const std::vector<uint64_t> &finished_ids) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (uint64_t id : finished_ids) {
    auto it = request_map_.find(id);
    if (it != request_map_.end()) {
      it->second->state = RequestState::FINISHED;
    }
  }
}

size_t ContinuousBatcher::get_pending_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return waiting_queue_.size();
}

size_t ContinuousBatcher::get_running_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return running_requests_.size();
}

} // namespace cudaforge::scheduler