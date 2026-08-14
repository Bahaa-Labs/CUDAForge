#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <chrono>
#include <string>

namespace cudaforge::scheduler {

enum class RequestState {
    WAITING,
    RUNNING,
    FINISHED,
    PREEMPTED
};

/*
    Representation of an individual sequence generation request.
 */
struct Request {
    uint64_t id{0};
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    int32_t max_new_tokens{128};
    int32_t priority{0}; // Higher integer = higher execution priority
    
    RequestState state{RequestState::WAITING};
    std::chrono::steady_clock::time_point arrival_time;

    Request(uint64_t req_id, std::vector<int32_t> prompt, int32_t max_tokens, int32_t prio = 0)
        : id(req_id),
          prompt_tokens(std::move(prompt)),
          max_new_tokens(max_tokens),
          priority(prio),
          state(RequestState::WAITING),
          arrival_time(std::chrono::steady_clock::now()) {}

    size_t get_total_length() const {
        return prompt_tokens.size() + generated_tokens.size();
    }

    bool is_finished() const {
        return state == RequestState::FINISHED || generated_tokens.size() >= static_cast<size_t>(max_new_tokens);
    }
};

/*
    Priority Comparator: Highest priority first, then First-Come-First-Served (FCFS).
 */
struct RequestPriorityComparator {
    bool operator()(const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) const {
        if (a->priority != b->priority) {
            return a->priority < b->priority; // Higher priority integer wins
        }
        return a->arrival_time > b->arrival_time; // Earlier arrival time wins
    }
};

/*
    Output container for a single iteration step execution batch.
 */
struct BatchStepResult {
    std::vector<std::shared_ptr<Request>> prefill_requests;
    std::vector<std::shared_ptr<Request>> decode_requests;
    size_t total_batched_tokens{0};
    size_t preempted_count{0};
};

/*
    High-Concurrency Continuous Batcher Scheduler with Backpressure Control.
 */
class ContinuousBatcher {
public:
    ContinuousBatcher(size_t max_num_seqs, size_t max_num_batched_tokens);
    ~ContinuousBatcher() = default;

    // Disallow copy operations
    ContinuousBatcher(const ContinuousBatcher&) = delete;
    ContinuousBatcher& operator=(const ContinuousBatcher&) = delete;

    /*
        Submits a new generation request into the waiting queue.
     */
    bool add_request(std::shared_ptr<Request> request);

    /*
        Cancels an active or pending request by ID.
     */
    bool cancel_request(uint64_t request_id);

    /*
        Schedules requests for the current iteration step under token budget constraints.
     */
    BatchStepResult schedule_step();

    /*
        Notifies scheduler of finished requests and updates state.
     */
    void update_requests_state(const std::vector<uint64_t>& finished_ids);

    size_t get_pending_count() const;
    size_t get_running_count() const;

private:
    size_t max_num_seqs_{0};
    size_t max_num_batched_tokens_{0};

    mutable std::mutex mutex_;
    
    // Priority queue for pending/waiting requests
    std::priority_queue<std::shared_ptr<Request>, std::vector<std::shared_ptr<Request>>, RequestPriorityComparator> waiting_queue_;
    
    // Active running requests indexed by request ID
    std::vector<std::shared_ptr<Request>> running_requests_;
    std::unordered_map<uint64_t, std::shared_ptr<Request>> request_map_;

    void preempt_requests(size_t required_tokens, BatchStepResult& result);
};

} // namespace cudaforge::scheduler