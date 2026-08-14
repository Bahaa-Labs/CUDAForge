#include <clocale>
#include <locale>
#include <memory>
#include <string>
#include <vector>
#include <span>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cudaforge/memory/block_allocator.h"
#include "cudaforge/memory/buffer_pool.h"
#include "cudaforge/memory/paged_kv_cache.h"
#include "cudaforge/scheduler/continuous_batcher.h"
#include "cudaforge/scheduler/speculative.h"
#include "cudaforge/engine/model_runner.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "cudaforge C++/CUDA high-performance engine python extension";

    // ------------------------------------------------------------------------
    // 1. Memory Management & KV Cache
    // ------------------------------------------------------------------------
    py::class_<cudaforge::memory::KVCacheConfig>(m, "KVCacheConfig")
        .def(py::init<>())
        .def_readwrite("num_layers", &cudaforge::memory::KVCacheConfig::num_layers)
        .def_readwrite("num_kv_heads", &cudaforge::memory::KVCacheConfig::num_kv_heads)
        .def_readwrite("head_dim", &cudaforge::memory::KVCacheConfig::head_dim)
        .def_readwrite("block_size", &cudaforge::memory::KVCacheConfig::block_size)
        .def_readwrite("element_size_bytes", &cudaforge::memory::KVCacheConfig::element_size_bytes)
        .def_readwrite("total_gpu_blocks", &cudaforge::memory::KVCacheConfig::total_gpu_blocks)
        .def_readwrite("enable_prefix_caching", &cudaforge::memory::KVCacheConfig::enable_prefix_caching)
        .def_readwrite("max_supported_batch_size", &cudaforge::memory::KVCacheConfig::max_supported_batch_size)
        .def_readwrite("max_blocks_per_sequence", &cudaforge::memory::KVCacheConfig::max_blocks_per_sequence);

    py::class_<cudaforge::memory::KVCacheStats>(m, "KVCacheStats")
        .def_readwrite("total_blocks", &cudaforge::memory::KVCacheStats::total_blocks)
        .def_readwrite("free_blocks", &cudaforge::memory::KVCacheStats::free_blocks)
        .def_readwrite("allocated_blocks", &cudaforge::memory::KVCacheStats::allocated_blocks)
        .def_readwrite("cached_prefix_blocks", &cudaforge::memory::KVCacheStats::cached_prefix_blocks)
        .def_readwrite("total_memory_bytes", &cudaforge::memory::KVCacheStats::total_memory_bytes)
        .def_readwrite("used_memory_bytes", &cudaforge::memory::KVCacheStats::used_memory_bytes)
        .def_readwrite("pool_utilization_ratio", &cudaforge::memory::KVCacheStats::pool_utilization_ratio)
        .def_readwrite("active_sequences_count", &cudaforge::memory::KVCacheStats::active_sequences_count)
        .def_readwrite("prefix_cache_hits", &cudaforge::memory::KVCacheStats::prefix_cache_hits)
        .def_readwrite("prefix_cache_misses", &cudaforge::memory::KVCacheStats::prefix_cache_misses)
        .def_readwrite("lru_evictions_count", &cudaforge::memory::KVCacheStats::lru_evictions_count);

    py::class_<cudaforge::memory::BlockAllocator, std::shared_ptr<cudaforge::memory::BlockAllocator>>(m, "BlockAllocator");

    py::class_<cudaforge::memory::PagedKVCache, std::shared_ptr<cudaforge::memory::PagedKVCache>>(m, "PagedKVCache")
        .def(py::init<const cudaforge::memory::KVCacheConfig&, cudaforge::memory::BlockAllocator&>(), 
             py::arg("config"), py::arg("allocator"))
        .def("register_sequence_with_prefix", [](cudaforge::memory::PagedKVCache& self, uint64_t seq_id, const std::vector<int32_t>& tokens) {
            return self.register_sequence_with_prefix(seq_id, std::span<const int32_t>(tokens.data(), tokens.size()));
        }, py::arg("sequence_id"), py::arg("prompt_tokens"))
        .def("append_tokens", &cudaforge::memory::PagedKVCache::append_tokens, py::arg("sequence_id"), py::arg("num_tokens") = 1)
        .def("unregister_sequence", &cudaforge::memory::PagedKVCache::unregister_sequence, py::arg("sequence_id"))
        .def("get_block_table", &cudaforge::memory::PagedKVCache::get_block_table, py::arg("sequence_id"))
        .def("get_config", &cudaforge::memory::PagedKVCache::get_config)
        .def("get_stats", &cudaforge::memory::PagedKVCache::get_stats);

    // ------------------------------------------------------------------------
    // 2. Scheduler Enums and Structs
    // ------------------------------------------------------------------------
    py::enum_<cudaforge::scheduler::RequestState>(m, "RequestState")
        .value("WAITING", cudaforge::scheduler::RequestState::WAITING)
        .value("RUNNING", cudaforge::scheduler::RequestState::RUNNING)
        .value("FINISHED", cudaforge::scheduler::RequestState::FINISHED)
        .value("PREEMPTED", cudaforge::scheduler::RequestState::PREEMPTED)
        .export_values();

    py::class_<cudaforge::scheduler::Request, std::shared_ptr<cudaforge::scheduler::Request>>(m, "Request")
        .def(py::init<uint64_t, std::vector<int32_t>, int32_t, int32_t>(),
             py::arg("req_id"),
             py::arg("prompt"),
             py::arg("max_tokens") = 128,
             py::arg("priority") = 0)
        .def_readwrite("id", &cudaforge::scheduler::Request::id)
        .def_readwrite("prompt_tokens", &cudaforge::scheduler::Request::prompt_tokens)
        .def_readwrite("generated_tokens", &cudaforge::scheduler::Request::generated_tokens)
        .def_readwrite("max_new_tokens", &cudaforge::scheduler::Request::max_new_tokens)
        .def_readwrite("priority", &cudaforge::scheduler::Request::priority)
        .def_readwrite("state", &cudaforge::scheduler::Request::state)
        .def("get_total_length", &cudaforge::scheduler::Request::get_total_length)
        .def("is_finished", &cudaforge::scheduler::Request::is_finished);

    py::class_<cudaforge::scheduler::BatchStepResult>(m, "BatchStepResult")
        .def(py::init<>())
        .def_readwrite("prefill_requests", &cudaforge::scheduler::BatchStepResult::prefill_requests)
        .def_readwrite("decode_requests", &cudaforge::scheduler::BatchStepResult::decode_requests)
        .def_readwrite("total_batched_tokens", &cudaforge::scheduler::BatchStepResult::total_batched_tokens)
        .def_readwrite("preempted_count", &cudaforge::scheduler::BatchStepResult::preempted_count);

    py::class_<cudaforge::scheduler::ContinuousBatcher, std::shared_ptr<cudaforge::scheduler::ContinuousBatcher>>(m, "ContinuousBatcher")
        .def(py::init<size_t, size_t>(),
             py::arg("max_num_seqs") = 256,
             py::arg("max_num_batched_tokens") = 8192)
        .def("add_request", &cudaforge::scheduler::ContinuousBatcher::add_request, py::arg("request"))
        .def("cancel_request", &cudaforge::scheduler::ContinuousBatcher::cancel_request, py::arg("request_id"))
        .def("schedule_step", &cudaforge::scheduler::ContinuousBatcher::schedule_step)
        .def("update_requests_state", &cudaforge::scheduler::ContinuousBatcher::update_requests_state, py::arg("finished_ids"))
        .def("get_pending_count", &cudaforge::scheduler::ContinuousBatcher::get_pending_count)
        .def("get_running_count", &cudaforge::scheduler::ContinuousBatcher::get_running_count);

    // ------------------------------------------------------------------------
    // 3. Streaming Engine & Model Runner
    // ------------------------------------------------------------------------
    py::class_<cudaforge::engine::StreamOutput>(m, "StreamOutput")
        .def(py::init<>())
        .def(py::init<uint64_t, int32_t, bool>(), py::arg("req_id"), py::arg("tok_id"), py::arg("final_flag"))
        .def_readwrite("request_id", &cudaforge::engine::StreamOutput::request_id)
        .def_readwrite("token_id", &cudaforge::engine::StreamOutput::token_id)
        .def_readwrite("is_final", &cudaforge::engine::StreamOutput::is_final);

    py::class_<cudaforge::engine::TokenStreamBuffer, std::shared_ptr<cudaforge::engine::TokenStreamBuffer>>(m, "TokenStreamBuffer")
        .def(py::init<>())
        .def("push", &cudaforge::engine::TokenStreamBuffer::push, py::arg("output"))
        .def("push_batch", &cudaforge::engine::TokenStreamBuffer::push_batch, py::arg("outputs"))
        .def("pop_all", &cudaforge::engine::TokenStreamBuffer::pop_all)
        .def("size", &cudaforge::engine::TokenStreamBuffer::size)
        .def("empty", &cudaforge::engine::TokenStreamBuffer::empty);

    py::class_<cudaforge::engine::ModelRunner, std::shared_ptr<cudaforge::engine::ModelRunner>>(m, "ModelRunner")
        .def(py::init<std::shared_ptr<cudaforge::scheduler::ContinuousBatcher>, size_t, int32_t>(),
             py::arg("batcher"), py::arg("vocab_size"), py::arg("eos_token_id"))
        .def("cancel_request", &cudaforge::engine::ModelRunner::cancel_request, py::arg("request_id"))
        .def("is_cancelled", &cudaforge::engine::ModelRunner::is_cancelled, py::arg("request_id"))
        .def("step", &cudaforge::engine::ModelRunner::step)
        .def("get_stream_buffer", &cudaforge::engine::ModelRunner::get_stream_buffer);

    // ------------------------------------------------------------------------
    // 4. Speculative Decoding
    // ------------------------------------------------------------------------
    py::class_<cudaforge::scheduler::SpeculativeConfig>(m, "SpeculativeConfig")
        .def(py::init<>())
        .def_readwrite("gamma", &cudaforge::scheduler::SpeculativeConfig::gamma)
        .def_readwrite("temperature", &cudaforge::scheduler::SpeculativeConfig::temperature)
        .def_readwrite("top_p", &cudaforge::scheduler::SpeculativeConfig::top_p)
        .def_readwrite("rng_seed", &cudaforge::scheduler::SpeculativeConfig::rng_seed);

    py::class_<cudaforge::scheduler::SpeculativeVerificationResult>(m, "SpeculativeVerificationResult")
        .def(py::init<>())
        .def_readwrite("accepted_tokens", &cudaforge::scheduler::SpeculativeVerificationResult::accepted_tokens)
        .def_readwrite("final_token", &cudaforge::scheduler::SpeculativeVerificationResult::final_token)
        .def_readwrite("num_accepted", &cudaforge::scheduler::SpeculativeVerificationResult::num_accepted)
        .def_readwrite("end_of_sequence", &cudaforge::scheduler::SpeculativeVerificationResult::end_of_sequence);

    py::class_<cudaforge::scheduler::SpeculativeEngine, std::shared_ptr<cudaforge::scheduler::SpeculativeEngine>>(m, "SpeculativeEngine")
        .def(py::init<const cudaforge::scheduler::SpeculativeConfig&>(), py::arg("config"))
        .def("verify_candidates", &cudaforge::scheduler::SpeculativeEngine::verify_candidates,
             py::arg("draft_tokens"), py::arg("draft_probs"), py::arg("target_probs"), py::arg("eos_token_id"))
        .def("verify_candidates_greedy", &cudaforge::scheduler::SpeculativeEngine::verify_candidates_greedy,
             py::arg("draft_tokens"), py::arg("target_probs"), py::arg("eos_token_id"))
        .def("set_config", &cudaforge::scheduler::SpeculativeEngine::set_config, py::arg("config"))
        .def("get_config", &cudaforge::scheduler::SpeculativeEngine::get_config);
}