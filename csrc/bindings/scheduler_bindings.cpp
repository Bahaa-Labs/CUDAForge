#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cudaforge/scheduler/continuous_batcher.h"

namespace py = pybind11;
using namespace cudaforge::scheduler;

PYBIND11_MODULE(_C, m) {
    m.doc() = "cudaforge C++/CUDA high-performance engine python extension";

    // RequestState Enum
    py::enum_<RequestState>(m, "RequestState")
        .value("WAITING", RequestState::WAITING)
        .value("RUNNING", RequestState::RUNNING)
        .value("FINISHED", RequestState::FINISHED)
        .value("PREEMPTED", RequestState::PREEMPTED)
        .export_values();

    // Request Class
    py::class_<Request, std::shared_ptr<Request>>(m, "Request")
        .def(py::init<uint64_t, std::vector<int32_t>, int32_t, int32_t>(),
             py::arg("req_id"),
             py::arg("prompt"),
             py::arg("max_tokens") = 128,
             py::arg("priority") = 0)
        .def_readwrite("id", &Request::id)
        .def_readwrite("prompt_tokens", &Request::prompt_tokens)
        .def_readwrite("generated_tokens", &Request::generated_tokens)
        .def_readwrite("max_new_tokens", &Request::max_new_tokens)
        .def_readwrite("priority", &Request::priority)
        .def_readwrite("state", &Request::state)
        .def("get_total_length", &Request::get_total_length)
        .def("is_finished", &Request::is_finished);

    // BatchStepResult Struct
    py::class_<BatchStepResult>(m, "BatchStepResult")
        .def(py::init<>())
        .def_readwrite("prefill_requests", &BatchStepResult::prefill_requests)
        .def_readwrite("decode_requests", &BatchStepResult::decode_requests)
        .def_readwrite("total_batched_tokens", &BatchStepResult::total_batched_tokens)
        .def_readwrite("preempted_count", &BatchStepResult::preempted_count);

    // ContinuousBatcher Engine
    py::class_<ContinuousBatcher>(m, "ContinuousBatcher")
        .def(py::init<size_t, size_t>(),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"))
        .def("add_request", &ContinuousBatcher::add_request, py::arg("request"))
        .def("cancel_request", &ContinuousBatcher::cancel_request, py::arg("request_id"))
        .def("schedule_step", &ContinuousBatcher::schedule_step)
        .def("update_requests_state", &ContinuousBatcher::update_requests_state, py::arg("finished_ids"))
        .def("get_pending_count", &ContinuousBatcher::get_pending_count)
        .def("get_running_count", &ContinuousBatcher::get_running_count);
}