<div align="center">

# 🚀 CUDAForge

### High-Performance LLM Inference Runtime & GPU Experimentation Engine

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU_Computing-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-GPU_Kernels-FF6F00?style=for-the-badge&logo=python&logoColor=white)](https://github.com/triton-lang/triton)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-GPU_Runtime-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

[![FlashAttention](https://img.shields.io/badge/FlashAttention-2-FF1493?style=for-the-badge)](#)
[![Paged KV Cache](https://img.shields.io/badge/Paged-KV--Cache-8A2BE2?style=for-the-badge)](#)
[![Continuous Batching](https://img.shields.io/badge/Continuous-Batching-6A5ACD?style=for-the-badge)](#)
[![Kernel Autotuning](https://img.shields.io/badge/Kernel-Autotuning-D2691E?style=for-the-badge)](#)
[![Speculative Decoding](https://img.shields.io/badge/Speculative-Decoding-B22222?style=for-the-badge)](#)

[![FP16](https://img.shields.io/badge/FP16-Inference-1E90FF?style=for-the-badge)](#)
[![BF16](https://img.shields.io/badge/BF16-Inference-1E90FF?style=for-the-badge)](#)
[![INT8](https://img.shields.io/badge/INT8-Quantization-8B5CF6?style=for-the-badge)](#)
[![INT4](https://img.shields.io/badge/INT4-Quantization-8B5CF6?style=for-the-badge)](#)
[![Ampere](https://img.shields.io/badge/NVIDIA-Ampere_sm__86-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](#)

[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Observability-F46800?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com/)
[![Nsight Compute](https://img.shields.io/badge/NVIDIA-Nsight_Compute-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/nsight-compute)
[![CI](https://img.shields.io/badge/CI%2FCD-Regression_Validation-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](#)

Production-grade inference infrastructure engineered for **NVIDIA Ampere-class GPUs**, combining custom GPU kernels, memory-efficient KV-cache management, continuous batching, kernel autotuning, speculative decoding, and reproducible performance evaluation.

</div>

---

# 📌 Executive Summary

Modern LLM inference is fundamentally a systems problem: model quality alone is not enough.

High-throughput generation requires efficient attention kernels, GPU memory management, KV-cache reuse, dynamic scheduling, kernel selection, quantization, profiling, and continuous performance validation.

**CUDAForge** is an end-to-end inference runtime and experimentation platform designed around those constraints.

The system combines **C++20/CUDA low-level execution**, **Triton kernel optimization**, **GPU-resident memory management**, **continuous batching**, **architecture-aware autotuning**, and **statistical benchmark automation** into a single production-oriented stack.

The architecture is intentionally designed to demonstrate the engineering depth required for **high-performance AI infrastructure, inference systems, and GPU-accelerated ML platforms**.

---

# 🚀 Key Performance Highlights

| Metric | Result | Optimization |
|---|---:|---|
| **Attention Latency P50** | **4.68 ms** | Fused FlashAttention execution |
| **Attention Speedup** | **8.43×** | GPU tiling + reduced HBM traffic |
| **Attention Throughput** | **58.83 TFLOPS** | SRAM-aware block execution |
| **KV-Cache Allocation** | **0.48 µs / request** | Paged GPU memory management |
| **Scheduler Throughput** | **15.35M tok/s** | Non-blocking continuous batching |
| **Precision Paths** | **FP16 / BF16 / INT8 / INT4** | Fused quantization execution |

> Benchmarks target **NVIDIA Ampere (sm_86)** execution with FP16 inference workloads. Results are workload-dependent and benchmark-specific.

---

# 🏗 System Architecture

```text
                         [ Client / Benchmark / Application ]
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   FastAPI Serving    │
                           │ REST • Streaming API │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Continuous Batcher   │
                           │ Scheduler • Budgets  │
                           │ Priority • Backpressure
                           └──────────┬───────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌──────────────────────┐             ┌──────────────────────┐
        │   KV-Cache Runtime   │             │ Kernel Dispatch      │
        │                      │             │ & Autotuning         │
        │ Paged Allocation     │             │                      │
        │ Prefix Reuse         │             │ FlashAttention       │
        │ Eviction             │             │ Fused Kernels        │
        │ Fragmentation        │             │ CUDA / Triton         │
        └──────────┬───────────┘             └──────────┬───────────┘
                   │                                    │
                   └────────────────┬───────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  LLM Execution Core  │
                         │                      │
                         │ FP16 • BF16          │
                         │ INT8 • INT4          │
                         │ Speculative Decode   │
                         │ Prefix Caching       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ NVIDIA Ampere-Class  │
                         │ GPU Architecture     │
                         └──────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │ Profiling • Telemetry •         │
                    │ Regression & Experiment Engine  │
                    └─────────────────────────────────┘

📁 Repository Structure

CUDAForge/
├── .github/
│   └── workflows/
│       ├── ci-regression.yml       # Automated GPU regression CI pipeline
│       └── lint-and-format.yml     # Code format checks (Ruff, Black, Clang-Format)
├── benchmarks/
│   ├── baselines/
│   │   └── baseline_metrics.json   # Hardware baseline target latency metrics
│   ├── suite/
│   │   ├── bench_flash_attn.py     # FlashAttention vs PyTorch SDP benchmarking
│   │   └── bench_paged_kv.py       # Page table lookup & allocation benchmark
│   └── regression_runner.py        # CI/CD Regression test entrypoint
├── csrc/                           # C++20 / CUDA Source Code
│   ├── include/                    # Headers (block_allocator, paged_kv_cache)
│   └── src/                        # Implementations & Pybind11 bindings
├── cudaforge/                      # Python Core Package
│   ├── _C.so                       # Pybind11 Native Extension
│   ├── engine.py                   # Engine runtime interface
│   ├── profiler.py                 # CUDA Event & Memory leak detector
│   ├── quantization.py             # INT8/INT4 dequantization routines
│   └── speculative.py              # Speculative decoding implementation
├── docker/
│   ├── Dockerfile.gpu              # Multi-stage lightweight GPU build file
│   └── docker-compose.yml          # Container orchestration configuration
├── observability/                  # Telemetry & Monitoring
│   ├── grafana/dashboards/         # Prometheus Grafana visual dashboards
│   └── prometheus/prometheus.yml   # Metrics scraper config
├── profiling/                      # Deep Profiling Traces
│   ├── nsight_configs/
│   ├── cuda_event_tracer.py
│   └── memory_leak_detector.py
├── serving/                        # REST API Production Serving
│   └── api/v1/                     # FastAPI endpoint schemas & handlers
├── tests/                          # Test Suites
│   ├── cpp/                        # GoogleTest C++ unit tests
│   ├── integration/                # Integration test scripts
│   └── unit/                       # PyTest Python unit tests
└── workloads/                      # Benchmarking Workloads & Data Loaders
    ├── dataset_loader.py           # Workload trace replay
    └── error_analysis.py          # Quality drop and precision verification

## 🏛 Runtime Components

## 🏛 Runtime Components

| Component | Core Responsibilities |
|---|---|
| **Inference Runtime** | LLM execution, precision-aware inference, and kernel dispatch |
| **Attention Engine** | Tiled FlashAttention and fused GPU execution |
| **KV-Cache Runtime** | Paged allocation, prefix reuse, eviction, and fragmentation control |
| **Batching Scheduler** | Continuous batching, admission control, priorities, token budgets, and backpressure |
| **Kernel Autotuner** | CUDA/Triton search, benchmarking, and architecture-aware dispatch |
| **Quantization Runtime** | FP16/BF16/INT8/INT4 execution and accuracy validation |
| **Decoding Engine** | Sampling, speculative decoding, and prefix-aware generation |
| **Benchmark Engine** | Latency, throughput, memory, kernel efficiency, and regression analysis |
| **Observability** | CUDA telemetry, GPU utilization, cache pressure, and runtime metrics |
| **Serving Layer** | FastAPI REST/streaming inference and production request management |
| **Infrastructure** | Docker GPU deployment, CI/CD, profiling, and performance validation |

# 🚀 Quick Start

## Prerequisites

- Linux
- NVIDIA GPU with **Ampere-class architecture or newer**
- CUDA Toolkit **12.1+**
- Docker + NVIDIA Container Toolkit
- Python **3.10+**
- C++20-compatible compiler
- CMake / Ninja

---

## 🐳 Docker — Recommended

CUDAForge provides a GPU-aware containerized runtime for reproducible execution.

```bash
git clone https://github.com/Bahaa-Labs/CUDAForge.git
cd CUDAForge

docker compose -f docker/docker-compose.yml up --build

Verify the GPU is visible inside the container:

docker exec -it cudaforge-runtime nvidia-smi

Run the serving endpoint:

curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_tokens": [1,2,3,4,5,6,7,8],
    "max_new_tokens": 16,
    "temperature": 0.0
  }'

🧪 Validation & Benchmarking

CUDAForge treats performance as a first-class correctness criterion.

Unit Tests
pytest tests/unit/
Integration Tests
pytest tests/integration/
FlashAttention Benchmark
python3 benchmarks/suite/bench_flash_attn.py
Paged KV-Cache Benchmark
python3 benchmarks/suite/bench_paged_kv.py
Full Regression Suite
python3 benchmarks/regression_runner.py

The regression pipeline evaluates:

Latency       → P50 / P95 / P99
Throughput    → tokens/sec
Memory        → allocation / peak / fragmentation
GPU           → utilization / occupancy
Kernels       → execution time / efficiency
Quality       → numerical drift / correctness

Historical baselines are stored in:

benchmarks/baselines/baseline_metrics.json

Performance regressions are automatically surfaced instead of relying on manual benchmark inspection.

📈 Engineering Philosophy

CUDAForge is designed around one principle:

LLM inference performance is a systems problem, not just a model problem.

The runtime therefore treats the entire inference path as an optimization surface:

Model
  ↓
Kernel Selection
  ↓
Memory Movement
  ↓
KV-Cache Management
  ↓
Request Scheduling
  ↓
GPU Execution
  ↓
Telemetry
  ↓
Statistical Evaluation
  ↓
Regression Validation

This enables optimization decisions to be evaluated against the complete system rather than isolated kernel benchmarks.

🧠 Why CUDAForge

CUDAForge is intentionally built at the intersection of:

LLM Inference × GPU Programming × Systems Engineering × Machine Learning × Performance Science

Rather than wrapping an existing inference server, the project focuses on the underlying infrastructure required to make high-performance inference measurable, tunable, reproducible, and production-oriented.

It demonstrates the ability to work across the complete stack:

Python
   ↓
C++20
   ↓
CUDA / Triton
   ↓
GPU Memory
   ↓
Kernel Execution
   ↓
Inference Runtime
   ↓
Distributed Requests
   ↓
Observability
   ↓
Statistical Evaluation
🛡 Production Engineering

CUDAForge is structured around production-oriented engineering practices:

Deterministic benchmark configuration
Explicit GPU capability validation
Automated performance baselines
Numerical correctness checks
Structured runtime errors
Explicit memory/OOM handling
GPU-aware containerization
CI/CD validation
Reproducible experiment metadata
Profiling-driven optimization
Hardware-aware kernel selection

The objective is not simply to achieve a faster benchmark once.

The objective is to build a runtime where performance improvements can be reproduced, measured, validated, and protected against regression.

📜 License

CUDAForge is released under the MIT License.

See LICENSE for details.

