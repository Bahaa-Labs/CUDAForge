import torch

try:
    from . import _C
except ImportError as e:
    raise ImportError(
        "Failed to load cudaforge C++ extension module '_C'. "
        "Run `python setup.py build_ext --inplace` to build."
    ) from e

# 1. Bind C++ extension objects first
ContinuousBatcher = _C.ContinuousBatcher
Request = _C.Request
RequestState = _C.RequestState
BatchStepResult = _C.BatchStepResult
StreamOutput = _C.StreamOutput
TokenStreamBuffer = _C.TokenStreamBuffer
ModelRunner = _C.ModelRunner
SpeculativeConfig = _C.SpeculativeConfig
SpeculativeVerificationResult = _C.SpeculativeVerificationResult
SpeculativeEngine = _C.SpeculativeEngine

# 2. Import Python submodules (engine, quantization, profiler)
from cudaforge.engine import (
    AsyncLLMEngine,
    EngineConfig,
    GenerationOutput,
    LLMEngine,
    SamplingParams,
    ptr_to_torch_tensor,
)

from cudaforge.quantization import (
    NumericalDriftReport,
    QuantizationCalibrator,
    Quantizer,
    QuantMode,
    evaluate_numerical_drift,
)

from cudaforge.profiler import (
    CUDAEventProfiler,
    CUDATimerReport,
    NCUMetrics,
    NsightComputeProfiler,
    nvtx_range,
)

# 3. Import kernels LAST to avoid circular import loops
try:
    from cudaforge import kernels
except ImportError:
    kernels = None

__version__ = "0.1.0"

__all__ = [
    "_C",
    "ContinuousBatcher",
    "Request",
    "RequestState",
    "BatchStepResult",
    "StreamOutput",
    "TokenStreamBuffer",
    "ModelRunner",
    "SpeculativeConfig",
    "SpeculativeVerificationResult",
    "SpeculativeEngine",
    "LLMEngine",
    "AsyncLLMEngine",
    "EngineConfig",
    "SamplingParams",
    "GenerationOutput",
    "ptr_to_torch_tensor",
    "QuantMode",
    "QuantizationCalibrator",
    "Quantizer",
    "evaluate_numerical_drift",
    "NumericalDriftReport",
    "NsightComputeProfiler",
    "CUDAEventProfiler",
    "NCUMetrics",
    "CUDATimerReport",
    "nvtx_range",
    "kernels",
    "__version__",
]