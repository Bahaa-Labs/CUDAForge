set -euo pipefail

OUTPUT_DIR="profiling/reports"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/ncu_report_${TIMESTAMP}"

echo "[CUDAForge] Launching Nsight Compute Kernel Profiler..."

ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-regex "flash_attn|fused_rms|int4_gemm|rope" \
  --set full \
  --import-source yes \
  --clock-control base \
  --export "${REPORT_FILE}" \
  python3 benchmarks/suite/bench_flash_attn.py

echo "[CUDAForge] NCU Report exported to: ${REPORT_FILE}.ncu-rep"