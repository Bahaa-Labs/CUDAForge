set -euo pipefail

OUTPUT_DIR="profiling/reports"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/nsys_report_${TIMESTAMP}"

echo "[CUDAForge] Launching Nsight Systems Timeline Trace..."

nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --cuda-memory-usage=true \
  --sample=cpu \
  --backtrace=dwarf \
  --output="${REPORT_FILE}" \
  --force-overwrite=true \
  python3 benchmarks/suite/bench_continuous_batch.py --num-requests 100

echo "[CUDAForge] NSYS Timeline Report generated: ${REPORT_FILE}.nsys-rep"