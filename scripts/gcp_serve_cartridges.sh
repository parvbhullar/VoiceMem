#!/usr/bin/env bash
# Serve the two engines the cartridge benchmark and demo compare, on ONE GPU.
#
#   :8001  baseline  vLLM, prefix caching OFF   -> every turn is a full prefill
#   :8002  cartridge vLLM + LMCache (CPU tier)  -> cartridge KV is reused
#
# Same model, same GPU, same vLLM build; the only difference is KV reuse.
# The benchmark sends one request at a time, so the two processes never
# compete for the GPU during a measurement.
#
# Usage (on the GCP GPU VM):
#   bash scripts/gcp_serve_cartridges.sh            # start both, wait until healthy
#   bash scripts/gcp_serve_cartridges.sh stop
#   BLEND=1 bash scripts/gcp_serve_cartridges.sh    # also :8003 CacheBlend (experimental)
#
# One-time VM + install (check image names / zones for your project):
#   gcloud compute instances create unpod-cartridges --zone=asia-south1-a \
#     --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
#     --maintenance-policy=TERMINATE --boot-disk-size=200GB \
#     --image-project=deeplearning-platform-release --image-family=common-cu124-ubuntu-2204-py310 \
#     --metadata=install-nvidia-driver=True
#   pip install vllm lmcache          # pin the versions you benchmark with; they go on the slide
#   pip install -e .                  # this repo (for the benchmark + demo)
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.42}"          # two engines share one GPU
LOG_DIR="${LOG_DIR:-results/cartridges/engine-logs}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "stop" ]]; then
  for port in 8001 8002 8003; do
    pid=$(lsof -t -i ":$port" 2>/dev/null || true)
    [[ -n "$pid" ]] && kill $pid && echo "stopped :$port"
  done
  exit 0
fi

COMMON=(--max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL"
        --enable-prompt-tokens-details --disable-log-requests)

echo "baseline  :8001  $MODEL  (prefix caching OFF)"
nohup vllm serve "$MODEL" --port 8001 --no-enable-prefix-caching "${COMMON[@]}" \
  > "$LOG_DIR/baseline.log" 2>&1 &

echo "cartridge :8002  $MODEL  (prefix caching ON + LMCache CPU offload)"
LMCACHE_CONFIG_FILE="$HERE/lmcache/cpu_offload.yaml" \
nohup vllm serve "$MODEL" --port 8002 "${COMMON[@]}" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  > "$LOG_DIR/cartridge.log" 2>&1 &

PORTS=(8001 8002)
if [[ "${BLEND:-0}" == "1" ]]; then
  echo "blend     :8003  $MODEL  (LMCache CacheBlend, EXPERIMENTAL)"
  LMCACHE_CONFIG_FILE="$HERE/lmcache/blend.yaml" \
  nohup vllm serve "$MODEL" --port 8003 "${COMMON[@]}" --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    > "$LOG_DIR/blend.log" 2>&1 &
  PORTS+=(8003)
fi

for port in "${PORTS[@]}"; do
  printf "waiting for :%s " "$port"
  until curl -sf "http://localhost:$port/health" >/dev/null; do printf "."; sleep 5; done
  echo " up ($(curl -s "http://localhost:$port/version"))"
done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

cat <<MSG

Engines ready. Next:
  python evaluation/cartridges/run_bench.py --model $MODEL \\
      --full-url http://localhost:8001/v1 --cartridge-url http://localhost:8002/v1 \\
      --turns 100 --gpu-label "\$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  python web/cartridge_demo.py --model $MODEL \\
      --full-url http://localhost:8001/v1 --cartridge-url http://localhost:8002/v1 --space demo
MSG
