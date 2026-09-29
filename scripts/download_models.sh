#!/usr/bin/env bash
# Download the **local models** supermem uses, in one directory per purpose:
#
#   models/
#     vad/        silero_vad.onnx                                  end-of-speech detection
#     asr/        funasr-paraformer-zh-streaming                   streaming ASR (default, more accurate for Chinese)
#                 sherpa-onnx-streaming-zipformer-bilingual-zh-en... streaming ASR (fallback, pure onnx, no torch)
#     speaker/    3dspeaker_speech_eres2net_base_sv_zh-cn...onnx   speaker verification
#     embedding/  intfloat/multilingual-e5-small                   memory vectors + slot classification (shared)
#     scene/      MIT/ast-finetuned-audioset-10-10-0.4593          acoustic scene
#     emotion/    FunAudioLLM/SenseVoiceSmall                      emotion + refined transcription
#     tts/        (optional) piper voice .onnx, only needed for TTS_BACKEND=local
#
# With these downloaded the whole pipeline needs no network (except the reply model's API call). It also runs
# without them -- the code falls back to HF ids and transformers fetches them on first use.
#
# Two things not covered here:
#   - Reply model -- a PEFT adapter (180MB) on top of Qwen/Qwen3.6-35B-A3B; get the base separately.
#     Pass --reply-adapter to fetch it, see below.
#   - Qwen2.5-Omni for emotion attribution -- no fine-tuned version is published yet; the code defaults to the official
#     Qwen/Qwen2.5-Omni-3B and fetches it on first use. If you have your own fine-tune,
#     export SUPERMEM_OMNI_MODEL=/your/path to point at it.
#
# Usage (from the repo root):
#   bash scripts/download_models.sh                  # fetch the models/ set
#   bash scripts/download_models.sh /path/to/models  # use a different target directory
#   bash scripts/download_models.sh --reply-adapter  # also fetch the reply model adapter
#   SUPERMEM_FROM_UPSTREAM=1 bash scripts/download_models.sh   # fetch each model from its official source instead
set -euo pipefail

DEST="models"
WANT_SLM=0
for arg in "$@"; do
  case "$arg" in
    --reply-adapter|--slm) WANT_SLM=1 ;;   # --slm is the old name, kept so old commands still work
    *)     DEST="$arg" ;;
  esac
done

REPO="${SUPERMEM_MODELS_REPO:-zhifeixie/VoiceMem_Default_Models_Env}"
# Note: the repo name says Qwen25_omni, but it contains the Qwen3.6-35B reply adapter.
ADAPTER_REPO="${SUPERMEM_REPLY_ADAPTER_REPO:-${SUPERMEM_SLM_REPO:-zhifeixie/VoiceMem_SLM_Qwen25_omni}}"
mkdir -p "${DEST}"

if [ "${SUPERMEM_FROM_UPSTREAM:-0}" != "1" ]; then
  echo "[1/2] Fetching all local models from ${REPO} ..."
  python3 - "${REPO}" "${DEST}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])   # resumable; re-running does not re-download
PY
else
  # Fetch each from its official public source (when HF is unreachable, or to verify sources and licenses)
  REL="https://github.com/k2-fsa/sherpa-onnx/releases/download"
  ASR_DIR="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
  mkdir -p "${DEST}"/{vad,asr,speaker,embedding,scene,emotion}

  echo "[1/2] Official source: VAD silero (MIT) ..."
  curl -L -o "${DEST}/vad/silero_vad.onnx" "${REL}/asr-models/silero_vad.onnx"

  echo "      Official source: fallback streaming ASR ${ASR_DIR} (Apache-2.0, k2-fsa) ..."
  [ -d "${DEST}/asr/${ASR_DIR}" ] || curl -L "${REL}/asr-models/${ASR_DIR}.tar.bz2" | tar xj -C "${DEST}/asr"

  # Note: the official release tag really is spelled "recongition"
  echo "      Official source: speaker verification 3D-Speaker ERes2Net (Apache-2.0) ..."
  curl -L -o "${DEST}/speaker/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
    "${REL}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

  echo "      Official source: embedding / scene / emotion (each from its original HF repo) ..."
  python3 - "${DEST}" <<'PY'
import sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
# Fetch only the weights that actually get loaded. These repos also carry safetensors / pytorch_model.bin /
# various onnx quantizations / openvino; the whole repos are 4.2G, the needed part about 1.4G -- a big saving
# both for the download and for uploading to the release repo afterwards.
SKIP = ["*.bin", "onnx/*", "openvino/*", "*.tflite", "*.h5", "*.msgpack", "coreml/*"]
for kind, repo, skip in [# Default streaming ASR. Without it funasr downloads 848M only when the user speaks the first sentence
                         ("asr/funasr-paraformer-zh-streaming",
                          "funasr/paraformer-zh-streaming", ["example/*", "fig/*"]),
                         ("embedding", "intfloat/multilingual-e5-small", SKIP),
                         ("scene",     "MIT/ast-finetuned-audioset-10-10-0.4593", SKIP),
                         # SenseVoice weights are model.pt, so the *.bin exclusion set cannot be used
                         ("emotion",   "FunAudioLLM/SenseVoiceSmall", ["*.onnx", "*.tflite"])]:
    print(f"        {kind} <- {repo}")
    snapshot_download(repo_id=repo, local_dir=f"{dest}/{kind}", ignore_patterns=skip)
PY
fi

if [ "${WANT_SLM}" = "1" ]; then
  echo "[2/2] Reply model adapter ${ADAPTER_REPO} ..."
  python3 - "${ADAPTER_REPO}" "${DEST}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=f"{sys.argv[2]}/reply_adapter")
PY
  echo "      To use it: export SUPERMEM_REPLY_ADAPTER=${DEST}/reply_adapter"
  echo "      This is an adapter, not a full model -- get the base Qwen/Qwen3.6-35B-A3B separately, under its own license."
  echo "      Run: python examples/03_simple_agent_with_supermem_memory.py"
else
  echo "[2/2] Skipping reply model adapter (pass --reply-adapter to fetch it)"
fi

echo
echo "Done: ${DEST}/ is organised by purpose (vad / asr / speaker / embedding / scene / emotion)."
echo "The code prefers these local models and falls back to HF downloads for anything missing."
