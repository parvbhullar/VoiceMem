"""Long-running worker for 3D-Speaker ERes2Net voiceprint embeddings.

The supermem main program (in another conda env without funasr/modelscope) launches
this script via subprocess and fetches voiceprint vectors across environments using a
"one line of path -> one line of JSON" protocol:

    Request (stdin, one per line):   /path/to/audio.wav
    Response (stdout, one per line): {"embedding": [0.01, -0.02, ...]}   # 192-dim
                                     or {"error": "..."}

After startup it first prints a line "READY", which the caller uses to confirm the model has loaded.
All non-protocol output (download progress bars, version-check notices, etc.) must go to stderr and never pollute stdout.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def main() -> None:
    import soundfile as sf
    import sherpa_onnx
    from scipy.signal import resample_poly

    # Keep the old device argument for compatibility; sherpa-onnx CPU inference is fast enough.
    _device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    # This is a standalone subprocess (possibly running in another conda env), so it does not import supermem;
    # the path resolution logic mirrors supermem/utils/common/paths.py:
    #   SUPERMEM_SPEAKER_MODEL > <models dir>/speaker/<name> > <models dir>/<name>
    # The speaker/ subdirectory is the layout of the released model repo; the flat one is
    # where earlier downloaders have it. Accept both so a layout change doesn't suddenly break lookup.
    # Note: the old code used parents[2], which is supermem/utils/ and pointed at a nonexistent directory --
    # without SUPERMEM_SPEAKER_MODEL set, voiceprint couldn't find the model out of the box. The repo root is parents[4].
    #
    # parents[4] is the repo root only when **running from the repo**; after pip install it is site-packages,
    # which has no models/ under it. So, like paths.py:19, fall back to models/ under cwd -- otherwise users of the
    # installed package, even with models/ right at hand, only get "campplus_worker failed to start: ''"
    # (stderr was swallowed by DEVNULL, so the reason wasn't even visible).
    from pathlib import Path

    _NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    model_path = os.environ.get("SUPERMEM_SPEAKER_MODEL")
    if not model_path:
        _root = os.environ.get("SUPERMEM_MODELS_DIR")
        _repo = Path(__file__).resolve().parents[4] / "models"
        _dir = Path(_root) if _root else (_repo if _repo.is_dir() else Path("models"))
        _grouped = _dir / "speaker" / _NAME
        model_path = str(_grouped if _grouped.exists() else _dir / _NAME)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Voiceprint model {_NAME} not found: {model_path}\n"
            f"Download: bash scripts/download_models.sh models\n"
            f"or point SUPERMEM_SPEAKER_MODEL / SUPERMEM_MODELS_DIR at an existing location."
        )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=2)
    )
    window = int(float(os.environ.get("SUPERMEM_SPEAKER_WINDOW", "3.0")) * 16000)
    hop = int(float(os.environ.get("SUPERMEM_SPEAKER_HOP", "1.5")) * 16000)

    print("READY", flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        if path == "__exit__":
            break
        try:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sample_rate != 16000:
                audio = resample_poly(audio, 16000, int(sample_rate))
            audio = np.asarray(audio, dtype=np.float32)
            # A single vector over the whole clip is easily skewed by leading noise / trailing silence; split into short windows,
            # filter out silence with an RMS gate, then average the valid windows.
            if len(audio) <= window:
                chunks = [audio]
            else:
                chunks = [audio[i:i + window] for i in range(0, len(audio) - window + 1, hop)]
            embeddings = []
            for chunk in chunks:
                if len(chunk) < 16000 or float(np.sqrt(np.mean(chunk * chunk))) < 0.005:
                    continue
                stream = extractor.create_stream()
                stream.accept_waveform(16000, chunk)
                stream.input_finished()
                if extractor.is_ready(stream):
                    embeddings.append(np.asarray(extractor.compute(stream), dtype=np.float32))
            if not embeddings:
                raise RuntimeError("speaker embedding did not reach the minimum speech length")
            vec = np.mean(embeddings, axis=0)
            vec /= np.linalg.norm(vec) + 1e-8
            vec = vec.tolist()
            print(json.dumps({"embedding": vec}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
