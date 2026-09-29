"""SpeakerEncoder — local 3D-Speaker voiceprint embedding extraction (optionally cross-environment).

The 3D-Speaker ERes2Net ONNX model is invoked through a long-running subprocess
(``supermem/voiceprint/campplus_worker.py``); the model is loaded once and kept
resident for reuse, avoiding the cost of reloading it on every call.

By default the subprocess is launched with the current interpreter (``sys.executable``), so it works
directly if funasr/modelscope are installed in the same environment. If you want to isolate it in a
separate environment (e.g. a dedicated conda env with funasr installed), set the environment variable
``SUPERMEM_AUDIOMEM_PYTHON`` to that environment's python executable.

Usage::

    enc = SpeakerEncoder()
    vec = enc.embed(Path("recording.wav"))  # np.ndarray [192]; None means failure
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from supermem.utils.audio.voiceprint import l2norm

# The worker sits right next to this module. Previously an extra "voiceprint/" level was joined in (left over from a
# refactor that moved directories), pointing at a nonexistent file: the subprocess exited immediately, readline got an
# empty string, and it was reported as "campplus_worker failed to start: ''" -- no reason in the error, voiceprint entirely unusable.
_WORKER_SCRIPT = Path(__file__).resolve().parent / "campplus_worker.py"
_WORKER_PYTHON = os.environ.get("SUPERMEM_AUDIOMEM_PYTHON", sys.executable)


class SpeakerEncoder:
    """Lazy-loading, thread-safe wrapper around the long-running 3D-Speaker ERes2Net worker subprocess."""

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        # Resolve the models dir in the parent process and pass it down. The worker is a standalone script and computes
        # its own copy, but that copy missed the post-pip-install fallback -- two copies of the logic will diverge sooner or later, so pin one here.
        from supermem.utils.common.paths import local_model_override, models_dir

        env = dict(os.environ)
        env.setdefault("SUPERMEM_MODELS_DIR", str(models_dir()))
        # The worker doesn't import supermem and only recognises the old SUPERMEM_SPEAKER_MODEL; the new name
        # (SUPERMEM_VOICEPRINT_MODEL) is resolved here and translated over.
        picked = local_model_override("voiceprint")
        if picked:
            env["SUPERMEM_SPEAKER_MODEL"] = picked

        # Capture stderr in a pipe, don't discard it. It used to be DEVNULL, so when the worker exited because it couldn't
        # find the model file, the outside only saw "campplus_worker failed to start: ''" -- no reason at all, even though the
        # worker had actually printed "model not found: <path>, download: bash scripts/download_models.sh".
        self._proc = subprocess.Popen(
            [_WORKER_PYTHON, str(_WORKER_SCRIPT), self._device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        ready = self._proc.stdout.readline().strip()
        if ready != "READY":
            why = ""
            if self._proc.poll() is not None:      # already exited, so stderr can be read to the end without blocking
                err = (self._proc.stderr.read() or "").strip().splitlines()
                # Keep only the last few lines -- the earlier ones are the traceback call stack, useless to the user
                why = "\n".join(err[-4:])
            raise RuntimeError("campplus_worker failed to start" + (f":\n{why}" if why else f": {ready!r}"))

    def embed(self, audio_path: Path) -> np.ndarray | None:
        """Return a 192-dim L2-normalised voiceprint vector, or None on failure."""
        try:
            with self._lock:
                self._ensure_started()
                self._proc.stdin.write(f"{audio_path}\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("campplus_worker gave no response (the process may have exited)")
            resp = json.loads(line)
            if "error" in resp:
                raise RuntimeError(resp["error"])

            vec = np.asarray(resp["embedding"], dtype=np.float64)
            return l2norm(vec)
        except Exception as e:
            print(f"  [speaker_encoder] embed failed: {e}", flush=True)
            return None
