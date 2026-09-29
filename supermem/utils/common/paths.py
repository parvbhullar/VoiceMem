"""The single place that resolves the local models directory.

Before this, three places each did their own thing: ``defaults.py`` used ``"models"`` (cwd-relative), ``stream_io.py`` used
``"../models"`` (assuming it runs from web/), ``campplus_worker.py`` counted up from ``__file__`` -- and
counted the wrong number of levels, landing on ``supermem/utils/models/`` (which does not exist). It is unified here.
"""
from __future__ import annotations

import os
from pathlib import Path


def models_dir() -> Path:
    """``SUPERMEM_MODELS_DIR`` > ``models/`` at the repo root > search upward from cwd for ``models/``.

    Search upward rather than only looking at cwd: after a pip install ``parents[3]`` is site-packages, which has no
    ``models/``, so only the cwd-relative path is left -- and if the user runs from a subdirectory of the project (e.g.
    ``cd tests && python test_streaming.py``) it is not found, and the error still says "model not downloaded"
    when the models are actually one level up.
    """
    env = os.environ.get("SUPERMEM_MODELS_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[3] / "models"   # utils/common/paths.py -> repo root
    if repo.is_dir():
        return repo
    here = Path.cwd().resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "models"
        if candidate.is_dir():
            return candidate
    return Path("models")          # none found: let require() report it in plain words


#: Legacy env var names for local models. The official name is derived from the capability name (see local_model_env); legacy names still work.
#: These names were once invented one per model, with no rule to remember -- ``SUPERMEM_SILERO_VAD`` becomes
#: misleading as soon as the VAD implementation changes, and so does ``SUPERMEM_SENSEVOICE_MODEL``.
_LOCAL_LEGACY = {
    "asr":        "SUPERMEM_SENSEVOICE_MODEL",
    "vad":        "SUPERMEM_SILERO_VAD",
    "voiceprint": "SUPERMEM_SPEAKER_MODEL",
    "scene":      "SUPERMEM_ENVIRONMENT_MODEL",
    "e5":         "SUPERMEM_E5_MODEL",
}


def local_model_env(cap: str) -> str:
    """Env var name for a local model, derived from the capability name: ``vad`` -> ``SUPERMEM_VAD_MODEL``.

    Same rule as for API models (``llm_config.env_name``). Local models are not merged into that
    role table because the two hold different kinds of values: there it is a model name (``gpt-4o-mini``), here
    it is a file path or HF repo id, tied to a specific implementation -- for the same embedding capability,
    the API path takes ``text-embedding-3-small`` and the local path takes ``intfloat/multilingual-e5-small``;
    one merged variable would only get filled in wrong. Same naming rule, separate meanings.
    """
    return f"SUPERMEM_{cap.upper()}_MODEL"


def local_model_override(cap: str) -> str:
    """The local model the user specified for this capability (new name first, legacy name still honoured); "" if none."""
    legacy = _LOCAL_LEGACY.get(cap, "")
    return (os.environ.get(local_model_env(cap), "").strip()
            or (os.environ.get(legacy, "").strip() if legacy else ""))


def model_path(name: str, cap: str | None = None, kind: str = "") -> Path:
    """Path to a specific model file; the env var for ``cap`` has the highest priority.

    ``cap`` is the capability name (``vad`` / ``asr`` / ``voiceprint`` ...); its env var is derived from it.
    ``kind`` is the per-purpose subdirectory (``vad`` / ``asr`` / ``speaker``), matching the layout of the release repo
    the model bundle repo. If not found, fall back to the previous flat layout -- people who
    downloaded earlier should not suddenly lose their models because the organisation changed.
    """
    if cap:
        explicit = local_model_override(cap)
        if explicit:
            return Path(explicit)
    root = models_dir()
    if kind:
        grouped = root / kind / name
        if grouped.exists():
            return grouped
    return root / name          # legacy flat layout; require() reports it if it really does not exist


def hf_model(kind: str, default_id: str, cap: str | None = None) -> str:
    """Where an HF model should be loaded from: ``env`` > local offline bundle > HF repo id (auto-download).

    The offline bundle is ``<models>/<kind>/``, with the same layout as the model bundle repo
    (``embedding`` / ``scene`` / ``emotion`` ..., one folder per purpose). If not downloaded, return the
    HF id and transformers fetches it on first run as before -- **the zero-config default is unchanged**; people who
    downloaded the offline bundle no longer need network access anywhere in the pipeline.

    "This directory really contains a model" is judged by config.json or .onnx, not by whether the directory exists:
    an empty directory (e.g. left by an interrupted download) should still fall back to HF instead of the loader failing on it.
    """
    if cap:
        explicit = local_model_override(cap)
        if explicit:
            return explicit
    local = models_dir() / kind
    if (local / "config.json").exists() or any(local.glob("*.onnx")):
        return str(local)
    return default_id


def require(path: Path, what: str, how: str = "") -> Path:
    """If the model file is missing, say so in plain words instead of letting the underlying library throw a cryptic error.

    The hint has to address two kinds of people: those who cloned the repo just run scripts/download_models.sh;
    those who did ``pip install supermem`` have no scripts/ directory, so telling them to run it says nothing.
    Whether that script exists in the current directory decides which hint to give.
    """
    if not Path(path).exists():
        if not how:
            if Path("scripts/download_models.sh").is_file():
                how = "bash scripts/download_models.sh models"
            else:
                how = ("git clone https://github.com/xzf-thu/SuperMem && "
                       "bash SuperMem/scripts/download_models.sh models")
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            f"Download: {how}\n"
            f"or point SUPERMEM_MODELS_DIR at your existing models directory."
        )
    return Path(path)
