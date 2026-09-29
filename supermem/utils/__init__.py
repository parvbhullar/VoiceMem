"""supermem.utils -- the capability/tooling layer supporting the left and right brain.

- ``audio/``    native audio perception: ASR, speaker voiceprints, emotion VAD, acoustic environment/scene.
- ``common/``   cross-cutting tools: session tracking, turn_id, voice input adapters, shared graph helpers, cost logging, config.
- ``fusion/``   fusion of left/right-brain outputs and reply orchestration (an upper orchestration layer independent of the main engine).

These are "how to perceive, how to coordinate" tools, not the memory itself -- the memory's left/right
brain live in ``supermem.leftbrain`` / ``supermem.rightbrain``. Submodules are imported lazily on demand (see the
PEP 562 mapping in the top-level ``supermem/__init__.py``), so ``import supermem`` does not pull in torch/sherpa.
"""
