# BACH Studio

BACH Studio is a local Gradio interface for the repository's current `code/inference/infer.py` music-generation pipeline.

## Features

- Structured lyrics editor with section validation.
- Searchable Genre, Mood, Instrument, Voice/Gender, and Vocal Timbre controls populated from `code/top_200_tags.json`.
- Automatic conversion of selected tags into the flat style prompt expected by inference.
- No-reference, single-audio-reference, and dual vocal/instrumental-reference modes.
- Advanced controls for seed, GPU index, section count, repetition penalty, max tokens, Stage 2 batch size, model offloading, and output rescaling.
- Mixed output, vocal stem, and instrumental stem playback when produced by inference.
- Per-run logs and unique output folders.
- System/preflight page with one-click runtime repair.
- Automatic installation and verification of the official XCodec runtime and Stage 1/Stage 2 generation models.

## Windows launch

Double-click `run_ui.bat` from the repository root.

BACH Studio does not install into your global Python environment. The launcher creates and reuses a repository-local `.venv`. It prefers Python 3.10 when the Windows `py` launcher is available, then falls back through Python 3.11, 3.12, or 3.13.

Before opening the UI, the launcher performs the complete setup:

1. Creates/reuses `.venv` and updates its `pip`.
2. Installs and verifies `torch==2.7.1` and `torchaudio==2.7.1` from PyTorch's official CUDA 12.8 Windows wheel index. CUDA availability is checked in a fresh Python process before setup continues.
3. Installs the remaining dependencies from `code/requirements.txt` without replacing the verified Torch/TorchAudio pair.
4. Runs `code/setup_runtime.py`, which downloads and verifies the official `m-a-p/xcodec_mini_infer` runtime under `code/inference/xcodec_mini_infer/`, then downloads and verifies the Stage 1 and Stage 2 generation model snapshots.
5. Starts BACH Studio and opens it in the default browser only after setup succeeds.

If an install or model download is interrupted, run `run_ui.bat` again; Hugging Face downloads are resumable and already-complete files are verified/reused.

## Runtime assets

The XCodec tree and model weights are intentionally not committed to Git because they are multi-gigabyte runtime/model assets. They are not a manual prerequisite: BACH Studio provisions them automatically from their authoritative Hugging Face repositories.

The exact generation assets are:

- `m-a-p/xcodec_mini_infer`
- `m-a-p/YuE-s1-7B-anneal-en-cot`
- `m-a-p/YuE-s2-1B-general`

`code/setup_runtime.py` verifies the XCodec configuration/checkpoint/decoder/module files as well as every required Stage 1 model shard and the Stage 2 model file. A partially downloaded cache therefore does not count as ready.

Runtime maintenance commands:

- `python setup_runtime.py` — install/repair XCodec and both generation models.
- `python setup_runtime.py --xcodec-only` — install/repair only XCodec.
- `python setup_runtime.py --verify` — verify installed XCodec and generation model files without downloading.
- `python setup_runtime.py --force` — force fresh runtime/model downloads.

The **System** tab also provides **Download / Repair Complete Runtime** if an installed asset is removed or damaged later. Pressing **Generate Song** also performs a runtime verification/repair before inference, so direct `python ui.py` launches remain self-healing.

## Windows inference compatibility

The UI runs `code/inference/infer_ui.py`, which keeps the original `infer.py` intact on disk and applies narrow runtime compatibility fixes:

- On Windows, the external FlashAttention-2 request is replaced by PyTorch SDPA, avoiding a fragile native `flash-attn` build dependency. Other platforms attempt FlashAttention-2 and fall back to SDPA/eager attention if necessary.
- `torch.compile` is skipped on Windows for broader native-Windows compatibility.
- The upstream lyric-section count boundary is corrected at runtime so the final requested lyric section is not silently omitted.

## System readiness

The System tab checks:

- supported Python version
- UI compatibility launcher and upstream inference script
- required Python packages
- CUDA/GPU availability
- every required XCodec runtime/checkpoint/decoder file
- complete Stage 1 model snapshot
- complete Stage 2 model snapshot

The UI reports **Ready to generate** only when all required checks pass.

## Outputs

Each run receives a unique folder under `code/output_ui/`. BACH Studio exposes the final mix, vocal stem, instrumental stem, and generation log when produced by inference.
