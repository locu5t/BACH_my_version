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
- Automatic installation of the official XCodec runtime and pre-caching of the Stage 1/Stage 2 generation models.

## Windows launch

Double-click `run_ui.bat` from the repository root.

The launcher performs the complete setup before opening the UI:

1. Installs/updates the Python dependencies from `code/requirements.txt`.
2. Verifies CUDA-enabled PyTorch. If Windows has an NVIDIA driver but the current PyTorch build cannot use CUDA, it installs a CUDA-enabled `torch`/`torchaudio` build from the official PyTorch wheel index.
3. Runs `code/setup_runtime.py`, which downloads and verifies the official `m-a-p/xcodec_mini_infer` runtime under `code/inference/xcodec_mini_infer/`.
4. Pre-caches `m-a-p/YuE-s1-7B-anneal-en-cot` and `m-a-p/YuE-s2-1B-general`, matching the model identifiers used by the checked-in inference script.
5. Starts BACH Studio and opens it in the default browser.

Interrupted Hugging Face downloads are resumable: run `run_ui.bat` again.

## Runtime assets

The XCodec tree and model weights are intentionally not committed to Git because they are multi-gigabyte model/runtime assets. They are no longer a manual prerequisite: BACH Studio provisions them automatically from the authoritative Hugging Face repositories.

`code/setup_runtime.py` supports:

- `python setup_runtime.py` — install/verify XCodec and cache both generation models.
- `python setup_runtime.py --xcodec-only` — install/repair only XCodec.
- `python setup_runtime.py --verify` — verify the local XCodec runtime without downloading.
- `python setup_runtime.py --force` — force a fresh runtime/model download.

The **System** tab also provides **Download / Repair Complete Runtime** if an installed asset is removed or damaged later.

## Windows inference compatibility

The UI runs `code/inference/infer_ui.py`, which keeps the original `infer.py` intact on disk and applies narrow runtime compatibility fixes:

- On Windows, the external FlashAttention-2 request is replaced by PyTorch SDPA, avoiding a fragile native `flash-attn` build dependency. Other platforms attempt FlashAttention-2 and fall back to SDPA/eager attention if necessary.
- `torch.compile` is skipped on Windows for broader native-Windows compatibility.
- The upstream lyric-section count boundary is corrected at runtime so the final requested lyric section is not silently omitted.

## Outputs

Each run receives a unique folder under `code/output_ui/`. BACH Studio exposes the final mix, vocal stem, instrumental stem, and generation log when produced by inference.
