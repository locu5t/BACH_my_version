# BACH Studio

BACH Studio is a local Gradio interface for the repository's current `code/inference/infer.py` pipeline.

## Features

- Structured lyrics editor with section validation.
- Searchable Genre, Mood, Instrument, Voice/Gender, and Vocal Timbre controls populated from `code/top_200_tags.json`.
- Automatic conversion of selected tags into the flat style prompt expected by inference.
- No-reference, single-audio-reference, and dual vocal/instrumental-reference modes.
- Advanced controls for seed, GPU index, section count, repetition penalty, max tokens, Stage 2 batch size, model offloading, and output rescaling.
- Mixed output, vocal stem, and instrumental stem playback when produced by inference.
- Per-run logs and unique output folders.
- System/preflight page that detects missing dependencies and checkpoint files before generation.

## Windows launch

1. Activate the Python environment intended for BACH.
2. Install the repository dependencies:

   `pip install -r code/requirements.txt`

3. Double-click `run_ui.bat`.

The launcher starts `code/ui.py` and opens the Gradio UI in the default browser.

## Current repository limitation

The checked-in `code/inference/infer.py` imports and references an `xcodec_mini_infer` runtime tree containing codec configs, codec checkpoints, decoder weights, `models/soundstream_hubert_new.py`, `vocoder.py`, and `post_process_audio.py`. Those files are not currently committed to this fork.

BACH Studio therefore performs a preflight check and blocks generation with an explicit missing-component report until those runtime assets are installed under `code/inference/xcodec_mini_infer/`.

## Compatibility launcher

The UI runs `code/inference/infer_ui.py`, which keeps the original `infer.py` unchanged while applying two narrow compatibility fixes:

- FlashAttention 2 automatically falls back to SDPA/eager attention when FlashAttention is unavailable.
- The UI compensates for the current segment-slicing boundary so the user's final real lyric section is included in generation.

This wrapper approach makes it easier to sync future upstream changes without maintaining a large forked copy of `infer.py`.
