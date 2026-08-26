from __future__ import annotations

import gradio as gr

from setup_runtime import setup_runtime
from ui_backend import (
    OUTPUT_ROOT,
    compose_style,
    generate_song,
    load_tags,
    open_output_folder,
    system_check_markdown,
    validate_lyrics,
)


tags = load_tags()


def style_preview(genres, instruments, moods, gender, timbres, extra):
    return compose_style(genres, instruments, moods, gender, timbres, extra)


def lyric_status(lyrics):
    ok, message = validate_lyrics(lyrics)
    prefix = "✅" if ok else "❌"
    return f"{prefix} {message}"


def run_generation(*args):
    try:
        result = generate_song(*args)
        status = f"✅ Generation complete. Output: `{result.output_dir}`"
        return result.mixed, result.vocal, result.instrumental, result.output_dir, result.log, status
    except Exception as exc:
        return None, None, None, "", str(exc), f"❌ {exc}"


def open_folder(path):
    target = open_output_folder(path)
    return f"Opened: `{target}`"


def repair_runtime():
    try:
        message = setup_runtime(full=True, force=False)
        return system_check_markdown(), f"✅ {message}"
    except Exception as exc:
        return system_check_markdown(), f"❌ Runtime setup failed: {exc}"


CSS = """
#bach-title {text-align:center; margin-bottom:0.15rem}
#bach-subtitle {text-align:center; opacity:0.78; margin-top:0}
.generate-button {min-height:52px; font-weight:700}
.status-box {min-height:42px}
"""

with gr.Blocks(title="BACH Studio", css=CSS) as demo:
    gr.Markdown("# 🎼 BACH Studio", elem_id="bach-title")
    gr.Markdown("Lyrics + style + optional reference audio, wrapped around the repository's existing inference pipeline.", elem_id="bach-subtitle")

    with gr.Tabs():
        with gr.Tab("Create"):
            with gr.Row():
                with gr.Column(scale=6):
                    lyrics = gr.Textbox(
                        label="Lyrics",
                        lines=22,
                        placeholder="[verse]\nWrite lyrics here...\n\n[chorus]\n...",
                    )
                    lyric_check = gr.Markdown("Add lyrics with section headers such as `[verse]` and `[chorus]`.")
                    lyrics.change(lyric_status, lyrics, lyric_check)

                with gr.Column(scale=5):
                    genres = gr.Dropdown(tags["genre"], label="Genre", multiselect=True, filterable=True)
                    moods = gr.Dropdown(tags["mood"], label="Mood", multiselect=True, filterable=True)
                    instruments = gr.Dropdown(tags["instrument"], label="Instruments", multiselect=True, filterable=True)
                    gender = gr.Dropdown(tags["gender"], label="Voice / Gender", filterable=True)
                    timbres = gr.Dropdown(tags["timbre"], label="Vocal Timbre", multiselect=True, filterable=True)
                    extra_style = gr.Textbox(
                        label="Extra style tags",
                        placeholder="Optional free-form tags, e.g. syncopated percussion warm bass",
                    )
                    style_string = gr.Textbox(label="Model style string", interactive=False)
                    for control in (genres, moods, instruments, gender, timbres, extra_style):
                        control.change(
                            style_preview,
                            [genres, instruments, moods, gender, timbres, extra_style],
                            style_string,
                        )

            gr.Markdown("### Reference audio")
            reference_mode = gr.Radio(
                ["None", "Single Audio Reference", "Dual Vocal + Instrumental Reference"],
                value="None",
                label="Reference mode",
            )
            with gr.Row():
                audio_reference = gr.Audio(label="Single reference", type="filepath")
                vocal_reference = gr.Audio(label="Vocal reference", type="filepath")
                instrumental_reference = gr.Audio(label="Instrumental reference", type="filepath")
            with gr.Row():
                prompt_start = gr.Number(label="Reference start (seconds)", value=0.0, minimum=0)
                prompt_end = gr.Number(label="Reference end (seconds)", value=30.0, minimum=0)

            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    cuda_idx = gr.Number(label="GPU index", value=0, precision=0, minimum=0)
                    run_n_segments = gr.Number(
                        label="Sections to generate",
                        value=99,
                        precision=0,
                        minimum=1,
                        info="Use a large number to process all lyric sections.",
                    )
                with gr.Row():
                    repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.01, label="Repetition penalty")
                    max_new_tokens = gr.Number(label="Max new tokens / section", value=3000, precision=0, minimum=100)
                    stage2_batch_size = gr.Number(label="Stage 2 batch size", value=4, precision=0, minimum=1)
                with gr.Row():
                    keep_intermediate = gr.Checkbox(label="Keep intermediate files", value=False)
                    disable_offload = gr.Checkbox(label="Keep Stage 1 model on GPU", value=False)
                    rescale = gr.Checkbox(label="Rescale output to reduce clipping", value=False)

            generate = gr.Button("Generate Song", variant="primary", elem_classes="generate-button")
            status = gr.Markdown("Ready.", elem_classes="status-box")

            gr.Markdown("### Result")
            with gr.Row():
                mixed_audio = gr.Audio(label="Mixed song")
                vocal_audio = gr.Audio(label="Vocal stem")
                instrumental_audio = gr.Audio(label="Instrumental stem")
            output_dir = gr.Textbox(label="Output folder", interactive=False)
            with gr.Row():
                open_output = gr.Button("Open Output Folder")
            generation_log = gr.Textbox(label="Generation log", lines=14, interactive=False)

            generate.click(
                run_generation,
                inputs=[
                    lyrics,
                    genres,
                    instruments,
                    moods,
                    gender,
                    timbres,
                    extra_style,
                    reference_mode,
                    audio_reference,
                    vocal_reference,
                    instrumental_reference,
                    prompt_start,
                    prompt_end,
                    seed,
                    cuda_idx,
                    run_n_segments,
                    repetition_penalty,
                    max_new_tokens,
                    stage2_batch_size,
                    keep_intermediate,
                    disable_offload,
                    rescale,
                ],
                outputs=[mixed_audio, vocal_audio, instrumental_audio, output_dir, generation_log, status],
            )
            open_output.click(open_folder, output_dir, status)

        with gr.Tab("System"):
            system_report = gr.Markdown(system_check_markdown())
            runtime_status = gr.Markdown(
                "`run_ui.bat` automatically installs and verifies the full runtime before opening the UI."
            )
            with gr.Row():
                refresh_system = gr.Button("Refresh System Check")
                repair_system = gr.Button("Download / Repair Complete Runtime", variant="primary")
            refresh_system.click(system_check_markdown, None, system_report)
            repair_system.click(repair_runtime, None, [system_report, runtime_status])
            gr.Markdown(
                "The repair action downloads the official `m-a-p/xcodec_mini_infer` runtime into "
                "`code/inference/xcodec_mini_infer` and ensures the Stage 1 and Stage 2 generation models are cached."
            )

        with gr.Tab("About"):
            gr.Markdown(
                """
### What this UI controls

BACH Studio exposes the current `code/inference/infer.py` arguments as a browser interface: lyrics, genre/style tags, optional single or dual-track audio references, seed, GPU, segment count, repetition penalty, Stage 2 batch size, model offloading, and rescaling.

The tag pickers are populated directly from `code/top_200_tags.json` and the selected values are converted into the flat genre/style string expected by the inference script.

### Runtime setup

`run_ui.bat` installs the Python requirements and runs `code/setup_runtime.py` before launching the UI. The setup script downloads the official XCodec runtime plus the Stage 1 and Stage 2 model snapshots required by the current inference code. Interrupted Hugging Face downloads can be resumed by running the launcher again.

### Outputs

Each run receives a unique folder under `code/output_ui/`. The UI attempts to locate and expose the final mix, vocal stem, and instrumental stem after inference finishes. A complete command/output log is also saved in the job folder.
                """
            )


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    demo.queue(default_concurrency_limit=1).launch(inbrowser=True, show_error=True)
