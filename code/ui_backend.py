from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from setup_runtime import install_xcodec, verify_xcodec


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
INFERENCE_DIR = CODE_DIR / "inference"
INFER_SCRIPT = INFERENCE_DIR / "infer_ui.py"
UPSTREAM_INFER_SCRIPT = INFERENCE_DIR / "infer.py"
TAGS_PATH = CODE_DIR / "top_200_tags.json"
OUTPUT_ROOT = CODE_DIR / "output_ui"
INPUT_ROOT = CODE_DIR / ".ui_inputs"

REQUIRED_INFERENCE_PATHS = [
    INFERENCE_DIR / "xcodec_mini_infer" / "final_ckpt" / "config.yaml",
    INFERENCE_DIR / "xcodec_mini_infer" / "final_ckpt" / "ckpt_00360000.pth",
    INFERENCE_DIR / "xcodec_mini_infer" / "decoders" / "config.yaml",
    INFERENCE_DIR / "xcodec_mini_infer" / "decoders" / "decoder_131000.pth",
    INFERENCE_DIR / "xcodec_mini_infer" / "decoders" / "decoder_151000.pth",
    INFERENCE_DIR / "xcodec_mini_infer" / "models" / "soundstream_hubert_new.py",
    INFERENCE_DIR / "xcodec_mini_infer" / "vocoder.py",
    INFERENCE_DIR / "xcodec_mini_infer" / "post_process_audio.py",
]

SECTION_RE = re.compile(r"^\s*\[([A-Za-z][A-Za-z0-9 _-]*)\]\s*$", re.MULTILINE)


@dataclass
class GenerationResult:
    mixed: str | None
    vocal: str | None
    instrumental: str | None
    output_dir: str
    log: str
    command: str


def load_tags() -> dict[str, list[str]]:
    if not TAGS_PATH.exists():
        return {"genre": [], "instrument": [], "mood": [], "gender": [], "timbre": []}
    with TAGS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned: dict[str, list[str]] = {}
    for key in ("genre", "instrument", "mood", "gender", "timbre"):
        values = data.get(key, [])
        seen = set()
        result = []
        for value in values:
            value = str(value).strip()
            folded = value.casefold()
            if value and folded not in seen:
                seen.add(folded)
                result.append(value)
        cleaned[key] = sorted(result, key=str.casefold)
    return cleaned


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value if str(v).strip()]


def compose_style(
    genres: Iterable[str] | str | None,
    instruments: Iterable[str] | str | None,
    moods: Iterable[str] | str | None,
    gender: str | None,
    timbres: Iterable[str] | str | None,
    extra: str | None = None,
) -> str:
    parts: list[str] = []
    for group in (
        _as_list(moods),
        _as_list([gender] if gender else []),
        _as_list(genres),
        _as_list(timbres),
        _as_list(instruments),
    ):
        parts.extend(group)
    if extra and extra.strip():
        parts.append(extra.strip())

    seen = set()
    deduped = []
    for part in parts:
        folded = part.casefold().strip()
        if folded and folded not in seen:
            seen.add(folded)
            deduped.append(part.strip())
    return " ".join(deduped)


def validate_lyrics(lyrics: str) -> tuple[bool, str]:
    text = (lyrics or "").strip()
    if not text:
        return False, "Lyrics are empty. Add at least one section such as [verse]."
    sections = SECTION_RE.findall(text)
    if not sections:
        return False, "No section headers found. Use headers such as [verse], [chorus], [bridge], or [outro]."
    if len(sections) < 2:
        return True, f"1 section detected: {sections[0]}. Generation is allowed, but multiple sections generally work better."
    return True, f"{len(sections)} sections detected: " + ", ".join(sections)


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def system_check() -> dict[str, Any]:
    checks: list[dict[str, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"component": name, "status": "OK" if ok else "MISSING", "detail": detail})

    add("Python", sys.version_info >= (3, 10), sys.version.split()[0])
    add("UI inference launcher", INFER_SCRIPT.exists(), str(INFER_SCRIPT))
    add("Upstream inference script", UPSTREAM_INFER_SCRIPT.exists(), str(UPSTREAM_INFER_SCRIPT))
    add("Tag registry", TAGS_PATH.exists(), str(TAGS_PATH))
    add("Gradio", _module_available("gradio"), "Python package")
    add("Hugging Face Hub", _module_available("huggingface_hub"), "Python package")
    add("PyTorch", _module_available("torch"), "Python package")
    add("Torchaudio", _module_available("torchaudio"), "Python package")
    add("Transformers", _module_available("transformers"), "Python package")
    add("SoundFile", _module_available("soundfile"), "Python package")

    cuda_detail = "Unavailable"
    cuda_ok = False
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            cuda_detail = f"{torch.cuda.device_count()} GPU(s): " + ", ".join(names)
    except Exception as exc:
        cuda_detail = str(exc)
    add("CUDA", cuda_ok, cuda_detail)

    for path in REQUIRED_INFERENCE_PATHS:
        add(path.name, path.exists(), str(path.relative_to(REPO_ROOT)))

    missing = [c for c in checks if c["status"] != "OK"]
    return {
        "ready": not missing,
        "checks": checks,
        "missing_count": len(missing),
    }


def system_check_markdown() -> str:
    report = system_check()
    lines = ["### System check", "", "| Component | Status | Detail |", "|---|---|---|"]
    for item in report["checks"]:
        icon = "✅" if item["status"] == "OK" else "❌"
        detail = item["detail"].replace("|", "\\|")
        lines.append(f"| {item['component']} | {icon} {item['status']} | `{detail}` |")
    lines.append("")
    if report["ready"]:
        lines.append("**Ready to generate.**")
    else:
        lines.append(
            f"**Not ready:** {report['missing_count']} required check(s) are unresolved. "
            "Use Download / Repair Complete Runtime in the System tab, or re-run run_ui.bat."
        )
    return "\n".join(lines)


def _resolve_upload_path(upload: Any) -> str | None:
    if upload is None:
        return None
    if isinstance(upload, str):
        return upload
    if isinstance(upload, Path):
        return str(upload)
    for attr in ("name", "path"):
        value = getattr(upload, attr, None)
        if value:
            return str(value)
    return str(upload)


def _new_job_dir() -> tuple[Path, Path]:
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    input_dir = INPUT_ROOT / job_id
    output_dir = OUTPUT_ROOT / job_id
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def build_command(
    genre_path: Path,
    lyrics_path: Path,
    output_dir: Path,
    reference_mode: str,
    audio_reference: Any,
    vocal_reference: Any,
    instrumental_reference: Any,
    prompt_start: float,
    prompt_end: float,
    seed: int,
    cuda_idx: int,
    run_n_segments: int,
    repetition_penalty: float,
    max_new_tokens: int,
    stage2_batch_size: int,
    keep_intermediate: bool,
    disable_offload: bool,
    rescale: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(INFER_SCRIPT),
        "--genre_txt",
        str(genre_path),
        "--lyrics_txt",
        str(lyrics_path),
        "--output_dir",
        str(output_dir),
        "--seed",
        str(int(seed)),
        "--cuda_idx",
        str(int(cuda_idx)),
        "--run_n_segments",
        str(int(run_n_segments)),
        "--repetition_penalty",
        str(float(repetition_penalty)),
        "--max_new_tokens",
        str(int(max_new_tokens)),
        "--stage2_batch_size",
        str(int(stage2_batch_size)),
    ]

    mode = (reference_mode or "None").lower()
    if mode.startswith("single"):
        path = _resolve_upload_path(audio_reference)
        if not path:
            raise ValueError("Single Audio Reference mode requires an audio file.")
        cmd += [
            "--use_audio_prompt",
            "--audio_prompt_path",
            path,
            "--prompt_start_time",
            str(float(prompt_start)),
            "--prompt_end_time",
            str(float(prompt_end)),
        ]
    elif mode.startswith("dual"):
        vocal = _resolve_upload_path(vocal_reference)
        instrumental = _resolve_upload_path(instrumental_reference)
        if not vocal or not instrumental:
            raise ValueError("Dual Reference mode requires both vocal and instrumental files.")
        cmd += [
            "--use_dual_tracks_prompt",
            "--vocal_track_prompt_path",
            vocal,
            "--instrumental_track_prompt_path",
            instrumental,
            "--prompt_start_time",
            str(float(prompt_start)),
            "--prompt_end_time",
            str(float(prompt_end)),
        ]

    if keep_intermediate:
        cmd.append("--keep_intermediate")
    if disable_offload:
        cmd.append("--disable_offload_model")
    if rescale:
        cmd.append("--rescale")
    return cmd


def _latest_matching(paths: list[Path]) -> str | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return str(max(existing, key=lambda p: p.stat().st_mtime))


def discover_outputs(output_dir: Path) -> tuple[str | None, str | None, str | None]:
    all_audio = list(output_dir.rglob("*.mp3")) + list(output_dir.rglob("*.wav"))
    mixed_candidates = [p for p in all_audio if "mix" in p.parts or "mixed" in p.name.lower()]
    vocal_candidates = [p for p in all_audio if "vtrack" in p.name.lower() or "vocal" in p.name.lower()]
    inst_candidates = [p for p in all_audio if "itrack" in p.name.lower() or "instrument" in p.name.lower()]

    root_audio = [p for p in all_audio if p.parent == output_dir]
    mixed = _latest_matching(root_audio) or _latest_matching(mixed_candidates)
    vocal = _latest_matching(vocal_candidates)
    instrumental = _latest_matching(inst_candidates)
    return mixed, vocal, instrumental


def generate_song(
    lyrics: str,
    genres: Any,
    instruments: Any,
    moods: Any,
    gender: str | None,
    timbres: Any,
    extra_style: str,
    reference_mode: str,
    audio_reference: Any,
    vocal_reference: Any,
    instrumental_reference: Any,
    prompt_start: float,
    prompt_end: float,
    seed: int,
    cuda_idx: int,
    run_n_segments: int,
    repetition_penalty: float,
    max_new_tokens: int,
    stage2_batch_size: int,
    keep_intermediate: bool,
    disable_offload: bool,
    rescale: bool,
) -> GenerationResult:
    ok, lyric_message = validate_lyrics(lyrics)
    if not ok:
        raise ValueError(lyric_message)
    if prompt_end <= prompt_start and (reference_mode or "None") != "None":
        raise ValueError("Reference end time must be greater than start time.")

    style = compose_style(genres, instruments, moods, gender, timbres, extra_style)
    if not style:
        raise ValueError("Choose at least one style tag or enter Extra style tags.")

    xcodec_ready, _ = verify_xcodec()
    if not xcodec_ready:
        install_xcodec(force=False)

    report = system_check()
    if not report["ready"]:
        missing = [c["detail"] for c in report["checks"] if c["status"] != "OK"]
        raise RuntimeError(
            "System preflight failed after automatic runtime repair. Missing/unavailable:\n- "
            + "\n- ".join(missing)
        )

    input_dir, output_dir = _new_job_dir()
    genre_path = input_dir / "genre.txt"
    lyrics_path = input_dir / "lyrics.txt"
    genre_path.write_text(style, encoding="utf-8")
    lyrics_path.write_text(lyrics.strip() + "\n", encoding="utf-8")

    cmd = build_command(
        genre_path,
        lyrics_path,
        output_dir,
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
    )

    process = subprocess.Popen(
        cmd,
        cwd=str(INFERENCE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
    return_code = process.wait()

    log_path = output_dir / "generation.log"
    command_string = subprocess.list2cmdline(cmd)
    log_text = "$ " + command_string + "\n\n" + "\n".join(lines)
    log_path.write_text(log_text, encoding="utf-8")

    if return_code != 0:
        tail = "\n".join(lines[-60:])
        raise RuntimeError(
            f"Generation failed with exit code {return_code}.\n\n{tail}\n\nFull log: {log_path}"
        )

    mixed, vocal, instrumental = discover_outputs(output_dir)
    return GenerationResult(
        mixed=mixed,
        vocal=vocal,
        instrumental=instrumental,
        output_dir=str(output_dir),
        log=log_text[-12000:],
        command=command_string,
    )


def open_output_folder(path: str) -> str:
    target = Path(path) if path else OUTPUT_ROOT
    target.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            opener = shutil.which("xdg-open")
            if opener:
                subprocess.Popen([opener, str(target)])
    except Exception:
        pass
    return str(target)
