from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

PYTHON_FILES = [
    CODE_DIR / "ui.py",
    CODE_DIR / "ui_backend.py",
    CODE_DIR / "setup_cuda.py",
    CODE_DIR / "setup_runtime.py",
    CODE_DIR / "inference" / "infer_ui.py",
]


class SmokeFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def read(path: Path) -> str:
    require(path.exists(), f"Missing required source file: {path.relative_to(REPO_ROOT)}")
    return path.read_text(encoding="utf-8")


def compile_sources() -> None:
    for path in PYTHON_FILES:
        source = read(path)
        compile(source, str(path), "exec")
        print(f"[PASS] Python syntax: {path.relative_to(REPO_ROOT)}")


def check_wiring() -> None:
    infer = read(CODE_DIR / "inference" / "infer.py")
    infer_ui = read(CODE_DIR / "inference" / "infer_ui.py")
    backend = read(CODE_DIR / "ui_backend.py")
    runtime = read(CODE_DIR / "setup_runtime.py")
    cuda = read(CODE_DIR / "setup_cuda.py")
    requirements = read(CODE_DIR / "requirements.txt")
    launcher = read(REPO_ROOT / "run_ui.bat")
    gitignore = read(REPO_ROOT / ".gitignore")

    old_boundary = "run_n_segments = min(args.run_n_segments+1, len(lyrics))"
    fixed_boundary = "run_n_segments = min(args.run_n_segments+1, len(lyrics)+1)"
    require(old_boundary in infer, "Upstream infer.py section boundary changed unexpectedly.")
    require(old_boundary in infer_ui, "infer_ui.py no longer checks the expected upstream boundary.")
    require(fixed_boundary in infer_ui, "infer_ui.py is missing the final-section boundary fix.")
    print("[PASS] Inference section compatibility patch is wired.")

    require('INFER_SCRIPT = INFERENCE_DIR / "infer_ui.py"' in backend, "UI backend is not routed through infer_ui.py.")
    require("setup_runtime(full=True, force=False)" in backend, "Generate path is not self-healing the runtime.")
    print("[PASS] UI backend uses compatibility inference and runtime repair.")

    for repo_id in (
        "m-a-p/xcodec_mini_infer",
        "m-a-p/YuE-s1-7B-anneal-en-cot",
        "m-a-p/YuE-s2-1B-general",
    ):
        require(repo_id in runtime, f"Runtime installer is missing authoritative repo ID: {repo_id}")
    for model_file in (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "model.safetensors",
        "ckpt_00360000.pth",
        "decoder_131000.pth",
        "decoder_151000.pth",
    ):
        require(model_file in runtime, f"Runtime verification is missing: {model_file}")
    print("[PASS] Runtime installer verifies XCodec and all generation model files.")

    require('TARGET_TORCH = "2.7.1"' in cuda, "Unexpected Torch target.")
    require('TARGET_TORCHAUDIO = "2.7.1"' in cuda, "Unexpected TorchAudio target.")
    require("https://download.pytorch.org/whl/cu128" in cuda, "CUDA 12.8 PyTorch index is not configured.")
    print("[PASS] Windows CUDA PyTorch bootstrap is pinned.")

    requirement_lines = {
        line.strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    require(not any(line.startswith("torch==") or line == "torch" for line in requirement_lines), "requirements.txt must not replace the CUDA Torch build.")
    require(not any(line.startswith("torchaudio==") or line == "torchaudio" for line in requirement_lines), "requirements.txt must not replace the CUDA TorchAudio build.")
    for expected in ("transformers>=4.48,<5", "gradio>=5.13,<6", "huggingface_hub>=0.28,<1"):
        require(expected in requirement_lines, f"Missing dependency constraint: {expected}")
    print("[PASS] Remaining dependency set preserves the verified CUDA Torch pair.")

    required_launcher_fragments = [
        ".venv\\Scripts\\python.exe",
        "code\\setup_cuda.py",
        "code\\requirements.txt",
        "code\\setup_runtime.py",
        "ui.py",
    ]
    for fragment in required_launcher_fragments:
        require(fragment in launcher, f"Windows launcher is missing step: {fragment}")
    require(
        launcher.index("code\\setup_cuda.py") < launcher.index("code\\requirements.txt"),
        "Windows launcher must install CUDA Torch before the remaining requirements.",
    )
    require(
        launcher.index("code\\requirements.txt") < launcher.index("code\\setup_runtime.py"),
        "Windows launcher must install dependencies before runtime/model setup.",
    )
    print("[PASS] Windows launcher setup order is correct.")

    for ignored in (".venv/", "code/inference/xcodec_mini_infer/", "code/output_ui/"):
        require(ignored in gitignore, f".gitignore does not exclude generated runtime path: {ignored}")
    print("[PASS] Generated environment/runtime/output paths are ignored by Git.")


def main() -> int:
    print("BACH Studio source smoke test")
    print("=" * 40)
    try:
        compile_sources()
        check_wiring()
    except SmokeFailure as exc:
        print(f"[FAIL] {exc}")
        return 1
    except Exception as exc:
        print(f"[FAIL] Unexpected smoke-test error: {exc}")
        return 1

    print("=" * 40)
    print("[PASS] BACH Studio source integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
