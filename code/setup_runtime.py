from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


CODE_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = CODE_DIR / "inference"
XCODEC_DIR = INFERENCE_DIR / "xcodec_mini_infer"

XCODEC_REPO = "m-a-p/xcodec_mini_infer"
STAGE1_REPO = "m-a-p/YuE-s1-7B-anneal-en-cot"
STAGE2_REPO = "m-a-p/YuE-s2-1B-general"

REQUIRED_XCODEC_PATHS = [
    XCODEC_DIR / "final_ckpt" / "config.yaml",
    XCODEC_DIR / "final_ckpt" / "ckpt_00360000.pth",
    XCODEC_DIR / "decoders" / "config.yaml",
    XCODEC_DIR / "decoders" / "decoder_131000.pth",
    XCODEC_DIR / "decoders" / "decoder_151000.pth",
    XCODEC_DIR / "models" / "soundstream_hubert_new.py",
    XCODEC_DIR / "vocoder.py",
    XCODEC_DIR / "post_process_audio.py",
]

MODEL_REQUIREMENTS = {
    STAGE1_REPO: [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ],
    STAGE2_REPO: [
        "config.json",
        "model.safetensors",
    ],
}


def verify_xcodec() -> tuple[bool, list[Path]]:
    missing = [path for path in REQUIRED_XCODEC_PATHS if not path.exists()]
    return not missing, missing


def verify_model(repo_id: str) -> tuple[bool, list[str]]:
    required = MODEL_REQUIREMENTS[repo_id]
    try:
        snapshot = Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    except Exception:
        return False, required.copy()
    missing = [name for name in required if not (snapshot / name).exists()]
    return not missing, missing


def verify_models() -> tuple[bool, dict[str, list[str]]]:
    failures: dict[str, list[str]] = {}
    for repo_id in (STAGE1_REPO, STAGE2_REPO):
        ready, missing = verify_model(repo_id)
        if not ready:
            failures[repo_id] = missing
    return not failures, failures


def install_xcodec(force: bool = False) -> None:
    ready, missing = verify_xcodec()
    if ready and not force:
        print("[BACH Studio] XCodec runtime is already complete.")
        return

    if missing:
        print("[BACH Studio] Missing XCodec runtime files:")
        for path in missing:
            print(f"  - {path.relative_to(CODE_DIR.parent)}")

    print(f"[BACH Studio] Downloading official runtime: {XCODEC_REPO}")
    XCODEC_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=XCODEC_REPO,
        local_dir=str(XCODEC_DIR),
        ignore_patterns=["**/__pycache__/**", "*.pyc"],
        force_download=force,
    )

    ready, missing = verify_xcodec()
    if not ready:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            "XCodec download completed but required files are still missing:\n" + formatted
        )
    print("[BACH Studio] XCodec runtime verified.")


def cache_generation_models(force: bool = False) -> None:
    for repo_id in (STAGE1_REPO, STAGE2_REPO):
        ready, _ = verify_model(repo_id)
        if ready and not force:
            print(f"[BACH Studio] Model already complete: {repo_id}")
            continue
        print(f"[BACH Studio] Downloading/repairing model: {repo_id}")
        snapshot_download(repo_id=repo_id, force_download=force)
        ready, missing = verify_model(repo_id)
        if not ready:
            raise RuntimeError(
                f"Model snapshot verification failed for {repo_id}: " + ", ".join(missing)
            )
    print("[BACH Studio] Stage 1 and Stage 2 model snapshots are verified.")


def setup_runtime(full: bool = True, force: bool = False) -> str:
    install_xcodec(force=force)
    if full:
        cache_generation_models(force=force)

    xcodec_ready, xcodec_missing = verify_xcodec()
    if not xcodec_ready:
        raise RuntimeError("XCodec verification failed: " + ", ".join(map(str, xcodec_missing)))

    if full:
        models_ready, model_failures = verify_models()
        if not models_ready:
            details = "; ".join(
                f"{repo}: {', '.join(missing)}" for repo, missing in model_failures.items()
            )
            raise RuntimeError("Generation model verification failed: " + details)

    return "Runtime setup complete. XCodec is verified" + (
        " and Stage 1/Stage 2 model snapshots are verified." if full else "."
    )


def print_verification(full: bool) -> bool:
    ready, missing = verify_xcodec()
    if ready:
        print("[BACH Studio] XCodec runtime verification passed.")
    else:
        print("[BACH Studio] XCodec runtime verification failed.")
        for path in missing:
            print(f"  - {path}")

    all_ready = ready
    if full:
        models_ready, failures = verify_models()
        if models_ready:
            print("[BACH Studio] Stage 1 and Stage 2 model verification passed.")
        else:
            print("[BACH Studio] Generation model verification failed.")
            for repo_id, names in failures.items():
                print(f"  {repo_id}:")
                for name in names:
                    print(f"    - {name}")
        all_ready = all_ready and models_ready
    return all_ready


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify the BACH Studio/YuE runtime.")
    parser.add_argument(
        "--xcodec-only",
        action="store_true",
        help="Only install/repair XCodec; Stage models can still download automatically during generation.",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download of runtime assets.")
    parser.add_argument("--verify", action="store_true", help="Verify installed runtime/model files without downloading.")
    args = parser.parse_args()

    try:
        full = not args.xcodec_only
        if args.verify:
            return 0 if print_verification(full=full) else 2

        print(setup_runtime(full=full, force=args.force))
        return 0
    except KeyboardInterrupt:
        print("\n[BACH Studio] Setup cancelled.")
        return 130
    except Exception as exc:
        print(f"\n[BACH Studio] Runtime setup failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
