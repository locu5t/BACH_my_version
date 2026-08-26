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


def verify_xcodec() -> tuple[bool, list[Path]]:
    missing = [path for path in REQUIRED_XCODEC_PATHS if not path.exists()]
    return not missing, missing


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
        print(f"[BACH Studio] Ensuring model is available: {repo_id}")
        snapshot_download(repo_id=repo_id, force_download=force)
    print("[BACH Studio] Stage 1 and Stage 2 models are cached and ready.")


def setup_runtime(full: bool = True, force: bool = False) -> str:
    install_xcodec(force=force)
    if full:
        cache_generation_models(force=force)
    ready, missing = verify_xcodec()
    if not ready:
        raise RuntimeError("Runtime verification failed: " + ", ".join(map(str, missing)))
    return "Runtime setup complete. XCodec is verified" + (
        " and Stage 1/Stage 2 models are cached." if full else "."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify the BACH Studio/YuE runtime.")
    parser.add_argument(
        "--xcodec-only",
        action="store_true",
        help="Only install/repair the XCodec runtime. Stage models will download automatically during generation.",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download of runtime assets.")
    parser.add_argument("--verify", action="store_true", help="Verify XCodec files without downloading anything.")
    args = parser.parse_args()

    try:
        if args.verify:
            ready, missing = verify_xcodec()
            if ready:
                print("[BACH Studio] XCodec runtime verification passed.")
                return 0
            print("[BACH Studio] XCodec runtime verification failed.")
            for path in missing:
                print(f"  - {path}")
            return 2

        print(setup_runtime(full=not args.xcodec_only, force=args.force))
        return 0
    except KeyboardInterrupt:
        print("\n[BACH Studio] Setup cancelled.")
        return 130
    except Exception as exc:
        print(f"\n[BACH Studio] Runtime setup failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
