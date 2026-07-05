"""Local orchestrator for remote Kaggle GPU training.

Runs on the developer/CI machine (NOT on Kaggle). It finalizes the kernel
metadata, pushes ``kaggle/train_kernel.py`` to Kaggle Kernels, polls until the
run finishes, and downloads the produced artifacts into ``MODELS_DIR``.

Only the ``kaggle`` CLI is used (via :mod:`subprocess`) -- the Python API objects
are intentionally avoided because the CLI's text contract is far more stable
across versions. All CLI output is parsed defensively.

Run as::

    python -m fraud_scoring.pipeline.run_kaggle_gpu

This module is wired as the ``train_gpu`` DVC stage.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fraud_scoring.config import (
    KAGGLE_DIR,
    MODELS_DIR,
    PARAMS,
    ensure_dirs,
    get_settings,
)

KERNEL_CFG = PARAMS["kaggle_kernel"]
KERNEL_NAME = KERNEL_CFG["kernel_name"]
POLL_INTERVAL_SEC = int(KERNEL_CFG.get("poll_interval_sec", 30))
POLL_TIMEOUT_SEC = int(KERNEL_CFG.get("poll_timeout_sec", 3600))

METADATA_PATH = KAGGLE_DIR / "kernel-metadata.json"
TEMPLATE_PATH = KAGGLE_DIR / "kernel-metadata.json"

# Artifacts we expect the kernel to emit into /kaggle/working.
EXPECTED_OUTPUTS = (
    "model_xgb.json",
    "model_torch.pt",
    "metrics.json",
    "feature_importance.json",
)

# Kaggle kernel statuses that mean "stop polling".
TERMINAL_STATUSES = {"complete", "error", "cancelacknowledged"}


# --------------------------------------------------------------------------- #
# Subprocess helper.
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a CLI command, capturing text output. Logs the command."""
    print(f"[cmd] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed (exit {proc.returncode}): {' '.join(cmd)}"
        )
    return proc


# --------------------------------------------------------------------------- #
# Username resolution.
# --------------------------------------------------------------------------- #
def resolve_username() -> str:
    """Resolve the Kaggle username.

    Precedence: ``KAGGLE_USERNAME`` env -> ``~/.kaggle/kaggle.json`` ->
    ``FRAUD_KAGGLE_USERNAME`` (Settings). Raises if none is found.
    """
    env_user = os.environ.get("KAGGLE_USERNAME")
    if env_user:
        return env_user.strip()

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text(encoding="utf-8"))
            user = data.get("username")
            if user:
                return str(user).strip()
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] could not parse {kaggle_json}: {exc!r}")

    settings_user = getattr(get_settings(), "kaggle_username", None)
    if settings_user:
        return str(settings_user).strip()

    raise RuntimeError(
        "Could not resolve Kaggle username. Set KAGGLE_USERNAME, provide "
        "~/.kaggle/kaggle.json, or set FRAUD_KAGGLE_USERNAME."
    )


# --------------------------------------------------------------------------- #
# Metadata.
# --------------------------------------------------------------------------- #
def build_metadata(username: str) -> dict:
    """Write the finalized ``kernel-metadata.json`` with the real id/title.

    Starts from the existing template file (falling back to sane defaults) and
    overwrites ``id`` and ``title`` for ``username``. Returns the written dict.
    """
    slug = f"{username}/{KERNEL_NAME}"

    if TEMPLATE_PATH.exists():
        meta = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    else:
        meta = {
            "code_file": "train_kernel.py",
            "language": "python",
            "kernel_type": "script",
            "dataset_sources": [PARAMS["data"]["kaggle_dataset"]],
            "competition_sources": [],
            "kernel_sources": [],
        }

    meta["id"] = slug
    meta["title"] = KERNEL_NAME
    meta["code_file"] = "train_kernel.py"
    meta["language"] = "python"
    meta["kernel_type"] = "script"
    meta["is_private"] = True
    meta["enable_gpu"] = bool(KERNEL_CFG.get("enable_gpu", True))
    meta["enable_internet"] = bool(KERNEL_CFG.get("enable_internet", True))
    meta.setdefault("dataset_sources", [PARAMS["data"]["kaggle_dataset"]])
    meta.setdefault("competition_sources", [])
    meta.setdefault("kernel_sources", [])

    METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[meta] wrote {METADATA_PATH} (id={slug})")
    return meta


# --------------------------------------------------------------------------- #
# Push / poll / pull.
# --------------------------------------------------------------------------- #
def push() -> None:
    """Push the kernel from ``KAGGLE_DIR`` to Kaggle."""
    print(f"[push] pushing kernel from {KAGGLE_DIR}")
    _run(["kaggle", "kernels", "push", "-p", str(KAGGLE_DIR)])


def _parse_status(text: str) -> str | None:
    """Extract a normalized status token from ``kaggle kernels status`` output.

    Example CLI line: ``"gulhanimsc/fraud-scoring-gpu-train has status "complete""``.
    Returns a lowercase status word, or ``None`` if it cannot be parsed.
    """
    lowered = text.lower()
    marker = "has status"
    if marker in lowered:
        tail = lowered.split(marker, 1)[1]
        # Strip quotes/punctuation and take the first token.
        token = tail.strip().strip('."\'` \t\r\n')
        token = token.replace('"', "").replace("'", "")
        token = token.split()[0] if token.split() else ""
        token = token.strip('."\'`')
        # The CLI reports an enum repr like ``KernelWorkerStatus.COMPLETE`` -- keep
        # only the final component so it matches TERMINAL_STATUSES ("complete", ...).
        token = token.rsplit(".", 1)[-1]
        if token:
            return token
    # Fallback: scan for any known status keyword.
    for known in (
        "complete",
        "error",
        "cancelacknowledged",
        "running",
        "queued",
    ):
        if known in lowered:
            return known
    return None


def poll(slug: str) -> str:
    """Poll kernel status until terminal or timeout. Returns final status."""
    print(
        f"[poll] polling {slug} every {POLL_INTERVAL_SEC}s "
        f"(timeout {POLL_TIMEOUT_SEC}s)"
    )
    deadline = time.monotonic() + POLL_TIMEOUT_SEC
    last_status: str | None = None

    while True:
        proc = _run(["kaggle", "kernels", "status", slug], check=False)
        status = _parse_status((proc.stdout or "") + "\n" + (proc.stderr or ""))

        if status and status != last_status:
            print(f"[poll] status: {last_status} -> {status}")
            last_status = status

        if status in TERMINAL_STATUSES:
            print(f"[poll] terminal status reached: {status}")
            return status

        if time.monotonic() >= deadline:
            print(f"[poll] timeout after {POLL_TIMEOUT_SEC}s (last={last_status})")
            return last_status or "timeout"

        time.sleep(POLL_INTERVAL_SEC)


def pull_outputs(slug: str) -> None:
    """Download kernel outputs into ``MODELS_DIR``."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[pull] downloading outputs for {slug} -> {MODELS_DIR}")
    _run(["kaggle", "kernels", "output", slug, "-p", str(MODELS_DIR)])

    missing = [f for f in EXPECTED_OUTPUTS if not (MODELS_DIR / f).exists()]
    if missing:
        print(f"[pull] WARNING: expected outputs not found: {missing}")
    else:
        print("[pull] all expected outputs present")


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def main() -> int:
    ensure_dirs()

    username = resolve_username()
    slug = f"{username}/{KERNEL_NAME}"
    print(f"[main] kaggle user: {username}  slug: {slug}")

    build_metadata(username)
    push()

    final_status = poll(slug)
    if final_status != "complete":
        print(f"[main] kernel did not complete cleanly (status={final_status})")
        # Still attempt to pull whatever outputs may exist for debugging.

    pull_outputs(slug)

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        print("[main] metrics.json:")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(json.dumps(metrics, indent=2))
    else:
        print(f"[main] no metrics.json found at {metrics_path}")

    if final_status != "complete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
