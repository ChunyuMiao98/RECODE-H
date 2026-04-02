#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_COMMENT_RE = re.compile(r"\s+#.*$")


def normalize_requirements_text(text: str) -> str:
    """
    Canonicalize requirements.txt content to improve env reuse:
    - strip comments/whitespace
    - drop blank lines
    - de-duplicate
    - sort
    """
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _COMMENT_RE.sub("", line).strip()
        if not line:
            continue
        lines.append(line)
    uniq = sorted(set(lines))
    return "\n".join(uniq) + ("\n" if uniq else "")


def fingerprint_requirements(requirements_path: Path) -> str:
    data = requirements_path.read_text(encoding="utf-8", errors="replace")
    norm = normalize_requirements_text(data)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def default_env_name(fingerprint: str, py_version: str) -> str:
    # keep under conda's name constraints; short and deterministic
    py_tag = py_version.replace(".", "")
    return f"rcg_py{py_tag}_{fingerprint}"


def _run(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def conda_env_exists(env_name: str) -> bool:
    cp = _run(["conda", "env", "list", "--json"], check=True)
    data = json.loads(cp.stdout or "{}")
    envs = data.get("envs", []) or []
    for p in envs:
        # conda includes base paths; infer name by last component
        if os.path.basename(p) == env_name:
            return True
    return False


def ensure_conda_env_for_requirements(
    *,
    work_dir: str,
    env_name: str,
    requirements_path: Path,
    python_version: str = "3.10",
    extra_pip: Optional[List[str]] = None,
) -> str:
    """
    Ensure a conda env exists and has the dependencies from requirements_path installed.
    Uses a stamp file under <work_dir>/cache/env_stamps/ to avoid redundant installs.
    """
    work_dir_p = Path(work_dir)
    stamp_dir = work_dir_p / "cache" / "env_stamps"
    stamp_dir.mkdir(parents=True, exist_ok=True)

    # requirements_path may be missing for some tasks; treat as a shared "base" env.
    if requirements_path.exists():
        fp = fingerprint_requirements(requirements_path)
    else:
        fp = "base"
    stamp_file = stamp_dir / f"{env_name}.sha"

    desired_stamp = fp
    if extra_pip:
        desired_stamp += ":" + hashlib.sha256("\n".join(extra_pip).encode("utf-8")).hexdigest()[:12]

    if conda_env_exists(env_name) and stamp_file.exists():
        if stamp_file.read_text(encoding="utf-8", errors="ignore").strip() == desired_stamp:
            return env_name

    if not conda_env_exists(env_name):
        _run(["conda", "create", "-y", "-n", env_name, f"python={python_version}", "pip"], check=True)

    # Install requirements via pip inside env (most annotation requirements are pip-style pins).
    _run(["conda", "run", "-n", env_name, "--no-capture-output", "python", "-m", "pip", "install", "-U", "pip"], check=True)
    if requirements_path.exists():
        _run(["conda", "run", "-n", env_name, "--no-capture-output", "python", "-m", "pip", "install", "-r", str(requirements_path)], check=True)
    if extra_pip:
        _run(["conda", "run", "-n", env_name, "--no-capture-output", "python", "-m", "pip", "install", *extra_pip], check=True)

    stamp_file.write_text(desired_stamp + "\n", encoding="utf-8")
    return env_name


@dataclass(frozen=True)
class EnvResolveResult:
    env_name: str
    fingerprint: Optional[str]


def resolve_env_for_task(
    *,
    work_dir: str,
    annotation_dir: Path,
    python_version: str = "3.10",
) -> EnvResolveResult:
    """
    Resolve a shared env for a task based on its annotation requirements.txt.
    If no requirements.txt, returns a base env name (still shared).
    """
    req = annotation_dir / "requirements.txt"
    if req.exists():
        fp = fingerprint_requirements(req)
        return EnvResolveResult(env_name=default_env_name(fp, python_version), fingerprint=fp)
    # fallback: base env for tasks that don't specify extra deps
    base_fp = "base"
    return EnvResolveResult(env_name=default_env_name(base_fp, python_version), fingerprint=None)

