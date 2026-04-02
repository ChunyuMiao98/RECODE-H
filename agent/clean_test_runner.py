#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean(ish) runner for dataset annotation tests.

Why this exists
---------------
The dataset's per-task `test.sh` scripts typically:
- `conda activate test_{id}` (requires interactive shell / modifies caller env)
- copy `canonical.py` + `test.py` into the target repo, run `pytest`, then delete them

This runner keeps the behavior but avoids mutating the target repo by:
- running the test command via `conda run -n <env>` (no shell activation)
- creating a temporary "overlay" Python package tree that injects `canonical.py`/`test.py`
  while extending `__path__` to the real package directories in the target repo.

It still produces the same JUnit XML at:
  `work_dir/dataset/annotations/annotation_<id>/pytest_report/report.xml`
so existing result parsing continues to work.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from env_cache import ensure_conda_env_for_requirements, resolve_env_for_task


@dataclass(frozen=True)
class ParsedTestSh:
    conda_env: str
    target_dir: Path
    cwd: Path
    pyroot: Path
    extra_pythonpath: List[Path]
    copy_actions: List[Tuple[str, Path]]  # (src_basename, dest_dir_abs)
    pytest_timeout: Optional[int]


_RE_CONDA_ACTIVATE = re.compile(r"^\s*conda\s+activate\s+([^\s#]+)\s*$")
_RE_TARGET_DIR = re.compile(r'^\s*TARGET_DIR\s*=\s*"\$\{SCRIPT_DIR\}/([^"]+)"\s*$')
_RE_CD = re.compile(r'^\s*cd\s+"\$\{TARGET_DIR\}/([^"]+)"\s*\|\|\s*exit\s*$')
_RE_CP = re.compile(
    r'^\s*cp\s+"\$\{SCRIPT_DIR\}/([^"]+)"\s+"\$\{TARGET_DIR\}/([^"]+)"\s*$'
)
_RE_PYTEST_TIMEOUT_EQ = re.compile(r"--timeout=(\d+)")
_RE_PYTEST_TIMEOUT_SP = re.compile(r"--timeout\s+(\d+)")
_RE_PYTHONPATH_PREFIX = re.compile(
    r'^\s*PYTHONPATH\s*=\s*"\$\{TARGET_DIR\}/([^"]+)"\s+pytest\s+.*$'
)


def _dedup_keep_order(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def parse_test_sh(annotation_dir: Path) -> ParsedTestSh:
    test_sh = annotation_dir / "test.sh"
    if not test_sh.is_file():
        raise FileNotFoundError(f"Missing test.sh: {test_sh}")

    conda_env: Optional[str] = None
    target_dir_name: Optional[str] = None
    cd_rel: Optional[str] = None
    extra_pythonpath_rel: Optional[str] = None
    copy_actions: List[Tuple[str, str]] = []
    pytest_timeout: Optional[int] = None

    for raw in test_sh.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = _RE_CONDA_ACTIVATE.match(line)
        if m:
            conda_env = m.group(1)
            continue

        m = _RE_TARGET_DIR.match(line)
        if m:
            target_dir_name = m.group(1)
            continue

        m = _RE_CD.match(line)
        if m:
            cd_rel = m.group(1)
            continue

        m = _RE_PYTHONPATH_PREFIX.match(line)
        if m:
            extra_pythonpath_rel = m.group(1)
            continue

        m = _RE_CP.match(line)
        if m:
            src = m.group(1)
            dest_rel = m.group(2)
            copy_actions.append((src, dest_rel))
            continue

        # Extract timeout from pytest line (either `--timeout=10` or `--timeout 10`)
        if "pytest" in line and "--timeout" in line and pytest_timeout is None:
            m1 = _RE_PYTEST_TIMEOUT_EQ.search(line)
            m2 = _RE_PYTEST_TIMEOUT_SP.search(line)
            if m1:
                pytest_timeout = int(m1.group(1))
            elif m2:
                pytest_timeout = int(m2.group(1))

    if conda_env is None:
        raise ValueError(f"Could not parse conda env from {test_sh}")
    if target_dir_name is None:
        raise ValueError(f"Could not parse TARGET_DIR from {test_sh}")
    if cd_rel is None:
        raise ValueError(f"Could not parse cd target from {test_sh}")

    target_dir = (annotation_dir / target_dir_name).resolve()
    cwd = (target_dir / cd_rel).resolve()

    extra_pythonpath: List[Path] = []
    if extra_pythonpath_rel:
        extra_pythonpath.append((target_dir / extra_pythonpath_rel).resolve())

    # Choose the python root used by the original script.
    # - If PYTHONPATH is set, that's the intended root (first entry).
    # - Else, default to TARGET_DIR (repo root).
    pyroot = extra_pythonpath[0] if extra_pythonpath else target_dir

    copy_actions_abs: List[Tuple[str, Path]] = []
    for src, dest_rel in copy_actions:
        # Dest in scripts is `${TARGET_DIR}/...` (a directory like `qa_metrics/` or `DLinear/exp/.`)
        dest_dir = (target_dir / dest_rel).resolve()
        copy_actions_abs.append((Path(src).name, dest_dir))

    return ParsedTestSh(
        conda_env=conda_env,
        target_dir=target_dir,
        cwd=cwd,
        pyroot=pyroot,
        extra_pythonpath=_dedup_keep_order(extra_pythonpath),
        copy_actions=copy_actions_abs,
        pytest_timeout=pytest_timeout,
    )


def _ensure_overlay_package_dir(
    overlay_root: Path, pyroot: Path, rel_pkg_dir: Path, real_pkg_dir: Path
) -> None:
    """
    Make sure each directory component is a Python package in the overlay and
    extend its __path__ to the corresponding directory in the real repo.
    """
    parts = list(rel_pkg_dir.parts)
    cur_rel = Path()
    for part in parts:
        cur_rel = cur_rel / part
        overlay_dir = overlay_root / cur_rel
        overlay_dir.mkdir(parents=True, exist_ok=True)

        init_py = overlay_dir / "__init__.py"
        # Only generate if missing; users may want to inspect/debug overlay dir.
        if init_py.exists():
            continue

        # Compute real dir for this package level (if it exists).
        real_dir_for_level = (pyroot / cur_rel).resolve()
        init_py.write_text(
            "\n".join(
                [
                    "# Auto-generated by clean_test_runner.py (overlay package).",
                    "from __future__ import annotations",
                    "",
                    f"_REAL_PKG_DIR = {str(real_dir_for_level)!r}",
                    "try:",
                    "    __path__.append(_REAL_PKG_DIR)  # type: ignore[name-defined]",
                    "except Exception:",
                    "    # Best-effort; if this isn't treated as a package, imports will fail anyway.",
                    "    pass",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _overlay_copy_file(
    overlay_root: Path,
    pyroot: Path,
    real_dest_dir: Path,
    src_file: Path,
) -> Path:
    """
    Place src_file into overlay path that mirrors where test.sh would copy it.
    Returns the overlay file path.
    """
    # Map real dest dir to a path relative to the python root.
    rel_dir = Path(os.path.relpath(real_dest_dir, pyroot))
    if rel_dir.parts and rel_dir.parts[0] == "..":
        # Fallback: keep overlay stable even if scripts are weird.
        rel_dir = Path(os.path.relpath(real_dest_dir, pyroot.parent))

    overlay_dir = (overlay_root / rel_dir).resolve()
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Ensure overlay packages exist for any intermediate directories.
    if rel_dir != Path("."):
        _ensure_overlay_package_dir(
            overlay_root=overlay_root,
            pyroot=pyroot,
            rel_pkg_dir=rel_dir,
            real_pkg_dir=real_dest_dir,
        )

    overlay_file = overlay_dir / src_file.name
    shutil.copy2(src_file, overlay_file)
    return overlay_file


def run_annotation_tests(
    work_dir: str,
    task_id: int,
    timeout_sec: int = 600,
    env_mode: str = "auto",
    python_version: str = "3.10",
    env_map_path: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run `annotation_<task_id>` tests in a clean way and generate JUnit XML.
    """
    annotation_dir = Path(work_dir) / "dataset" / "annotations" / f"annotation_{task_id}"
    parsed = parse_test_sh(annotation_dir)

    # Optional: use a precomputed task_id -> env mapping (e.g., consolidated generic envs).
    # This is the lowest-friction way to reuse a small set of stable environments.
    if env_mode in ("auto", "env_map"):
        map_path = (
            Path(env_map_path)
            if env_map_path
            else (Path(work_dir) / "dataset" / "env_map.json")
        )
        if map_path.exists():
            raw = json.loads(map_path.read_text(encoding="utf-8", errors="replace"))
            # keys may be strings in JSON
            env = raw.get(str(task_id)) if isinstance(raw, dict) else None
            if env:
                parsed = ParsedTestSh(
                    conda_env=str(env),
                    target_dir=parsed.target_dir,
                    cwd=parsed.cwd,
                    pyroot=parsed.pyroot,
                    extra_pythonpath=parsed.extra_pythonpath,
                    copy_actions=parsed.copy_actions,
                    pytest_timeout=parsed.pytest_timeout,
                )

    # Optional: resolve a shared env from requirements.txt to avoid N-per-task env sprawl.
    # This makes env creation "on demand + cached + reusable" across tasks that share deps.
    if env_mode == "requirements":
        resolved = resolve_env_for_task(
            work_dir=work_dir, annotation_dir=annotation_dir, python_version=python_version
        )
        req_path = annotation_dir / "requirements.txt"
        if req_path.exists():
            shared_env = ensure_conda_env_for_requirements(
                work_dir=work_dir,
                env_name=resolved.env_name,
                requirements_path=req_path,
                python_version=python_version,
                extra_pip=["pytest", "pytest-timeout"],
            )
            parsed = ParsedTestSh(
                conda_env=shared_env,
                target_dir=parsed.target_dir,
                cwd=parsed.cwd,
                pyroot=parsed.pyroot,
                extra_pythonpath=parsed.extra_pythonpath,
                copy_actions=parsed.copy_actions,
                pytest_timeout=parsed.pytest_timeout,
            )
        else:
            # If no requirements file, we still ensure pytest exists in the shared env.
            shared_env = ensure_conda_env_for_requirements(
                work_dir=work_dir,
                env_name=resolved.env_name,
                requirements_path=annotation_dir / "requirements.txt",  # doesn't exist; handled below
                python_version=python_version,
                extra_pip=["pytest", "pytest-timeout"],
            )
            parsed = ParsedTestSh(
                conda_env=shared_env,
                target_dir=parsed.target_dir,
                cwd=parsed.cwd,
                pyroot=parsed.pyroot,
                extra_pythonpath=parsed.extra_pythonpath,
                copy_actions=parsed.copy_actions,
                pytest_timeout=parsed.pytest_timeout,
            )

    report_dir = annotation_dir / "pytest_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_xml = report_dir / "report.xml"
    if report_xml.exists():
        report_xml.unlink()

    canonical_src = annotation_dir / "canonical.py"
    test_src = annotation_dir / "test.py"
    if not canonical_src.is_file():
        raise FileNotFoundError(f"Missing canonical.py: {canonical_src}")
    if not test_src.is_file():
        raise FileNotFoundError(f"Missing test.py: {test_src}")

    with tempfile.TemporaryDirectory(prefix=f"overlay_task_{task_id}_") as td:
        overlay_root = Path(td).resolve()

        overlay_test_file: Optional[Path] = None

        # Build overlay files according to copy actions in test.sh
        # (usually canonical.py and test.py).
        for src_basename, real_dest_dir in parsed.copy_actions:
            if src_basename == "canonical.py":
                _overlay_copy_file(overlay_root, parsed.pyroot, real_dest_dir, canonical_src)
            elif src_basename == "test.py":
                overlay_test_file = _overlay_copy_file(
                    overlay_root, parsed.pyroot, real_dest_dir, test_src
                )
            else:
                # Ignore other copies if present.
                continue

        if overlay_test_file is None:
            # Some scripts might not copy test.py (rare), fall back to original test.py.
            overlay_test_file = test_src

        env = os.environ.copy()

        # Construct PYTHONPATH similar to the original script, but with overlay first.
        existing_pp = env.get("PYTHONPATH", "")
        pp_parts: List[str] = [str(overlay_root)]
        pp_parts.extend(str(p) for p in parsed.extra_pythonpath)
        pp_parts.append(str(parsed.pyroot))
        if existing_pp:
            pp_parts.append(existing_pp)
        env["PYTHONPATH"] = ":".join([p for p in pp_parts if p])

        cmd: List[str] = [
            "conda",
            "run",
            "-n",
            parsed.conda_env,
            "--no-capture-output",
            "python",
            "-m",
            "pytest",
            str(overlay_test_file),
            f"--junitxml={str(report_xml)}",
        ]

        # Carry over pytest-timeout if the original script used it.
        if parsed.pytest_timeout is not None:
            cmd.append(f"--timeout={parsed.pytest_timeout}")

        completed = subprocess.run(
            cmd,
            cwd=str(parsed.cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        # Persist runner logs next to annotation folder like previous scripts do.
        header = "\n".join(
            [
                "[clean_test_runner]",
                f"task_id={task_id}",
                f"env_mode={env_mode}",
                f"conda_env={parsed.conda_env}",
                f"cwd={str(parsed.cwd)}",
                f"pyroot={str(parsed.pyroot)}",
                f"PYTHONPATH={env.get('PYTHONPATH','')}",
                f"cmd={' '.join(cmd)}",
                "",
            ]
        )
        (annotation_dir / "_runner_stdout.txt").write_text(
            header + (completed.stdout or ""), encoding="utf-8"
        )
        (annotation_dir / "_runner_stderr.txt").write_text(
            header + (completed.stderr or ""), encoding="utf-8"
        )

        return completed


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--work-dir", type=str, default="work_dir", help="Work dir containing dataset/"
    )
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--timeout-sec", type=int, default=600)
    p.add_argument(
        "--env-mode",
        type=str,
        default="auto",
        choices=["auto", "test_sh", "requirements", "env_map"],
        help="auto: env_map if present else test_sh; env_map: force mapping; requirements: build/reuse shared env by requirements.txt; test_sh: use env in test.sh",
    )
    p.add_argument("--python-version", type=str, default="3.10")
    p.add_argument(
        "--env-map-path",
        type=str,
        default=None,
        help="Path to env_map.json (task_id -> conda env name). Defaults to <work-dir>/dataset/env_map.json",
    )
    args = p.parse_args(argv)

    cp = run_annotation_tests(
        args.work_dir,
        args.task_id,
        timeout_sec=args.timeout_sec,
        env_mode=args.env_mode,
        python_version=args.python_version,
        env_map_path=args.env_map_path,
    )
    return int(cp.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

