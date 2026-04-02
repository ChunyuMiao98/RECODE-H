#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prebuild / warm up shared environments for annotation tasks.

This pairs with `clean_test_runner.py --env-mode requirements`:
- environments are keyed by normalized requirements.txt fingerprint
- created on demand and cached, but you can also prebuild them up front
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional

from env_cache import ensure_conda_env_for_requirements, resolve_env_for_task


def iter_task_ids_from_meta(meta_path: Path) -> List[int]:
    tids: List[int] = []
    for raw in meta_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        tids.append(int(obj["index"]))
    return tids


def iter_task_ids_from_annotations(annotations_dir: Path) -> List[int]:
    tids: List[int] = []
    for p in sorted(annotations_dir.glob("annotation_*")):
        if not p.is_dir():
            continue
        try:
            tid = int(p.name.split("_", 1)[1])
        except Exception:
            continue
        tids.append(tid)
    return tids


def _unique_envs_for_tasks(
    work_dir: str, annotations_dir: Path, task_ids: Iterable[int], python_version: str
) -> List[tuple[str, Optional[Path]]]:
    # env_name -> requirements_path (or None)
    envs: dict[str, Optional[Path]] = {}
    for tid in task_ids:
        ann = annotations_dir / f"annotation_{tid}"
        if not ann.exists():
            continue
        resolved = resolve_env_for_task(
            work_dir=work_dir, annotation_dir=ann, python_version=python_version
        )
        req = ann / "requirements.txt"
        envs.setdefault(resolved.env_name, req if req.exists() else None)
    # stable order
    return sorted(envs.items(), key=lambda kv: kv[0])


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=str, required=True, help="Work dir containing dataset/")
    p.add_argument(
        "--source",
        type=str,
        default="meta",
        choices=["meta", "scan"],
        help="Use annotation_meta.jsonl indices (meta) or scan annotations directory (scan)",
    )
    p.add_argument("--python-version", type=str, default="3.10")
    p.add_argument(
        "--max-workers", type=int, default=1, help="Parallelism for env creation (use with care)"
    )
    args = p.parse_args(argv)

    work_dir_p = Path(args.work_dir)
    annotations_dir = work_dir_p / "dataset" / "annotations"
    meta_path = work_dir_p / "dataset" / "annotation_meta.jsonl"

    if args.source == "meta":
        task_ids = iter_task_ids_from_meta(meta_path)
    else:
        task_ids = iter_task_ids_from_annotations(annotations_dir)

    envs = _unique_envs_for_tasks(args.work_dir, annotations_dir, task_ids, args.python_version)
    print(f"tasks: {len(task_ids)}")
    print(f"unique envs (by requirements fingerprint): {len(envs)}")

    def build_one(env_name: str, req_path: Optional[Path]) -> str:
        return ensure_conda_env_for_requirements(
            work_dir=args.work_dir,
            env_name=env_name,
            requirements_path=req_path if req_path is not None else Path("__missing_requirements__.txt"),
            python_version=args.python_version,
            extra_pip=["pytest", "pytest-timeout"],
        )

    if args.max_workers <= 1:
        for env_name, req in envs:
            print(f"[build] {env_name} req={'yes' if req else 'no'}")
            build_one(env_name, req)
        return 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(build_one, env_name, req): env_name for env_name, req in envs}
        for fut in as_completed(futs):
            env_name = futs[fut]
            try:
                fut.result()
                print(f"[ok] {env_name}")
            except Exception as e:
                print(f"[fail] {env_name}: {e}")
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

