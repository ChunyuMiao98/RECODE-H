#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract task_id -> conda env mapping from a dataset's annotation test.sh files.

Typical use case:
You have a reference dataset that already consolidates tasks into a small set of
generic conda envs (e.g., /home/linus/back/2025_9_15/dataset). This script
extracts the mapping and writes it to the current benchmark's work_dir so the
runner can reuse those envs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence


_RE_CONDA_ACTIVATE = re.compile(r"^\s*conda\s+activate\s+([^\s#]+)\s*$")


def load_task_ids(meta_jsonl: Path) -> List[int]:
    tids: List[int] = []
    for raw in meta_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        tids.append(int(obj["index"]))
    return tids


def parse_env_from_test_sh(test_sh: Path) -> Optional[str]:
    for raw in test_sh.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _RE_CONDA_ACTIVATE.match(raw)
        if m:
            return m.group(1)
    return None


def build_env_map(ref_dataset_dir: Path, task_ids: List[int]) -> Dict[int, str]:
    ann_dir = ref_dataset_dir / "annotations"
    out: Dict[int, str] = {}
    missing: List[int] = []
    for tid in task_ids:
        test_sh = ann_dir / f"annotation_{tid}" / "test.sh"
        if not test_sh.exists():
            missing.append(tid)
            continue
        env = parse_env_from_test_sh(test_sh)
        if env is None:
            missing.append(tid)
            continue
        out[tid] = env
    if missing:
        raise RuntimeError(
            f"Missing mapping for task_ids (no test.sh or no conda activate): {missing}"
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ref-dataset-dir",
        type=str,
        required=True,
        help="Reference dataset dir containing annotations/annotation_*/test.sh",
    )
    p.add_argument(
        "--meta-jsonl",
        type=str,
        required=True,
        help="Current benchmark annotation_meta.jsonl (used to determine task ids)",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON path (task_id->env_name).",
    )
    args = p.parse_args(argv)

    ref = Path(args.ref_dataset_dir)
    meta = Path(args.meta_jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tids = load_task_ids(meta)
    env_map = build_env_map(ref, tids)
    out_path.write_text(
        json.dumps(env_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote env map: {out_path} (tasks={len(env_map)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

