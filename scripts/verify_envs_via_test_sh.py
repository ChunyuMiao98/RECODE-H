#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify conda test environments by running each annotation's original `test.sh`.

Rationale
---------
The benchmark's `annotation_<id>/test.sh` typically:
- `conda activate <env>`
- copy `canonical.py` + `test.py` into a specific location inside the target repo
- run `pytest ... --junitxml=...`
- cleanup copied files

If you want to validate that your newly installed conda envs can run the benchmark,
the most faithful method is: run `test.sh` directly (no parsing / overlay).

What this script does
---------------------
- Load tasks from <dataset_dir>/annotation_meta.jsonl (if_test == true)
- For each task:
  - write init_content + append annotation's canonical.py into the target file
    (so the "generated" code is likely correct, turning this into an env sanity check)
  - run `./test.sh` via `/bin/bash -il -c ...` inside the annotation dir
  - require a fresh JUnit XML at `pytest_report/report.xml`
  - parse it and summarize (tests / failures / errors / skipped)
  - restore the original target file (unless --keep-modified)

This script validates executability of environments; it is not meant to measure model performance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TaskResult:
    task_id: int
    ok: bool
    rc: int
    total: int
    failures: int
    errors: int
    skipped: int
    note: str
    report_path: str


def load_test_items(jsonl_path: Path) -> List[Dict]:
    """Read JSONL and return if_test==True items, sorted by index."""
    seen = set()
    items: List[Dict] = []
    for raw in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            item = json.loads(s)
        except json.JSONDecodeError:
            continue
        if item.get("if_test") is True:
            idx = item.get("index")
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            items.append(item)
    items.sort(key=lambda x: int(x["index"]))
    return items


def build_target_path(base_dir: Path, index_num: int, repo_dir_name: str, target_file_path: str) -> Path:
    """annotations/annotation_{index}/{repo_dir_name}/{target_file_path}"""
    return base_dir / f"annotation_{index_num}" / repo_dir_name / Path(str(target_file_path))


def prepare_target_file(
    base_dir: Path,
    index_num: int,
    repo_dir_name: str,
    target_file_path: str,
    init_content,
) -> Tuple[bool, str, Path, Optional[str]]:
    """
    Write init_content (None/'N/A' treated as ''), then append annotation_<id>/canonical.py.
    Returns: (ok, err_msg, target_path, original_text)
    """
    annotation_dir = base_dir / f"annotation_{index_num}"
    canonical_path = annotation_dir / "canonical.py"
    target_path = build_target_path(base_dir, index_num, repo_dir_name, target_file_path)

    if not canonical_path.is_file():
        return False, f"canonical.py not found: {canonical_path}", target_path, None

    if not isinstance(init_content, str) or init_content == "N/A":
        init_content = ""

    os.makedirs(target_path.parent, exist_ok=True)

    original_text: Optional[str] = None
    try:
        if target_path.exists():
            original_text = target_path.read_text(encoding="utf-8", errors="replace")
        else:
            original_text = None

        canonical_code = canonical_path.read_text(encoding="utf-8", errors="replace")

        with open(target_path, "w", encoding="utf-8") as fw:
            fw.write(init_content)
            if init_content and not init_content.endswith("\n"):
                fw.write("\n")
            fw.write("\n# === Appended from canonical.py ===\n")
            fw.write(canonical_code)
            if canonical_code and not canonical_code.endswith("\n"):
                fw.write("\n")

        return True, "", target_path, original_text
    except Exception as e:
        return False, f"write/append error: {e}", target_path, original_text


def restore_target_file(target_path: Path, original_text: Optional[str]) -> None:
    try:
        if original_text is None:
            # file didn't exist
            if target_path.exists():
                target_path.unlink()
        else:
            target_path.write_text(original_text, encoding="utf-8")
    except Exception:
        # best-effort restore
        pass


def expected_report_path(base_dir: Path, annotation_index: int) -> Path:
    return base_dir / f"annotation_{annotation_index}" / "pytest_report" / "report.xml"


def fresh_report_exists(report_path: Path, started_after: float) -> bool:
    try:
        st = report_path.stat()
        return st.st_mtime >= started_after
    except FileNotFoundError:
        return False


def parse_junit_report(report_path: Path) -> Tuple[int, int, int, int]:
    """Parse JUnit XML, return (total, failures, errors, skipped)."""
    tree = ET.parse(str(report_path))
    root = tree.getroot()

    def agg_suite(suite) -> Tuple[int, int, int, int]:
        total = int(suite.attrib.get("tests", 0) or 0)
        failures = int(suite.attrib.get("failures", 0) or 0)
        errors = int(suite.attrib.get("errors", 0) or 0)
        skipped = int(suite.attrib.get("skipped", 0) or suite.attrib.get("disabled", 0) or 0)
        return total, failures, errors, skipped

    if root.tag == "testsuite":
        return agg_suite(root)
    if root.tag == "testsuites":
        total = failures = errors = skipped = 0
        for suite in root.findall("testsuite"):
            t, f, e, s = agg_suite(suite)
            total += t
            failures += f
            errors += e
            skipped += s
        return total, failures, errors, skipped

    suites = root.findall(".//testsuite")
    if suites:
        total = failures = errors = skipped = 0
        for suite in suites:
            t, f, e, s = agg_suite(suite)
            total += t
            failures += f
            errors += e
            skipped += s
        return total, failures, errors, skipped
    raise ValueError(f"Unrecognized JUnit XML root tag: {root.tag}")


def is_success(rc: int, total: int, failures: int, errors: int) -> bool:
    return rc == 0 and total > 0 and failures == 0 and errors == 0


def run_test_sh(base_dir: Path, annotation_index: int, timeout_sec: int) -> Tuple[int, Path, str, float]:
    annotation_dir = base_dir / f"annotation_{annotation_index}"
    test_sh_path = annotation_dir / "test.sh"
    report_path = expected_report_path(base_dir, annotation_index)
    stdout_log = annotation_dir / "_envcheck_stdout.txt"
    stderr_log = annotation_dir / "_envcheck_stderr.txt"

    if not test_sh_path.is_file():
        return -999, report_path, "test.sh not found", 0.0

    # clean old report + ensure dir
    try:
        if report_path.is_file():
            report_path.unlink()
        report_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # ensure executable
    try:
        st = test_sh_path.stat()
        test_sh_path.chmod(st.st_mode | 0o111)
    except Exception:
        pass

    # IMPORTANT:
    # Many annotation test.sh scripts do not `set -e`, so `pytest` failures can be
    # swallowed (script continues to cleanup and exits 0). For env validation we
    # want pytest errors to propagate to the return code, so run via `bash -e`.
    cmd = "bash -e ./test.sh"
    start_ts = time.time()
    try:
        completed = subprocess.run(
            ["/bin/bash", "-il", "-c", cmd],
            cwd=str(annotation_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            text=True,
        )
        rc = int(completed.returncode)
        stdout_log.write_text(completed.stdout or "", encoding="utf-8")
        stderr_log.write_text(completed.stderr or "", encoding="utf-8")

        note = ""
        err_lower = (completed.stderr or "").lower()
        if "unrecognized arguments" in err_lower and "--cov" in err_lower:
            note = "pytest-cov missing (pytest does not recognize --cov args)"
        if rc == 127 or "command not found" in err_lower:
            note = "possible PATH/env issue (pytest/conda/python not found)"

        return rc, report_path, note, start_ts
    except subprocess.TimeoutExpired:
        stdout_log.write_text(f"[TIMEOUT > {timeout_sec}s]\n", encoding="utf-8")
        stderr_log.write_text(f"[TIMEOUT > {timeout_sec}s]\n", encoding="utf-8")
        return -998, report_path, f"timeout > {timeout_sec}s", start_ts
    except Exception as e:
        stderr_log.write_text(f"[exec error] {e}\n", encoding="utf-8")
        return -997, report_path, f"exec error: {e}", start_ts


def _run_one(
    base_dir: Path,
    item: Dict,
    timeout_sec: int,
    keep_modified: bool,
) -> TaskResult:
    idx = int(item["index"])
    repo = item.get("repo_dir_name")
    tgt = item.get("target_file_path")
    init_content = item.get("init_content")

    if not repo or not tgt:
        return TaskResult(idx, False, -1, 0, 0, 0, 0, "missing repo_dir_name/target_file_path", "")

    ok, prep_msg, target_path, original_text = prepare_target_file(base_dir, idx, repo, tgt, init_content)
    if not ok:
        return TaskResult(idx, False, -1, 0, 0, 0, 0, f"prepare failed: {prep_msg}", "")

    try:
        rc, report_path, note, start_ts = run_test_sh(base_dir, idx, timeout_sec)
        if rc < 0:
            return TaskResult(idx, False, rc, 0, 0, 0, 0, note, str(report_path))
        if not fresh_report_exists(report_path, start_ts):
            return TaskResult(idx, False, rc, 0, 0, 0, 0, "missing_or_stale_report", str(report_path))

        total, failures, errors, skipped = parse_junit_report(report_path)
        ok2 = is_success(rc, total, failures, errors)
        return TaskResult(idx, ok2, rc, total, failures, errors, skipped, note, str(report_path))
    finally:
        if not keep_modified:
            restore_target_file(target_path, original_text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=str, default="work_dir/dataset", help="Path containing annotation_meta.jsonl and annotations/")
    p.add_argument("--timeout-sec", type=int, default=600)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--task-ids", type=str, default=None, help="Comma-separated indices, e.g. 1,2,3. Default: all if_test==true")
    p.add_argument("--keep-modified", action="store_true", help="Do not restore the target file after running.")
    args = p.parse_args(argv)

    dataset_dir = Path(args.dataset_dir).resolve()
    jsonl_path = dataset_dir / "annotation_meta.jsonl"
    base_dir = dataset_dir / "annotations"

    if not jsonl_path.is_file():
        print(f"[fatal] jsonl not found: {jsonl_path}")
        return 2
    if not base_dir.is_dir():
        print(f"[fatal] annotations dir not found: {base_dir}")
        return 2

    items = load_test_items(jsonl_path)
    if args.task_ids:
        allow = {int(x.strip()) for x in args.task_ids.split(",") if x.strip()}
        items = [it for it in items if int(it["index"]) in allow]

    print(f"Discovered {len(items)} testable annotations from {jsonl_path}")
    print(f"Running with workers={args.max_workers} timeout={args.timeout_sec}s keep_modified={args.keep_modified}")

    results: List[TaskResult] = []
    if args.max_workers <= 1:
        for it in items:
            r = _run_one(base_dir, it, args.timeout_sec, args.keep_modified)
            results.append(r)
            status = "PASS" if r.ok else "FAIL"
            print(f"[{status}] annotation_{r.task_id}: total={r.total} fail={r.failures} err={r.errors} skip={r.skipped} rc={r.rc} note={r.note}")
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(_run_one, base_dir, it, args.timeout_sec, args.keep_modified): int(it["index"]) for it in items}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                status = "PASS" if r.ok else "FAIL"
                print(f"[{status}] annotation_{r.task_id}: total={r.total} fail={r.failures} err={r.errors} skip={r.skipped} rc={r.rc} note={r.note}")

    failed = sorted({r.task_id for r in results if not r.ok})
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Total tested: {len(results)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failed annotation indices:")
        print(", ".join(str(i) for i in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

