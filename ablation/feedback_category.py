from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))

from constants import *
import os
import pickle
import json
import random
import hashlib
from utils import *
from inference import LLMAPIWrapper

from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, confloat
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

ErrorType = Literal["T0", "T1", "T2", "T3", "T4"]
AdoptStatus = Literal["YES", "NO", "PARTIAL"]

class FeedbackEvaluation(BaseModel):
    error_type: ErrorType = Field(..., description="T0–T4")
    adopted: AdoptStatus = Field(..., description="YES/NO/PARTIAL")
    resolved: AdoptStatus = Field(..., description="YES/NO/PARTIAL")
    explain_error_type: str = Field(..., min_length=1, max_length=500, description="Describe the reason the error is categorized to this error type.")
    explain_adopted_solved: str = Field(..., min_length=1, max_length=500, description="Describe the reason the feedback is adopted or not and the error is resolved")
    confidence_score: confloat(ge=0.0, le=1.0) = Field(..., description="0–1")

class EvaluationResult(BaseModel):
    feedback_evaluations: List[FeedbackEvaluation] = Field(default_factory=list)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sample_task_and_turn(sum_task_id_turn_pairs: List[Tuple[int, int]], n: int = 60):
    """
    Deterministically sample n pairs from a list of (task_id, turn) pairs.
    """
    if not sum_task_id_turn_pairs:
        return []

    n = min(n, len(sum_task_id_turn_pairs))
    data_bytes = str(sum_task_id_turn_pairs).encode("utf-8")
    seed = int(hashlib.md5(data_bytes).hexdigest(), 16)
    rng = random.Random(42)
    return rng.sample(sum_task_id_turn_pairs, n)


def get_turn_number(file_dir: str, task_id: int):
    file_path = os.path.join(file_dir, f'task_{task_id}_result.pkl')
    with open(file_path, 'rb') as result_file:
        data = pickle.load(result_file)
    # len(data) - 1 because code_v2 uses turn+1 below
    return [(task_id, i) for i in range(len(data) - 1)]


def load_task_ids_from_jsonl(jsonl_path: Optional[str] = None):
    if jsonl_path is None:
        jsonl_path = 'work_dir/dataset/annotation_meta.jsonl'
    task_ids = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("if_test") is True:
                task_ids.append(item.get("index"))
    return task_ids


def errors_to_markdown(errors: List[str]) -> str:
    if not errors:
        return "# Error Report\n\n_No errors found._\n"
    lines = ["# Error Report\n"]
    for i, err in enumerate(errors, start=1):
        lines.append(f"## Error {i}\n```text\n{err}\n```\n")
    return "\n".join(lines)


def feedback_to_markdown(feedback_items: List[dict], guidance_level: int) -> str:
    if not feedback_items:
        return "# Feedback Report\n\n_No feedback available._\n"
    lines = ["# Feedback Report\n"]
    for i, item in enumerate(feedback_items, start=1):
        lines.append(f"## Feedback {i}")
        lines.append(f"**Interface:** {item.get('interface', 'N/A')}")

        lines.append("### Description")
        lines.append(f"{item.get('description', 'N/A')}\n")
        if guidance_level > 1:
            lines.append("### Analysis")
            lines.append(f"{item.get('analysis', 'N/A')}\n")
        if guidance_level > 2:
            lines.append("### Actionable Feedback")
            lines.append(f"{item.get('actionable_feedback', 'N/A')}\n")
        if guidance_level > 3:
            lines.append("### Direct Code Feedback")
            lines.append(f"```text\n{item.get('direct_code_feedback', 'N/A')}\n```\n")
    return "\n".join(lines)


class Arg:
    pass


def build_api_wrapper(api_key: str,
                      api_provider: str,
                      model: str,
                      deployment: Optional[str] = None,
                      endpoint: Optional[str] = None,
                      api_version: Optional[str] = None) -> LLMAPIWrapper:
    """Create a fresh wrapper instance (safer for multi-threading)."""
    args = Arg()
    args.api_key = api_key
    args.api_provider = api_provider
    args.model = model
    args.deployment = deployment or model
    args.endpoint = endpoint
    args.api_version = api_version
    return LLMAPIWrapper(args)


def categorize_one(sample_pair: Tuple[int, int],
                   result_dir: str,
                   out_dir: str,
                   guidance_level: int,
                   work_dir: Optional[str],
                   api_key: str,
                   api_provider: str,
                   model: str,
                   deployment: Optional[str],
                   endpoint: Optional[str],
                   api_version: Optional[str],
                   api_model: str) -> Tuple[Tuple[int, int], Optional[str], Optional[str]]:
    """
    Run one categorization unit. Returns (sample_pair, out_path or None, error or None).
    """
    try:
        task_id, turn = sample_pair
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f'task_{task_id}_turn_{turn}.pkl')
        if os.path.exists(out_path):
            return sample_pair, out_path, None

        if work_dir is None:
            work_dir = 'work_dir'

        # Load result data
        file_path = os.path.join(result_dir, f'task_{task_id}_result.pkl')
        with open(file_path, 'rb') as result_file:
            data = pickle.load(result_file)

        paper_content = DatasetManager.load_task_latex(work_dir, task_id)
        canonical_code = DatasetManager.load_task_canonical(work_dir, task_id)
        instrction_code = DatasetManager.load_task_instruction(work_dir, task_id)

        code_v1 = data[turn]['final_code']
        code_v1_log = errors_to_markdown(data[turn]["execution_result"]['failure_messages'])
        feedback_content = feedback_to_markdown(
            data[turn]['feedback_interaction_history'][-1]['content'], guidance_level
        )
        code_v2 = data[turn + 1]['final_code']
        code_v2_log = errors_to_markdown(data[turn + 1]["execution_result"]['failure_messages'])
        canonical_code = 'No canonical code'
        user_promt = CATEGORIZE_PROMPT_USR.format(
            paper_content,
            instrction_code,
            canonical_code,
            code_v1,
            code_v1_log,
            feedback_content,
            code_v2,
            code_v2_log,  # <-- fixed (was incorrectly using code_v1_log again)
        )

        # Per-thread API client
        api_wrapper = build_api_wrapper(
            api_key=api_key,
            api_provider=api_provider,
            model=api_model,
            deployment=deployment,
            endpoint=endpoint,
            api_version=api_version,
        )

        result = api_wrapper.query_openai_parse_no_stat(
            CATEGORIZE_PROMPT_SYS, user_promt, EvaluationResult
        )

        with open(out_path, "wb") as f:
            pickle.dump(result, f)

        return sample_pair, out_path, None

    except Exception as e:
        return sample_pair, None, f"{type(e).__name__}: {e}"


def main():
    # ==== CONFIG ====
    models = [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "deepseek-chat",
        "claude-sonnet-4-20250514",
    ]
    bash_path = 'expriment_result'
    guidance_levels = [1, 2, 3, 4]
    # models = [
    #     "gpt-5",
    # ]
    # guidance_levels = [1]
    out_base_dir = 'feedback_categorize'

    # Fill in your real key/provider/etc.
    API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
    API_PROVIDER = "openai"
    DEFAULT_MODEL = "gpt-5"
    DEFAULT_DEPLOYMENT = "gpt-5"
    DEFAULT_ENDPOINT = None
    DEFAULT_API_VERSION = None

    # Threading settings
    max_workers = min(32, os.cpu_count() or 8)  # sensible default

    task_ids = load_task_ids_from_jsonl()

    for model in models:
        for guidance_level in guidance_levels:
            sum_task_id_turn_pairs: List[Tuple[int, int]] = []
            result_dir = os.path.join(bash_path, f'memory_agent_{model}_feedback_fixed_guidance_{guidance_level}')
            out_dir = os.path.join(out_base_dir, f'{model}_{guidance_level}')

            # Build the list of (task_id, turn) pairs
            for task_id in task_ids:
                try:
                    pairs = get_turn_number(result_dir, task_id)
                    sum_task_id_turn_pairs.extend(pairs)
                except FileNotFoundError:
                    # Skip missing tasks silently; you can log if desired
                    continue

            sampled_pairs = sample_task_and_turn(sum_task_id_turn_pairs)

            # Prepare partial job function for threads
            job = partial(
                categorize_one,
                result_dir=result_dir,
                out_dir=out_dir,
                guidance_level=guidance_level,
                work_dir='work_dir',
                api_key=API_KEY,
                api_provider=API_PROVIDER,
                model=model,
                deployment=DEFAULT_DEPLOYMENT,
                endpoint=DEFAULT_ENDPOINT,
                api_version=DEFAULT_API_VERSION,
                api_model = DEFAULT_MODEL
            )

            # Submit jobs
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(sampled_pairs), desc=f"{model} | G{guidance_level}") as pbar:
                for sp in sampled_pairs:
                    futures.append(executor.submit(job, sp))

                errors = []
                for fut in as_completed(futures):
                    _, out_path, err = fut.result()
                    if err:
                        errors.append(err)
                    pbar.update(1)

            # Optionally, write a run-level error report
            if errors:
                ensure_dir(out_dir)
                err_md = errors_to_markdown(errors)
                with open(os.path.join(out_dir, "_thread_run_errors.md"), "w", encoding="utf-8") as f:
                    f.write(err_md)


if __name__ == '__main__':
    main()
