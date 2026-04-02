import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))

import json
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple
import bisect
import argparse
import yaml
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
import traceback
import copy
import time

# Assuming these are your project's modules.
# Ensure they are importable in your environment.
from common_imports import * # noqa: F401,F403
from agents import * # noqa: F401,F403
from utils import * # noqa: F401,F403


def _file_path(code_agent: str, task_id: str) -> str:
    """Constructs the file path for a given task's result pickle file."""
    return f"expriment_result/memory_agent_{code_agent}_feedback_fixed_guidance_4/task_{task_id}_result.pkl"

def _valid_index_count(code_agent: str, task_id: str) -> int:
    """
    Returns how many valid indices a task contributes.
    For a list of length m:
      - if m <= 1 -> 0 (skip)
      - else -> m - 1 (valid indices are [0 .. m-2])
    Any errors or non-list payloads yield 0.
    """
    try:
        with open(_file_path(code_agent, task_id), "rb") as f:
            data = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        # Catch specific errors for better debugging
        return 0
    if not isinstance(data, list):
        return 0
    m = len(data)
    return max(m - 1, 0)

def sample_unique_task_index_pairs(task_ids: List[str], code_agent: str, n: int = 50, seed: int = 42) -> List[Tuple[str, int]]:
    """
    Samples n unique (task_id, idx) pairs across all tasks.
    - A task can appear multiple times (with different idx values).
    - Tasks with len(data) <= 1 are skipped entirely.
    - Indices are drawn from [0, len(data)-2] for each task.
    - Sampling is without replacement over the global pair space.
    - Fully reproducible via `seed`.

    Raises:
        ValueError if fewer than n unique pairs are available.
    """
    tasks = []
    counts = []
    for tid in task_ids:
        c = _valid_index_count(code_agent, tid)
        if c > 0:
            tasks.append(tid)
            counts.append(c)

    if not tasks:
        raise ValueError("No tasks with valid indices (len(data) > 1).")

    prefix = []
    running = 0
    for c in counts:
        running += c
        prefix.append(running)

    total_pairs = prefix[-1]
    if total_pairs < n:
        raise ValueError(
            f"Requested {n} pairs but only {total_pairs} unique pairs are available."
        )

    rng = random.Random(seed)
    global_choices = rng.sample(range(total_pairs), n)

    pairs: List[Tuple[str, int]] = []
    for g in global_choices:
        task_idx = bisect.bisect_right(prefix, g)
        prev_total = prefix[task_idx - 1] if task_idx > 0 else 0
        local_idx = g - prev_total
        pairs.append((tasks[task_idx], local_idx))

    return pairs

def update_task_id(code_agent, human_agent, task_id: int) -> None:
    """Propagate task id to all agents to track current task."""
    code_agent.update_task_id(task_id)
    human_agent.update_task_id(task_id)

def arg_modify(args, code_model, ablation_model, guidance_level):
    """Correctly modifies args with the provided guidance level."""
    new_args = copy.deepcopy(args)
    new_args.human_agent_model = ablation_model
    new_args.human_deployment = ablation_model
    new_args.model =code_model
    new_args.deployment =code_model
    # BUG FIX: Use the guidance_level passed as an argument.
    new_args.guidance_level = guidance_level
    return new_args

def init_code_agent(args):
    return CodeAgent(args)

def init_human_agent(args):
    return HumanAgent(args)

def record_data_agent(
        code_agent_data: Dict[str, Any],
        human_agent_data: Dict[str, Any],
        feedback_info:  Dict[str, Any],
        output_file_path: Optional[str] = None,
    ) -> None:
    """
    Persist an agent turn to a *unique* file. This is process-safe.
    """
    # Defensive check to ensure a file path is provided.
    if output_file_path is None:
        raise ValueError("output_file_path cannot be None.")

    new_record = {
        "turn_number": code_agent_data["turn_number"],
        "code_input_tokens": code_agent_data["input_token"],
        "code_output_tokens": code_agent_data["output_token"],
        "feedback_info": feedback_info,
        "final_code": code_agent_data["final_code"],
        "agent_memory": code_agent_data["memory"],
        "code_interaction_history": code_agent_data["interaction_history"],
        "execution_result": human_agent_data["execution_result"],
        "feedback_input_token": human_agent_data["input_token"],
        "feedback_output_token": human_agent_data["output_token"],
        "feedback_interaction_history": human_agent_data["interaction_history"],
    }
    # Each process writes to its own file, so no lock is needed.
    with open(output_file_path, "wb") as f:
        pickle.dump([new_record], f) # Store as a list for consistency with merged file

def generate_feedback_content(feedback_info, feedback_agent, feedback_dic, guidance_level):
    if feedback_dic is None:
        human_agent_data, _, _ = feedback_agent.execute_feedback(True)
        feedback_dic = human_agent_data["interaction_history"][1]["content"]
        feedback_info = human_agent_data
    return feedback_info, feedback_dic, format_feedback(feedback_dic, guidance_level, "fixed")

def run_single(code_model, ablation_model, sampled_pair, args):
    """
    Runs a single experimental unit. Writes its result to a unique file.
    """
    task_id, turn_index = sampled_pair
    input_dir = f'expriment_result/memory_agent_{code_model}_feedback_fixed_guidance_4'
    # Define a temporary directory for intermediate, per-process results
    temp_output_dir = f'expriment_result/temp_{code_model}_{ablation_model}'
    os.makedirs(temp_output_dir, exist_ok=True)

    result_file_path = os.path.join(input_dir, f"task_{task_id}_result.pkl")
    if not os.path.exists(result_file_path):
        print(f"Warning: Skipping pair ({task_id}, {turn_index}) because source file not found: {result_file_path}")
        return

    with open(result_file_path, "rb") as f:
        data = pickle.load(f)

    # Validate that the turn_index is valid for the loaded data
    if not (isinstance(data, list) and 0 <= turn_index < len(data)):
        print(f"Warning: Skipping pair ({task_id}, {turn_index}) due to invalid index or data format.")
        return

    execution_result = data[turn_index]["execution_result"]
    new_arg = arg_modify(args, code_model, ablation_model, 4)
    code_agent = init_code_agent(new_arg)
    feedback_agent = init_human_agent(new_arg)
    update_task_id(code_agent, feedback_agent, int(task_id))
    # LOGIC FIX: Reset feedback_dic to None for each guidance level to generate fresh feedback.
    feedback_dic = None
    feedback_info = None
    for guidance_level in range(1, 5):
        code_agent.set_history(data[turn_index]["agent_memory"])
        final_code = data[turn_index]["final_code"]
        DatasetManager.write_code_content(
                code_agent.work_dir, code_agent.task_id, final_code
            )

        # PROCESS-SAFE CHANGE: Create a unique file path for this specific run.
        output_file_path = os.path.join(temp_output_dir, f"result_{task_id}_{turn_index}_{guidance_level}.pkl")
        feedback_info, feedback_dic, feedback_content = generate_feedback_content(feedback_info, feedback_agent, feedback_dic, guidance_level)
        code_agent_data = code_agent.generate_agent_history(
            feedback=feedback_content, result=execution_result, turn_num=turn_index + 1
        )
        human_agent_data, _, _ = feedback_agent.execute_feedback(False)
        record_data_agent(code_agent_data, human_agent_data, feedback_info, output_file_path)
        DatasetManager.clean_target_file_content(code_agent.work_dir, code_agent.task_id)

def run_ablation(code_model, ablation_model, tested_idx, args, max_workers):
    """
    Runs the full ablation experiment for a given model pair using a process pool.
    """
    print(f"Starting ablation for code_model='{code_model}', ablation_model='{ablation_model}' with {max_workers} workers.")
    sampled_pairs = sample_unique_task_index_pairs(tested_idx, code_model)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the process pool
        futures = [executor.submit(run_single, code_model, ablation_model, pair, args) for pair in sampled_pairs]

        # Process results as they complete (optional, good for progress tracking)
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
                print(f"Completed task {i + 1}/{len(sampled_pairs)}")
            except Exception as e:
                # This prints the full, detailed traceback
                print(f"--- A task generated an exception: {e} ---")
                traceback.print_exc()
                print("------------------------------------------")

    print("All tasks completed. Merging results...")
    merge_results(code_model, ablation_model)
    print("Merging complete.")


def merge_results(code_model, ablation_model):
    """
    Moves temporary result files to their final destination, preserving all
    identifying information including the guidance level in the filename.
    """
    temp_output_dir = f'expriment_result/temp_{code_model}_{ablation_model}'
    final_output_dir = f'expriment_result/ablation_{code_model}_{ablation_model}'
    os.makedirs(final_output_dir, exist_ok=True)

    temp_files = glob.glob(os.path.join(temp_output_dir, "*.pkl"))

    for temp_file in temp_files:
        # Extract identifiers from filename: result_{task_id}_{turn_index}_{guidance_level}.pkl
        basename = os.path.basename(temp_file)
        
        # Check if the filename matches the expected format
        if basename.startswith('result_') and basename.count('_') >= 3:
            parts = basename.replace('.pkl', '').split('_')
            task_id = parts[1]
            turn_index = parts[2]
            guidance_level = parts[3]
            
            # Create a new filename that preserves the guidance level
            final_filename = f"task_{task_id}_{turn_index}_{guidance_level}.pkl"
            final_filepath = os.path.join(final_output_dir, final_filename)

            # Copy the file content to the new location.
            # This is safer than os.rename which might fail across filesystems.
            try:
                with open(temp_file, 'rb') as f_in:
                    data = pickle.load(f_in)
                with open(final_filepath, 'wb') as f_out:
                    pickle.dump(data, f_out)
            except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
                print(f"Could not process temporary file {temp_file}, skipping. Error: {e}")
        else:
            print(f"Skipping unexpected file in temp directory: {basename}")
    
    # Optional: Clean up temporary files after merging
    # for temp_file in temp_files:
    #     os.remove(temp_file)
    # if os.path.exists(temp_output_dir):
    #     os.rmdir(temp_output_dir)


def get_indexes_from_jsonl(file_path):
    """Reads a JSONL file and returns a list of indexes."""
    indexes = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record.get("if_test") is not None:
                        indexes.append(record.get("index"))
    except FileNotFoundError:
        print(f"Error: Dataset meta file not found at {file_path}")
        return []
    return indexes

class YamlDataHolder(SimpleNamespace):
    """Simple bag for YAML-derived attributes."""
    def __str__(self) -> str:
        return str(self.__dict__)

def parse_yaml(yaml_to_use: str) -> YamlDataHolder:
    """
    Parse the YAML config into a simple args-like object.
    """
    with open(yaml_to_use, "r") as file:
        yaml_data = yaml.safe_load(file)

    args = YamlDataHolder()

    # Core (use .get with sensible defaults)
    args.api_key = yaml_data.get("api-key")
    args.api_provider = yaml_data.get("api-provider")
    args.model = yaml_data.get("model")
    args.deployment = yaml_data.get("deployment")
    args.endpoint = yaml_data.get("endpoint")
    args.api_version = yaml_data.get("api-version")

    # Human agent config
    args.human_api_key = yaml_data.get("human-api-key")
    args.human_api_provider = yaml_data.get("human-api-provider")
    args.human_deployment = yaml_data.get("human-deployment")
    args.human_endpoint = yaml_data.get("human-endpoint")
    args.human_api_version = yaml_data.get("human-api-version")
    args.human_agent_model = yaml_data.get("human-agent-model")

    # Guidance / evaluation
    args.guidance_level = yaml_data.get("guidance-level")
    args.guidance_type = yaml_data.get("guidance-type")
    args.evaluation_setting = yaml_data.get("evaluation-setting")

    # Loop bounds
    args.max_turn = yaml_data.get("max-turn", 5)
    args.max_steps = yaml_data.get("max-steps", 5)

    # Paths
    args.work_dir = yaml_data.get("work-dir")
    args.output_dir = yaml_data.get("output-dir", "output_dir")

    
    # Seed store (supports multiple spellings)
    args.init_seed = yaml_data.get("init_seed", "seeds")
    # Memory settings
    args.memory_threshold = yaml_data.get("memory-threshold", 10)
    args.memory_to_keep = yaml_data.get("memory-to-keep", 5)

    # Replay / resume
    args.replay_feedback_level = yaml_data.get("replay-feedback-level")
    args.resume = yaml_data.get("resume", False)

    return args


def parse_arguments() -> argparse.Namespace:
    """Parse CLI args that control runner behavior."""
    parser = argparse.ArgumentParser(description="Run code-generation workflow.")
    parser.add_argument(
        "--yaml-location",
        type=str,
        default="config/config_gpt-5-mini-bak/default_gpt4o_o1feedback_openai_level4_gpt4.1-mini.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum number of parallel workers (default: 4)",
    )
    # Removed parallel-mode as this script is now process-based.
    return parser.parse_args()


def main():
    user_args = parse_arguments()
    args = parse_yaml(user_args.yaml_location)
    # Corrected placeholder model names - replace with actual model identifiers
    # ablation_models = ['gpt-5', 'o4-mini', 'o3', "gpt-o3-pro"]
    ablation_models = ["o3-pro"]
    # , "gpt-o3", "gpt-o3-pro"
    code_models = ['gpt-5-mini']
    
    dataset_mate_path = "work_dir/dataset/annotation_meta.jsonl"
    tested_idx = get_indexes_from_jsonl(dataset_mate_path)
    if not tested_idx:
        print("No tasks found to run. Exiting.")
        return
    
    for code_model in code_models:
        for ablation_model in ablation_models:
            run_ablation(code_model, ablation_model, tested_idx, args, user_args.max_workers)

if __name__ == '__main__':
    main()
