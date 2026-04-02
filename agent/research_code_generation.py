#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import os
# Add repo root to path so that top-level packages (retrieval/, metrics/) are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import traceback
import json
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Project-specific imports (left as wildcards to match your current structure)
from common_imports import *  
from agents import *          
from utils import *           


class ResearchCodeGenerationWorkflow:
    """
    Orchestrates baseline/agent/memory-agent code generation workflows,
    including optional replay, persistence, and parallel execution helpers.
    """

    def __init__(self, args: Any) -> None:
        # Core config from YAML
        self.api_key: Optional[str] = args.api_key
        self.api_provider: Optional[str] = args.api_provider
        self.evaluation_setting: Optional[str] = args.evaluation_setting
        self.replay_feedback_level: Optional[int] = args.replay_feedback_level
        self.resume: bool = bool(args.resume)
        self.print_cost: bool = True
        self.max_turn: int = int(args.max_turn)
        self.init_seed: str = getattr(args, "init_seed", "seeds")
        self.model_name: str = getattr(args, "model", "unknown_model")
        self.base_output_dir = args.output_dir
        
        # Paths / dirs
        self.init_result_dir(args)

        # Agents
        self.code_agent = CodeAgent(args)
        self.human_agent = HumanAgent(args)

        # Thread-safe file operations (NOTE: not process-safe; see comment in dump_result)
        self.file_lock = Lock()

    def _seed_pkl_path(self) -> str:
        return os.path.join(
        self.base_output_dir, self.init_seed, f"{self.model_name}_task{self.code_agent.task_id}_init.pkl"
    )
        
    def _ensure_seed_dir(self) -> None:
        seed_dir = os.path.dirname(self._seed_pkl_path())
        os.makedirs(seed_dir, exist_ok=True)

    # ---------- Shared helpers ----------

    def update_task_id(self, task_id: int) -> None:
        """Propagate task id to all agents to track current task."""
        self.code_agent.update_task_id(task_id)
        self.human_agent.update_task_id(task_id)

    def init_result_dir(self, args: Any) -> None:
        """
        Create a deterministic output-dir name based on the config knobs.
        """
        feedback_setting = args.guidance_type
        feedback_guidance_level = args.guidance_level
        evaluation_model = args.model
        evaluation_setting = args.evaluation_setting

        dir_name = (
            f"{evaluation_setting}_{evaluation_model}"
            f"_feedback_{feedback_setting}_guidance_{feedback_guidance_level}"
        )
        base_output_dir = args.output_dir
        output_dir = os.path.join(base_output_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    # ---------- Persistence ----------

    def dump_result(self, new_record: Dict[str, Any], output_file_path: Optional[str] = None) -> None:
        """
        Append a record (pickle list) with thread-safety.

        NOTE: This uses a threading.Lock. If you run with ProcessPoolExecutor,
        this is *not* cross-process safe. For process safety, write per-task files
        and merge later, or use a file-based lock (e.g., portalocker) across processes.
        """
        if output_file_path is None:
            output_file_path = os.path.join(self.output_dir, f"task_{self.code_agent.task_id}_result.pkl")

        with self.file_lock:
            if os.path.exists(output_file_path):
                with open(output_file_path, "rb") as f:
                    existing_data = pickle.load(f)
            else:
                existing_data = []
            existing_data.append(new_record)
            with open(output_file_path, "wb") as f:
                pickle.dump(existing_data, f)

    def record_data_baseline(self, code_result: Dict[str, Any], human_feedback_result: Dict[str, Any]) -> None:
        """
        Persist a baseline turn.
        """
        new_record = {
            "task_index": self.code_agent.task_id,
            "turn_number": code_result["turn_number"],
            "code_input_tokens": code_result["input_token"],
            "code_output_tokens": code_result["output_token"],
            "final_code": code_result["final_code"],
            "code_interaction_history": code_result["interaction_history"],
            "execution_result": human_feedback_result["execution_result"],
            "feedback_input_token": human_feedback_result["input_token"],
            "feedback_output_token": human_feedback_result["output_token"],
            "feedback_interaction_history": human_feedback_result["interaction_history"],
        }
        self.dump_result(new_record)

    def recored_data_agent(  # keeping your original name to avoid breaking callers
        self,
        code_agent_data: Dict[str, Any],
        human_agent_data: Dict[str, Any],
        output_file_path: Optional[str] = None,
    ) -> None:
        """
        Persist an agent turn (with memory/feedback).
        """
        new_record = {
            "task_index": self.code_agent.task_id,
            "turn_number": code_agent_data["turn_number"],
            "code_input_tokens": code_agent_data["input_token"],
            "code_output_tokens": code_agent_data["output_token"],
            "final_code": code_agent_data["final_code"],
            "agent_memory": code_agent_data["memory"],
            "code_interaction_history": code_agent_data["interaction_history"],
            "execution_result": human_agent_data["execution_result"],
            "feedback_input_token": human_agent_data["input_token"],
            "feedback_output_token": human_agent_data["output_token"],
            "feedback_interaction_history": human_agent_data["interaction_history"],
        }
        self.dump_result(new_record, output_file_path)

    # ---------- Modes ----------

    def run(self, task_id: int) -> None:
        """
        Route to the selected evaluation setting.
        """
        if not self.resume and os.path.exists(os.path.join(self.output_dir, f"task_{task_id}_result.pkl")):
            return
        DatasetManager.clean_target_file_content(self.code_agent.work_dir, task_id)
        self.update_task_id(task_id)
        if self.replay_feedback_level is not None:
            self.run_replay()
        elif self.evaluation_setting == "baseline":
            self.run_baseline()
        elif self.evaluation_setting == "agent":
            self.run_agent()
        elif self.evaluation_setting == "memory_agent":
            self.run_memory_agent()
        else:
            raise ValueError(
                f"Unknown evaluation setting: {self.evaluation_setting}. "
                "Supported settings are 'baseline', 'agent', and 'memory_agent'."
            )

    def run_replay(self) -> None:
        """
        Replays prior results, re-feeding feedback at a chosen level.
        """
        result_file_path = os.path.join(self.output_dir, f"task_{self.code_agent.task_id}_result.pkl")
        output_file_path = os.path.join(
            self.output_dir, f"task_{self.code_agent.task_id}_result_replay_level{self.replay_feedback_level}.pkl"
        )

        if not os.path.exists(result_file_path):
            raise ValueError(f"Replay generation needs existing result pkl: {result_file_path}")

        with open(result_file_path, "rb") as f:
            data = pickle.load(f)

        for i in range(1, len(data)):
            execution_result = data[i - 1]["execution_result"]
            feedbacks = data[i - 1]["feedback_interaction_history"][1]["content"]
            feedback_content = format_feedback(feedbacks, self.replay_feedback_level, "fixed")

            self.code_agent.set_history(data[i - 1]["agent_memory"])
            DatasetManager.write_code_content(
                self.code_agent.work_dir, self.code_agent.task_id, data[i - 1]["final_code"]
            )

            code_agent_data = self.code_agent.generate_agent_history(
                feedback=feedback_content, result=execution_result, turn_num=i
            )
            human_agent_data, _, _ = self.human_agent.execute_feedback(False)

            self.recored_data_agent(code_agent_data, human_agent_data, output_file_path)
            DatasetManager.clean_target_file_content(self.code_agent.work_dir, self.code_agent.task_id)

    def run_memory_agent(self) -> None:
        """
        Agent flow that persists/uses memory, with optional resume.
        """
        feedback: Optional[str] = None
        if_success = False
        human_agent_data: Optional[Dict[str, Any]] = None
        cur_turn = 0
        
        if self.resume:
            result_file_path = os.path.join(self.output_dir, f"task_{self.code_agent.task_id}_result.pkl")
            if os.path.exists(result_file_path):
                with open(result_file_path, "rb") as f:
                    data = pickle.load(f)
                cur_turn = data[-1]["turn_number"] + 1
                if cur_turn >= self.max_turn:
                    return
                parsed_result = data[-1]["execution_result"]
                if parsed_result['errors'] == 0 and parsed_result['failures'] == 0 and parsed_result['skipped'] == 0:
                    return
                self.code_agent.set_history(data[-1]["agent_memory"])
                DatasetManager.write_code_content(
                    self.code_agent.work_dir, self.code_agent.task_id, data[-1]["final_code"]
                )
                human_agent_data =  {"execution_result": parsed_result}
                

        if cur_turn == 0:
            seed_path = self._seed_pkl_path()
            human_agent_data = {}
            if os.path.exists(seed_path):
                # Load the shared per-model seed: same initial code & RAW feedback
                with open(seed_path, "rb") as f:
                    seed = pickle.load(f)
                init_result = seed[0]['execution_result']
                shutil.copy(seed_path, os.path.join(self.output_dir, f"task_{self.code_agent.task_id}_result.pkl"))
                if init_result['errors'] == 0 and init_result['failures'] == 0 and init_result['skipped'] == 0:
                    return
                # 1) Write the initial code into the work dir for this task
                DatasetManager.write_code_content(self.code_agent.work_dir, self.code_agent.task_id, seed[0]["final_code"])
                # 2) Seed the agent's memory as "empty start" (fresh history)
                self.code_agent.set_history(seed[0]["agent_memory"])  # or self.code_agent.reset_history() if you have one
                # 3) Prepare formatted feedback for turn 1 using our current guidance level
                feedback = self.human_agent.parse_feedback(seed[0]["feedback_interaction_history"][1]["content"])
                # 4) We *start* iterations at turn 1 now (turn 0 was the shared seed)
                human_agent_data["execution_result"] = seed[0]["execution_result"]
                cur_turn = 1
            else:
                # Create the seed: run turn 0 generation + RAW feedback and persist it.
                # Turn 0: generate initial code (no prior feedback/result)
                code_agent_data = self.code_agent.generate_agent_history(
                    feedback=None, result=None, turn_num=0
                )
                # Execute tests and get RAW feedback (unformatted)
                human_agent_data, if_success, feedback = self.human_agent.execute_feedback(
                    cur_turn != self.max_turn - 1
                )
                # Record turn 0 like normal
                self._ensure_seed_dir()
                self.recored_data_agent(code_agent_data, human_agent_data, seed_path)
                self.recored_data_agent(code_agent_data, human_agent_data)
                # Persist the per-model seed so other settings/runs reuse it
                cur_turn = 1
        
        for i in range(cur_turn, self.max_turn):
            execution_result = human_agent_data["execution_result"] if human_agent_data is not None else None

            code_agent_data = self.code_agent.generate_agent_history(
                feedback=feedback, result=execution_result, turn_num=i
            )

            human_agent_data, if_success, feedback = self.human_agent.execute_feedback(
                i != self.max_turn - 1
            )

            self.recored_data_agent(code_agent_data, human_agent_data)

            if if_success:
                break
        if code_agent_data['final_code'] == '':
            print(f'empty final code of task id: {self.code_agent.task_id} with gudance of {self.guidance_level}')
            exit()
        DatasetManager.clean_target_file_content(self.code_agent.work_dir, self.code_agent.task_id)

    def run_baseline(self) -> None:
        """
        Baseline generation loop without agent memory.
        """
        feedback: Optional[str] = None
        for i in range(self.max_turn):
            code_result = self.code_agent.generate_llm(feedback)
            human_agent_data, if_success, feedback = self.human_agent.execute_feedback(
                i != self.max_turn - 1
            )
            self.record_data_baseline(code_result, human_agent_data)
            if if_success:
                break

        DatasetManager.clean_target_file_content(self.code_agent.work_dir, self.code_agent.task_id)

    def run_agent(self) -> None:
        """
        Agent flow (no resume/memory restore at start).
        """
        feedback: Optional[str] = None
        for i in range(self.max_turn):
            # NOTE: original code called `generate_conversition_history` (typo).
            # Using `generate_agent_history` for consistency with other paths.
            code_agent_data = self.code_agent.generate_agent_history(feedback=feedback, turn_num=i)
            human_agent_data, if_success, feedback = self.human_agent.execute_feedback(
                i != self.max_turn - 1
            )
            self.recored_data_agent(code_agent_data, human_agent_data)
            if if_success:
                break

        DatasetManager.clean_target_file_content(self.code_agent.work_dir, self.code_agent.task_id)


# ---------- Parallel helpers ----------

def create_workflow_instance(args: Any) -> ResearchCodeGenerationWorkflow:
    """Create a new workflow instance (used by executors)."""
    return ResearchCodeGenerationWorkflow(args)


def run_single_task(args: Any, task_id: int) -> Tuple[int, str]:
    """Run one task (callable for executor pools)."""
    try:
        print(f"Starting task {task_id}")
        workflow = create_workflow_instance(args)
        workflow.run(task_id)
        print(f"Completed task {task_id}")
        return task_id, "Success"
    except Exception as e:
        print(f"Error in task {task_id}: {str(e)}")
        return task_id, f"Error: {str(e)}"


def run_tasks_parallel_threads(args: Any, task_ids: List[int], max_workers: int = 4) -> List[Tuple[int, str]]:
    """Run tasks in parallel using threads."""
    results: List[Tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_single_task, args, task_id): task_id for task_id in task_ids}
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Task {task_id} completed: {result[1]}")
            except Exception as exc:
                print(f"Task {task_id} generated an exception: {exc}")
                results.append((task_id, f"Exception: {exc}"))
    return results

def run_tasks_sequential(args: Any, task_ids: List[int]) -> List[Tuple[int, str]]:
    """Run tasks one by one, printing full tracebacks on failure."""
    results: List[Tuple[int, str]] = []
    for task_id in task_ids:
        print(f"Starting task {task_id}")
        try:
            workflow = create_workflow_instance(args)
            workflow.run(task_id)
            print(f"Completed task {task_id}")
            results.append((task_id, "Success"))
        except Exception as e:
            # Print a full traceback immediately
            tb = traceback.format_exc()
            print(f"Error in task {task_id}: {e}\n{tb}")
            results.append((task_id, f"Error:\n{tb}"))
    return results

def run_tasks_parallel_processes(args: Any, task_ids: List[int], max_workers: int = 4) -> List[Tuple[int, str]]:
    """
    Run tasks in parallel using processes.
    NOTE: dump_result uses a thread lock only. If multiple processes write
    to the same file, use per-task files and merge after, or add a process-safe file lock.
    """
    results: List[Tuple[int, str]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_single_task, args, task_id): task_id for task_id in task_ids}
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Task {task_id} completed: {result[1]}")
            except Exception as exc:
                print(f"Task {task_id} generated an exception: {exc}")
                results.append((task_id, f"Exception: {exc}"))
    return results


# ---------- CLI / Config ----------

def parse_arguments() -> argparse.Namespace:
    """
    Parse CLI args that control *runner* behavior; model/workflow config comes from YAML.
    """
    parser = argparse.ArgumentParser(
        description="Load YAML configuration and run code-generation workflow."
    )
    parser.add_argument(
        "--yaml-location",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--parallel-mode",
        type=str,
        choices=["threads", "processes", "none"],
        default="processes",
        help="Parallel execution mode (default: processes). threads may cause GPU memory lock contention.",
    )
    return parser.parse_args()


class YamlDataHolder:
    """Simple bag for YAML-derived attributes."""

    def __init__(self) -> None:
        pass

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
    args.task_ids = yaml_data.get("task-ids", None)

    return args


def load_task_ids_from_jsonl(jsonl_path: str) -> List[int]:
    """
    Load task indices from a JSONL file, keeping only rows with if_test == True.
    Expected schema per line: {"index": <int>, "if_test": <bool>, ...}
    """
    task_ids: List[int] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("if_test") is True:
                task_ids.append(int(item.get("index")))
    return task_ids


def main() -> None:
    user_args = parse_arguments()
    args = parse_yaml(user_args.yaml_location)

    # Read task ids from JSONL dataset meta
    jsonl_path = "work_dir/dataset/annotation_meta.jsonl"  # adjust to your path
    task_ids = args.task_ids if args.task_ids is not None else load_task_ids_from_jsonl(jsonl_path)
    print(f"Running {len(task_ids)} tasks in parallel with {user_args.max_workers} workers")
    print(f"Using {user_args.parallel_mode} for parallel execution")

    if user_args.parallel_mode == "threads":
        results = run_tasks_parallel_threads(args, task_ids, user_args.max_workers)
    elif user_args.parallel_mode in ("processes", "process"):
        results = run_tasks_parallel_processes(args, task_ids, user_args.max_workers)
    else:
        os.environ.setdefault("PYTHONFAULTHANDLER", "1")
        results = run_tasks_sequential(args, task_ids)
        
    # Summary
    print("\n" + "=" * 50)
    print("EXECUTION SUMMARY")
    print("=" * 50)
    successful_tasks = [r for r in results if r[1] == "Success"]
    failed_tasks = [r for r in results if r[1] != "Success"]
    print(f"Total tasks: {len(task_ids)}")
    print(f"Successful: {len(successful_tasks)}")
    print(f"Failed: {len(failed_tasks)}")

    if failed_tasks:
        print("\nFailed tasks:")
        for task_id, error in failed_tasks:
            print(f"  Task {task_id}: {error}")


if __name__ == "__main__":
    main()
