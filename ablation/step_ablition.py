import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))

import pickle
import subprocess

import json
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import time

from clean_test_runner import run_annotation_tests


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



def get_pytest_case_names(jsonl_file: str) -> List[str]:
    """
    Parse a pytest JUnit-style XML file and return a list of test case names.

    Parameters
    ----------
    xml_path : str
        Path to the XML file produced by `pytest --junitxml=<file>`.

    Returns
    -------
    List[str]
        A list of test case names (the <testcase> 'name' attribute).
        Example items: "test_addition", "test_api_returns_200"
    """
    test_case_name = {}
    with open(jsonl_file) as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue  # skip blank lines
            obj = json.loads(line)  # ✅ parse first

            # ✅ now you can safely access fields
            if not obj.get('if_test'):  
                continue
            test_case_name[obj['index']] = obj['test_cases']
    return test_case_name

def expected_report_path(annotation_index: int) -> str:
    return os.path.join(
        BASE_DIR, f"annotation_{annotation_index}", "pytest_report", "report.xml"
    )

def run_test_script(annotation_index: int) -> Tuple[int, str, bool, str, float]:
    """
    用 login+interactive shell 运行 test.sh；清理旧报告并预建目录。
    返回 (rc, report_path, ran, note, start_ts)
    """
    annotation_dir = os.path.join(BASE_DIR, f"annotation_{annotation_index}")
    test_sh_path = os.path.join(annotation_dir, "test.sh")
    report_path = expected_report_path(annotation_index)
    stdout_log = os.path.join(annotation_dir, "_runner_stdout.txt")
    stderr_log = os.path.join(annotation_dir, "_runner_stderr.txt")

    if not os.path.isfile(test_sh_path):
        return -999, report_path, False, "test.sh not found", 0.0

    # 清理旧报告 + 预建目录
    try:
        if os.path.isfile(report_path):
            os.remove(report_path)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
    except Exception:
        pass

    # 确保脚本可执行
    try:
        st = os.stat(test_sh_path)
        os.chmod(test_sh_path, st.st_mode | 0o111)
    except Exception:
        pass

    cmd = "set -e; ./test.sh"  # 在 annotation 目录下执行
    start_ts = time.time()
    try:
        completed = subprocess.run(
            ["/bin/bash", "-il", "-c", cmd],  # login + interactive，加载环境（conda/pytest等）
            cwd=annotation_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_SEC,
            text=True
        )
        rc = completed.returncode

        # 持久化日志
        with open(stdout_log, "w", encoding="utf-8") as f:
            f.write(completed.stdout or "")
        with open(stderr_log, "w", encoding="utf-8") as f:
            f.write(completed.stderr or "")

        note = ""
        err_lower = (completed.stderr or "").lower()
        if rc == 127 or "command not found" in err_lower or "not found" in err_lower:
            note = "possible PATH/env issue (pytest/conda/python not found)"

        if rc != 0:
            tail_stdout = "\n".join((completed.stdout or "").splitlines()[-20:])
            tail_stderr = "\n".join((completed.stderr or "").splitlines()[-20:])
            print(f"--- stdout (tail) annotation_{annotation_index} ---\n{tail_stdout}")
            print(f"--- stderr (tail) annotation_{annotation_index} ---\n{tail_stderr}")

        return rc, report_path, True, note, start_ts
    except subprocess.TimeoutExpired:
        with open(stdout_log, "w", encoding="utf-8") as f:
            f.write(f"[TIMEOUT > {TIMEOUT_SEC}s] partial stdout unavailable\n")
        with open(stderr_log, "w", encoding="utf-8") as f:
            f.write(f"[TIMEOUT > {TIMEOUT_SEC}s] partial stderr unavailable\n")
        return -998, report_path, True, f"timeout > {TIMEOUT_SEC}s", start_ts
    except Exception as e:
        with open(stderr_log, "w", encoding="utf-8") as f:
            f.write(f"[exec error] {e}\n")
        return -997, report_path, True, f"exec error: {e}", start_ts

jsonl_path = "work_dir/dataset/annotation_meta.jsonl"  # adjust to your path
task_ids = load_task_ids_from_jsonl(jsonl_path)


test_case_name = get_pytest_case_names(jsonl_path)
    

work_dir = os.environ.get("WORK_DIR", "work_dir")
code_model = 'deepseek-chat'
BASE_DIR = os.path.join(work_dir, 'dataset', "annotations")
TIMEOUT_SEC = 600  # 单个 annotation 测试超时（秒）

from action import *
class Arg:
    pass

args = Arg
args.max_steps = 5
args.work_dir = work_dir
args.task_id = 32  # set to the task id you want to replay
args.api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
args.api_provider = 'azure-openai'
args.model = 'gpt-4o-mini'
args.deployment = 'gpt-4o-mini-2'
args.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
args.api_version = "2024-12-01-preview"
args.max_turn = 5
args.human_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
args.human_api_provider = 'azure-openai'
args.feedback_mode = 'constant'
args.human_deployment = 'o1-mini'
args.human_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
args.human_api_version = "2024-12-01-preview"
args.human_agent_model = 'o1-mini'
args.guidance_level = 4
args.guidance_type = 'fixed'
args.evaluation_setting = 'baseline'


action_number_list = []
jsonl_path = os.path.join(work_dir, 'dataset', 'annotation_meta.jsonl') # 这里改成你的 jsonl 路径
task_ids = load_task_ids_from_jsonl(jsonl_path)
for task_id in task_ids:
    print(f'process task {task_id}')
    taks_dir = os.path.join(jsonl_path, 'dataset', 'annotations',f'annotation_{task_id}')
    args.task_id = task_id
    my_handler = ActionHandler(args, task_id)
    DatasetManager.clean_target_file_content(work_dir, task_id)
    current_result = None
    test_case_num = len(test_case_name[task_id])
    exp_file = f'expriment_result/memory_agent_{code_model}_feedback_fixed_guidance_1/task_{task_id}_result.pkl'
    with open(exp_file, "rb") as f:
        data = pickle.load(f)
    for turn_index in range(10):
        print(f'turn {turn_index} of task {task_id}')
        if turn_index >= len(data):
            break
        else:
            action_result = []
            for step_number in range(10):
                print(f'step: {step_number}, turn {turn_index} of task {task_id}')
                turn_data = data[turn_index]
                if step_number * 2 + 1 > len(turn_data['code_interaction_history']):
                    break
                interaction_replay_content = turn_data['code_interaction_history'][step_number * 2 + 1]['content']
                my_handler.process_command(interaction_replay_content)
                annotation_dir = DatasetManager.annotation_dir(work_dir, task_id)
                run_annotation_tests(work_dir, task_id, timeout_sec=TIMEOUT_SEC)
                # run_test_script(task_id)
                parsed_result = analyze_pytest_xml(work_dir, task_id)
                if parsed_result['errors'] != 0:
                    current_result = 0
                else:
                    current_result = (test_case_num - parsed_result['failures'])/test_case_num
                action_result.append(current_result)
            action_number_list.append(action_result)
with open(f"turn_success_info_{code_model}.pkl", "wb") as f:
    pickle.dump(action_number_list, f)