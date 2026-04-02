import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

    
class DatasetManager:
    
    @staticmethod
    def load_task_info(work_dir, task_id):
        dataset_info_path = os.path.join(work_dir, 'dataset', "annotation_meta.jsonl")
        
        data = []
        with open(dataset_info_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        return data[task_id - 1]
    
    @staticmethod
    def annotation_dir(work_dir, task_id):
        return os.path.join(work_dir, 'dataset','annotations', f'annotation_{task_id}')
    
    @staticmethod
    def code_repo_dir(work_dir, task_id):
        task_info = DatasetManager.load_task_info(work_dir, task_id)
        return os.path.join(DatasetManager.annotation_dir(work_dir, task_id), task_info['repo_dir_name'])
    
    @staticmethod
    def target_file_path(work_dir, task_id):
        task_info = DatasetManager.load_task_info(work_dir, task_id)
        raw_path = os.path.join(DatasetManager.annotation_dir(work_dir, task_id), task_info['repo_dir_name'], task_info['target_file_path'])
        return os.path.relpath(raw_path, DatasetManager.code_repo_dir(work_dir, task_id))
    
    @staticmethod
    def load_task_instruction(work_dir, task_id):
        code_generation_prompt_path = os.path.join(DatasetManager.annotation_dir(work_dir, task_id), 'instruction.txt')
        with open(code_generation_prompt_path, 'r', encoding='utf-8') as f:
            code_generation_prompt = f.read()
        return code_generation_prompt
    
    @staticmethod
    def append_new_code_content(work_dir, task_id, new_code_content):
        task_info = DatasetManager.load_task_info(work_dir, task_id)
        code_file_path = os.path.join(DatasetManager.annotation_dir(work_dir, task_id), task_info['repo_dir_name'], task_info['target_file_path'])
        with open(code_file_path, 'a', encoding='utf-8') as f:
            f.write('\n' + new_code_content + '\n')
        return
    
    @staticmethod
    def xml_report_path(work_dir, task_id):
        return os.path.join(DatasetManager.annotation_dir(work_dir, task_id), 'pytest_report', 'report.xml')

    @staticmethod
    def load_task_latex(work_dir, task_id):
        latex_path = os.path.join(DatasetManager.annotation_dir(work_dir, task_id), 'latex.txt')
        with open(latex_path, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        return latex_content
    
    @staticmethod
    def clean_target_file_content(work_dir, task_id):
        task_info = DatasetManager.load_task_info(work_dir, task_id)
        code_file_path = os.path.join(DatasetManager.annotation_dir(work_dir, task_id), task_info['repo_dir_name'], task_info['target_file_path'])
        code_init_content = DatasetManager.load_task_info(work_dir, task_id)['init_content']
        with open(code_file_path, 'w', encoding='utf-8') as f:
            f.write(code_init_content)
        return

    @staticmethod
    def load_repo_content(work_dir, task_id, file_path):
        task_info = DatasetManager.load_task_info(work_dir, task_id)
        code_file_path = os.path.join(DatasetManager.annotation_dir(work_dir, task_id), task_info['repo_dir_name'], file_path)
        try:
            with open(code_file_path, 'r', encoding='utf-8') as f:
                code_file_content = f.read()
        except:
            code_file_content = "file path error or not readable: " + file_path   
        return code_file_content

    @staticmethod
    def load_code_content(work_dir, task_id):
        task_info = DatasetManager.load_task_info(work_dir, task_id)
        code_file_path = os.path.join(DatasetManager.annotation_dir(work_dir, task_id), task_info['repo_dir_name'], task_info['target_file_path'])
        with open(code_file_path, 'r', encoding='utf-8') as f:
            code_file_content = f.read()
        code_file_content = code_file_content
        lines = code_file_content.splitlines()
        numbered_lines = [f"{i + 1:4d}:{line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)
    
    @staticmethod
    def load_code_raw_content(work_dir, task_id):
        task_info = DatasetManager.load_task_info(work_dir, task_id)
        code_file_path = os.path.join(DatasetManager.annotation_dir(work_dir, task_id), task_info['repo_dir_name'], task_info['target_file_path'])
        with open(code_file_path, 'r', encoding='utf-8') as f:
            code_file_content = f.read()
        code_file_content = code_file_content.strip()
        return code_file_content
    
    @staticmethod
    def load_repo_name(work_dir, task_id):
        info = DatasetManager.load_task_info(work_dir, task_id)
        return info['repo_dir_name']
    
    @staticmethod
    def load_cache_dir(work_dir):
        cache_dir = os.path.join(work_dir, 'cache')
        DatasetManager.make_needed_dir(cache_dir)
        return cache_dir  
        
    @staticmethod
    def make_needed_dir(file_path):
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    @staticmethod    
    def write_code_content(work_dir, task_id, content):
        task_info = DatasetManager.load_task_info(work_dir, task_id)
        code_file_path = os.path.join(DatasetManager.annotation_dir(work_dir, task_id), task_info['repo_dir_name'], task_info['target_file_path'])
        with open(code_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return
        
    @staticmethod
    def load_task_canonical(work_dir, task_id):
        latex_path = os.path.join(DatasetManager.annotation_dir(work_dir, task_id), 'canonical.py')
        with open(latex_path, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        return latex_content
        
def extract_prompt(text: str, word):
    lines = text.split('\n')
    in_block = False
    block_content = []
    blocks = []
    
    for line in lines:
        if line.strip().startswith(f"```{word}"):
            in_block = True
            block_content = []
        elif line.strip() == "```" and in_block:
            in_block = False
            blocks.append('\n'.join(block_content))
        elif in_block:
            block_content.append(line)
    
    return '\n'.join(blocks).strip()

def remove_heading(text, word):
    text = text.strip()
    return text[3+len(word):-3]


def analyze_test_log(log_content):
    """
    Analyzes a unit test execution log (from unittest, pytest, or pre-run Python errors)
    and returns the number of passed and failed test cases,
    distinguishing between test case failures, setup errors, and pre-run errors.

    Args:
        log_content (str): A string containing the entire execution log.

    Returns:
        dict: A dictionary with the following keys:
            'passed_tests' (int): Number of tests that passed.
            'failed_tests' (int): Number of items reported as failed/errored by the runner,
                                  or 1 if a pre-run error occurred.
            'test_case_failures' (list): Failures/errors from within individual test cases.
            'setup_errors' (list): Errors from setup/teardown phases or test collection.
            'pre_run_errors' (list): Fatal Python errors occurring before test framework execution.
    """

    stdout = log_content.stdout.strip()
    stderr = log_content.stderr.strip()
    log_content =  stdout + "\n" + stderr
    passed_tests = 0
    failed_tests = 0
    test_case_failures = []
    setup_errors = []
    pre_run_errors = [] # New category

    # --- General Python Traceback Patterns ---
    traceback_start_pattern = re.compile(r"^Traceback \(most recent call last\):$")
    python_error_line_pattern = re.compile(r"^(?:\s*)([a-zA-Z_][\w\.]*Error|Exception): .*$") # Catches indented or non-indented error lines

    # --- Pytest specific patterns ---
    pytest_passed_pattern = re.compile(r"^(.*?) (PASSED)(\s+\[\s*\d+%\])?$")
    pytest_failed_pattern = re.compile(r"^(.*?) (FAILED)(\s+\[\s*\d+%\])?$")
    pytest_failure_summary_start_pattern = re.compile(r"^=+ FAILURES =+$")
    pytest_failure_detail_start_pattern = re.compile(r"^_*(?: TESTCASE )?([\w\.\:\-\[\]\/<>]+(?:\[.*?\])?) _*")
    pytest_error_summary_start_pattern = re.compile(r"^=+ ERRORS =+$")
    pytest_error_detail_start_pattern = re.compile(r"^_*( ERROR at (?:setup|teardown) of .*? | ERROR collecting .*?) _*")

    # --- Unittest specific patterns ---
    unittest_setup_methods = {"setUpClass", "tearDownClass", "setUpModule", "tearDownModule"}
    unittest_passed_pattern_verbose = re.compile(r"^(test_[\w_]+) \(.*\) \.\.\. ok$")
    unittest_failed_pattern_verbose = re.compile(r"^(test_[\w_]+) \(.*\) \.\.\. (FAIL|ERROR)$")
    unittest_short_result_pattern = re.compile(r"^([\.FEsS]+)$")
    unittest_failure_header_pattern = re.compile(r"^======================================================================")
    unittest_failure_type_pattern = re.compile(r"^(FAIL|ERROR): ([\w_.]+) \((?:.*?)\)$")

    lines = log_content.splitlines()
    in_pytest_failure_summary = False
    in_pytest_error_summary = False
    current_block_pytest_failures = []
    current_block_pytest_errors = []
    current_block_unittest = []

    pytest_summary_pattern = re.compile(
        r"(\d+)\s+(?:passed|deselected)"
        r"(?:,\s*(\d+)\s+failed)?"
        r"(?:,\s*(\d+)\s+error(?:s)?)?"
        r"(?:,\s*(\d+)\s+skipped)?"
        r"(?:,\s*(\d+)\s+xfailed)?"
        r"(?:,\s*(\d+)\s+xpassed)?"
    )
    unittest_summary_ran_pattern = re.compile(r"^Ran (\d+) tests? in .*")
    unittest_summary_ok_pattern = re.compile(r"^OK$")
    unittest_summary_failed_pattern = re.compile(r"^FAILED \((failures=(\d+))?(, )?(errors=(\d+))?\)$")

    total_tests_ran_unittest = 0
    failed_unittest_count_summary = 0
    errors_unittest_count_summary = 0
    processed_pytest_summary = False
    processed_unittest_summary = False

    for line_idx, line in enumerate(reversed(lines)):
        if processed_pytest_summary or processed_unittest_summary:
            break
        if not processed_pytest_summary:
            match_pytest_summary = pytest_summary_pattern.search(line)
            if match_pytest_summary:
                p_passed_str, p_failed_str, p_errors_str = match_pytest_summary.group(1, 2, 3)
                if p_passed_str: passed_tests = int(p_passed_str)
                if p_failed_str: failed_tests += int(p_failed_str)
                if p_errors_str: failed_tests += int(p_errors_str)
                processed_pytest_summary = True; continue
        if not processed_unittest_summary:
            match_unittest_failed_sum = unittest_summary_failed_pattern.search(line)
            if match_unittest_failed_sum:
                f_val, e_val = match_unittest_failed_sum.group(2), match_unittest_failed_sum.group(5)
                if f_val: failed_unittest_count_summary = int(f_val)
                if e_val: errors_unittest_count_summary = int(e_val)
                continue
            match_unittest_ran_sum = unittest_summary_ran_pattern.search(line)
            if match_unittest_ran_sum:
                total_tests_ran_unittest = int(match_unittest_ran_sum.group(1))
                if failed_unittest_count_summary > 0 or errors_unittest_count_summary > 0:
                    failed_tests = failed_unittest_count_summary + errors_unittest_count_summary
                    passed_tests = max(0, total_tests_ran_unittest - failed_tests)
                else:
                    original_ran_idx = len(lines) - 1 - line_idx
                    if original_ran_idx + 1 < len(lines) and unittest_summary_ok_pattern.search(lines[original_ran_idx + 1]):
                        passed_tests = total_tests_ran_unittest; failed_tests = 0
                    elif total_tests_ran_unittest == 0: passed_tests = 0; failed_tests = 0
                    else: passed_tests = total_tests_ran_unittest; failed_tests = 0
                processed_unittest_summary = True; continue
    if processed_unittest_summary and total_tests_ran_unittest == 0 and (failed_unittest_count_summary > 0 or errors_unittest_count_summary > 0):
        failed_tests = failed_unittest_count_summary + errors_unittest_count_summary
        passed_tests = 0

    is_pytest_log = any("pytest" in l or "collected by pytest" in l or "test session starts" in l for l in lines[:20])
    is_unittest_log = any("Ran " in l and "tests in" in l for l in lines[-5:]) or \
                      any("..." in l and ("ok" in l or "FAIL" in l or "ERROR" in l) for l in lines)
    if is_pytest_log and is_unittest_log and not any("pytest" in l or "collected by pytest" for l in lines[:5]):
        is_pytest_log = False

    # Only parse detailed blocks if a test runner was likely involved (summary found or specific markers)
    # or if we haven't found pre-run errors yet.
    if processed_pytest_summary or processed_unittest_summary or not any(traceback_start_pattern.search(l) for l in lines):
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if is_pytest_log:
                if pytest_failure_summary_start_pattern.match(line):
                    in_pytest_failure_summary, in_pytest_error_summary = True, False
                    if current_block_pytest_failures: test_case_failures.append("\n".join(current_block_pytest_failures).strip()); current_block_pytest_failures = []
                    continue
                if pytest_error_summary_start_pattern.match(line):
                    in_pytest_error_summary, in_pytest_failure_summary = True, False
                    if current_block_pytest_errors: setup_errors.append("\n".join(current_block_pytest_errors).strip()); current_block_pytest_errors = []
                    continue
                if pytest_failure_detail_start_pattern.match(line) or (in_pytest_failure_summary and line.startswith("___") and not line.startswith("___ summary ___")):
                    if current_block_pytest_failures: test_case_failures.append("\n".join(current_block_pytest_failures).strip())
                    current_block_pytest_failures = [line]; continue
                elif in_pytest_failure_summary and current_block_pytest_failures and stripped_line and not line.startswith("=+"):
                    current_block_pytest_failures.append(line)
                if pytest_error_detail_start_pattern.match(line) or (in_pytest_error_summary and line.startswith("___") and not line.startswith("___ summary ___")):
                    if current_block_pytest_errors: setup_errors.append("\n".join(current_block_pytest_errors).strip())
                    current_block_pytest_errors = [line]; continue
                elif in_pytest_error_summary and current_block_pytest_errors and stripped_line and not line.startswith("=+"):
                    current_block_pytest_errors.append(line)
                if not processed_pytest_summary and passed_tests == 0 and failed_tests == 0:
                    if pytest_passed_pattern.match(line): passed_tests +=1
                    elif pytest_failed_pattern.match(line):
                        failed_tests +=1; m = pytest_failed_pattern.match(line)
                        if m and not any(m.group(1) in f for f in test_case_failures): test_case_failures.append(f"{m.group(1)} FAILED (Minimal info)")
            elif is_unittest_log:
                if unittest_failure_header_pattern.match(line):
                    if current_block_unittest:
                        block_text = "\n".join(current_block_unittest).strip()
                        first_line = current_block_unittest[0].strip()
                        match_det = unittest_failure_type_pattern.match(first_line)
                        is_setup = match_det and match_det.group(2) in unittest_setup_methods
                        (setup_errors if is_setup else test_case_failures).append(block_text)
                    current_block_unittest = []
                    if i + 1 < len(lines) and unittest_failure_type_pattern.match(lines[i+1].strip()):
                        current_block_unittest.append(lines[i+1].strip())
                    continue
                if current_block_unittest and current_block_unittest[0].strip().startswith(("FAIL:", "ERROR:")):
                    if stripped_line and not unittest_failure_header_pattern.match(line) and \
                       not any(p.match(line) for p in [unittest_summary_ran_pattern, unittest_summary_ok_pattern, unittest_summary_failed_pattern]) and \
                       not (stripped_line.startswith("FAIL:") or stripped_line.startswith("ERROR:")):
                        current_block_unittest.append(line)
                if not processed_unittest_summary and total_tests_ran_unittest == 0:
                    if unittest_passed_pattern_verbose.match(line): passed_tests += 1
                    elif unittest_failed_pattern_verbose.match(line):
                        failed_tests +=1; m = unittest_failed_pattern_verbose.match(line)
                        if m and not any(m.group(1) in f for f in test_case_failures): test_case_failures.append(f"{m.group(1)} ... {m.group(2)} (Minimal info)")
        if current_block_pytest_failures: test_case_failures.append("\n".join(current_block_pytest_failures).strip())
        if current_block_pytest_errors: setup_errors.append("\n".join(current_block_pytest_errors).strip())
        if current_block_unittest:
            block_text = "\n".join(current_block_unittest).strip()
            if block_text and current_block_unittest[0].strip().startswith(("FAIL:", "ERROR:")):
                match_det = unittest_failure_type_pattern.match(current_block_unittest[0].strip())
                is_setup = match_det and match_det.group(2) in unittest_setup_methods
                (setup_errors if is_setup else test_case_failures).append(block_text)

    # Check for pre-run errors if no test runner summary was found and no failures categorized
    if not processed_pytest_summary and not processed_unittest_summary and \
       passed_tests == 0 and failed_tests == 0 and not test_case_failures and not setup_errors:
        in_general_traceback = False
        current_general_traceback_block = []
        for i, line in enumerate(lines):
            if traceback_start_pattern.match(line):
                if current_general_traceback_block: # Should be rare, save previous if starting new
                    pre_run_errors.append("\n".join(current_general_traceback_block))
                current_general_traceback_block = [line]
                in_general_traceback = True
                continue
            if in_general_traceback:
                current_general_traceback_block.append(line)
                # End of traceback conditions
                # 1. Python error line is found
                # 2. Next line is empty or unindented (heuristic for end of block)
                # 3. We unexpectedly hit a test runner pattern (should not happen if pre-run)
                is_last_line_of_log = (i == len(lines) - 1)
                ends_with_error_line = python_error_line_pattern.match(line.strip())
                next_line_suggests_end = (not is_last_line_of_log and \
                                          (lines[i+1].strip() == "" or not lines[i+1].startswith("  ") or not lines[i+1].startswith("\t")) and \
                                          len(current_general_traceback_block) > 1)

                if ends_with_error_line or is_last_line_of_log or \
                   (next_line_suggests_end and any(err_type in line for err_type in ["Error:", "Exception:"])) : # Ensure it's an error block ending
                    pre_run_errors.append("\n".join(current_general_traceback_block))
                    in_general_traceback = False
                    current_general_traceback_block = []
                    # For fatal pre-run errors, one is usually enough to stop everything.
                    # If more are found, they will be appended.
        if in_general_traceback and current_general_traceback_block: # Append any trailing block
            pre_run_errors.append("\n".join(current_general_traceback_block))

    test_case_failures = [f for f in test_case_failures if f.strip()]
    setup_errors = [s for s in setup_errors if s.strip()]
    pre_run_errors = [p for p in pre_run_errors if p.strip()]

    if pre_run_errors and passed_tests == 0 and failed_tests == 0 and not test_case_failures and not setup_errors:
        failed_tests = len(pre_run_errors) # Count each distinct traceback block as a failure point

    # Final reconciliation if no summary counts but blocks were parsed
    if not processed_pytest_summary and not processed_unittest_summary and not pre_run_errors:
        calculated_failures = 0
        seen_item_headers = set()
        for f_block in test_case_failures: seen_item_headers.add(f_block.splitlines()[0] if f_block else "")
        for s_block in setup_errors: seen_item_headers.add(s_block.splitlines()[0] if s_block else "")
        calculated_failures = len(seen_item_headers)
        if failed_tests == 0 and calculated_failures > 0:
            failed_tests = calculated_failures

    return {
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'test_case_failures': test_case_failures,
        'setup_errors': setup_errors,
        'pre_run_errors': pre_run_errors,
        'log_content': log_content.strip()
    }


def analyze_pytest_xml(word_dir, taks_id):
    """
    Analyze a pytest XML report and extract test statistics and failure messages.
    
    Args:
        word_dir: Working directory
        taks_id: Task ID
        
    Returns:
        Dict containing:
            - 'errors': Number of test errors
            - 'failures': Number of test failures  
            - 'skipped': Number of skipped tests
            - 'tests': Total number of tests
            - 'failure_messages': List of detailed failure/error messages
            - 'test_summary': Dictionary mapping test names to pass status (True=passed, False=failed/skipped)
    """
    xml_path = DatasetManager.xml_report_path(word_dir, taks_id)
    try:
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Initialize counters
        errors = 0
        failures = 0
        skipped = 0
        tests = 0
        failure_messages = []
        test_summary = {}
        
        # Find all testcase elements
        for testcase in root.findall('.//testcase'):
            tests += 1
            
            # Get test identification info
            test_name = testcase.get('name', 'unknown')
            test_classname = testcase.get('classname', '')
            
            # Create full test path (similar to pytest format)
            if test_classname:
                # Convert classname format to file path format
                if '.' in test_classname:
                    # Handle nested classes: module.ClassName -> module.py::ClassName
                    parts = test_classname.split('.')
                    if len(parts) > 1:
                        test_path = f"{parts[0]}.py::{'.'.join(parts[1:])}::{test_name}"
                    else:
                        test_path = f"{test_classname}.py::{test_name}"
                else:
                    test_path = f"{test_classname}.py::{test_name}"
            else:
                test_path = f"test.py::{test_name}"
            
            # Check for errors
            error_elem = testcase.find('error')
            if error_elem is not None:
                errors += 1
                # Extract error message for detailed log
                error_msg = f"ERROR in {test_name}: "
                if error_elem.get('message'):
                    error_msg += error_elem.get('message')
                if error_elem.text:
                    error_msg += f"\n{error_elem.text.strip()}"
                failure_messages.append(error_msg)
                
                # Extract brief reason for summary
                brief_reason = "Error"
                if error_elem.get('message'):
                    # Get first line of error message for brief summary
                    brief_reason = error_elem.get('message').split('\n')[0].strip()
                elif error_elem.text:
                    # Extract exception type from error text
                    error_text = error_elem.text.strip()
                    lines = error_text.split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith(' '):
                            brief_reason = line.strip()
                            break
                
                test_summary[test_path] = False
            
            # Check for failures
            failure_elem = testcase.find('failure')
            if failure_elem is not None:
                failures += 1
                # Extract failure message for detailed log
                failure_msg = f"FAILURE in {test_name}: "
                if failure_elem.get('message'):
                    failure_msg += failure_elem.get('message')
                if failure_elem.text:
                    failure_msg += f"\n{failure_elem.text.strip()}"
                failure_messages.append(failure_msg)
                
                # Extract brief reason for summary
                brief_reason = "AssertionError"
                if failure_elem.get('message'):
                    # Get first line of failure message for brief summary
                    brief_reason = failure_elem.get('message').split('\n')[0].strip()
                elif failure_elem.text:
                    # Extract exception type from failure text
                    failure_text = failure_elem.text.strip()
                    lines = failure_text.split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith(' '):
                            brief_reason = line.strip()
                            break
                
                test_summary[test_path] = False
            
            # Check for skipped tests
            skipped_elem = testcase.find('skipped')
            if skipped_elem is not None:
                skipped += 1
                test_summary[test_path] = False  # Treat skipped as not passed
            else:
                # If no error, failure, or skip - it's a passed test
                if error_elem is None and failure_elem is None:
                    test_summary[test_path] = True
        
        # Alternative: get counts from testsuite attributes if available
        testsuite = root.find('testsuite')
        if testsuite is not None:
            # Use testsuite attributes as they're more reliable
            tests = int(testsuite.get('tests', tests))
            errors = int(testsuite.get('errors', errors))
            failures = int(testsuite.get('failures', failures))
            skipped = int(testsuite.get('skipped', skipped))
        
        return {
            'errors': errors,
            'failures': failures,
            'skipped': skipped,
            'tests': tests,
            'failure_messages': failure_messages,
            'test_summary': test_summary,
        }
        
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"XML report file not found: {xml_path}")
    except Exception as e:
        raise RuntimeError(f"Error analyzing pytest XML report: {e}")
    
def parse_code_result(code_result):
    """
    Parse the output from analyze_pytest_xml and return a formatted summary string.
    
    Args:
        code_result (dict): Dictionary containing test results from analyze_pytest_xml
                           Expected keys: 'errors', 'failures', 'skipped', 'tests'
    
    Returns:
        str: Formatted string summarizing test results
        
    Note:
        If errors > 0, it indicates syntax errors, so all tests are considered failed.
    """
    if not isinstance(code_result, dict):
        return "Invalid test result format"
    
    # Extract values with defaults
    errors = code_result.get('errors', 0)
    failures = code_result.get('failures', 0)
    skipped = code_result.get('skipped', 0)
    total_tests = code_result.get('tests', 0)
    
    # If there are syntax errors, all tests should be considered as failures
    if errors > 0:
        failures = total_tests
        passed = 0
        skipped = 0  # Can't skip tests if there are syntax errors
    else:
        # Calculate passed tests normally
        passed = total_tests - failures - skipped
        passed = max(0, passed)  # Ensure non-negative
    
    # Build the summary string
    summary_parts = []
    
    # Add total tests
    summary_parts.append(f"{total_tests} total test{'s' if total_tests != 1 else ''}")
    
    # Add passed tests
    if passed > 0:
        summary_parts.append(f"{passed} passed")
    
    # Add failures
    if failures > 0:
        summary_parts.append(f"{failures} failed")
    
    # Add errors (syntax errors)
    if errors > 0:
        summary_parts.append(f"{errors} syntax error{'s' if errors != 1 else ''}")
    
    # Add skipped (only if no syntax errors)
    if skipped > 0 and errors == 0:
        summary_parts.append(f"{skipped} skipped")
    
    # Join all parts
    if len(summary_parts) <= 1:
        summary = summary_parts[0] if summary_parts else "No test results"
    else:
        summary = ", ".join(summary_parts[:-1]) + f" and {summary_parts[-1]}"
    
    return summary

def build_test_info(result):
    code_status = parse_code_result(result)
    test_summary_str= '\n'.join([f'{test_path}: {"PASS" if test_status else "FAILED"}' for test_path, test_status in result['test_summary'].items()])
    return "Test pass Status:" + str(code_status) + "\nExecution info: " + str(result['failure_messages']) + "\nFailure summary:\n" + test_summary_str

