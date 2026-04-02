import os
import json
import pickle
from metric_passrate_unitest import load_task_ids_from_jsonl, format_passrates

def get_test_case_num(task_id, mate_data_path=None):
    if mate_data_path is None:
        mate_data_path = 'work_dir/dataset/annotation_meta.jsonl'
    with open(mate_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get("index") == task_id:
                    return len(record.get('test_cases'))
    raise Exception()

def get_pass_num(exec_result):
    if exec_result['errors'] != 0:
        return 0
    else:
        return exec_result['tests'] - exec_result['skipped'] - exec_result['failures']
    
def get_model_passrates(model, task_ids, result_dir=None):
    if result_dir is None:
        result_dir = 'expriment_result'
    level_results = []
    for guidance_level in [0,1,2,3,4]:
        model_result_dir = os.path.join(result_dir, f'memory_agent_{model}_feedback_fixed_guidance_{guidance_level}')
        pass_result = []
        for task_id in task_ids:
            test_case_num = get_test_case_num(task_id)
            if not os.path.exists(os.path.join(model_result_dir, f'task_{task_id}_result.pkl')):
                continue
            with open(os.path.join(model_result_dir, f'task_{task_id}_result.pkl'), 'rb') as result_file:
                result_data = pickle.load(result_file)
            pass_nums = []
            for i in range(len(result_data)):
                exec_result = result_data[i]['execution_result']
                pass_num = get_pass_num(exec_result)
                pass_nums.append(pass_num/test_case_num)
            pass_result.append(max(pass_nums))

        pass_rate = sum(pass_result)/len(pass_result)
        level_results.append(pass_rate)
    return level_results


def get_model_passrates_turn(model, task_ids, result_dir=None):
    if result_dir is None:
        result_dir = 'expriment_result'
    level_results = []
    for guidance_level in [0,1,2,3,4]:
        model_result_dir = os.path.join(result_dir, f'memory_agent_{model}_feedback_fixed_guidance_{guidance_level}')
        pass_result = []
        for task_id in task_ids:
            task_pass_status = []
            test_case_num = get_test_case_num(task_id)
            if not os.path.exists(os.path.join(model_result_dir, f'task_{task_id}_result.pkl')):
                continue
            with open(os.path.join(model_result_dir, f'task_{task_id}_result.pkl'), 'rb') as result_file:
                result_data = pickle.load(result_file)
            pass_nums = []
            for i in range(len(result_data)):
                exec_result = result_data[i]['execution_result']
                pass_num = get_pass_num(exec_result)
                pass_nums.append(pass_num/test_case_num)
            pass_result.append(task_pass_status)

        level_results.append(pass_result)
    return level_results

def get_init_model_passrates(model, task_ids, result_dir=None):
    if result_dir is None:
        result_dir = 'expriment_result'
    level_results = []
    for guidance_level in [0,1,2,3,4]:
        model_result_dir = os.path.join(result_dir, f'memory_agent_{model}_feedback_fixed_guidance_{guidance_level}')
        pass_result = []
        for task_id in task_ids:
            test_case_num = get_test_case_num(task_id)
            if not os.path.exists(os.path.join(model_result_dir, f'task_{task_id}_result.pkl')):
                continue
            with open(os.path.join(model_result_dir, f'task_{task_id}_result.pkl'), 'rb') as result_file:
                result_data = pickle.load(result_file)
            exec_result = result_data[0]['execution_result']
            pass_num = get_pass_num(exec_result)
            pass_result.append(pass_num/test_case_num)

        pass_rate = sum(pass_result)/len(pass_result)
        level_results.append(pass_rate)
    return level_results

if __name__ == '__main__':
    # task_ids = load_task_ids_from_jsonl()
    task_id_map = {2023:[1, 3, 23, 24, 25, 29, 41, 43, 44, 76, 81, 89, 91, 92, 99, 103, 104],
                   2024:[2, 5, 6, 7, 8, 10, 12, 14, 15, 16, 17, 18 , 20, 21, 22, 26, 27, 28, 30, 32, 33, 34, 36, 42, 47, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 62, 63, 67, 68, 69, 60, 71, 78, 80, 82, 83, 85, 87, 88, 90, 98, 100, 101, 102 ],
                   2025:[4, 9, 13, 19, 31, 35, 38, 39, 40, 45, 46, 48, 49, 55, 61, 64, 65, 66, 72, 73, 74, 75, 77, 84, 86, 93,94,95,96,97]}
    for key in task_id_map.keys():
        task_ids = task_id_map[key]
        print(f"==== Year {key} ====")
        models = [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "deepseek-chat",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            'claude-sonnet-4-20250514'
        ]

        for model in models:
            # pass_rates = get_model_passrates(model, task_ids)
            pass_rates = get_model_passrates(model, task_ids)
            
            formatted = format_passrates(pass_rates)
            init_pass_rates = get_init_model_passrates(model, task_ids)
            init_formatted = format_passrates(init_pass_rates)
            print(f"{model}: {formatted}")
            
            print('*'*30)
            
            # print(f"{model}: {init_formatted}")