import os
import json
import pickle


def load_task_ids_from_jsonl(jsonl_path=None):
    if jsonl_path is None:
        jsonl_path = 'work_dir/dataset/annotation_meta.jsonl'
    task_ids = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # 只取 if_test 为 True 的 index
            if item.get("if_test") is True:
                task_ids.append(item.get("index"))
    return task_ids

def eval_success(exec_result):
    return exec_result['errors'] == 0 and exec_result['failures'] == 0 and exec_result['skipped'] == 0



def get_model_mmr(model, task_ids, result_dir=None):
    if result_dir is None:
        result_dir = 'expriment_result'
    level_results = []
    for guidance_level in [0,1,2,3,4]:
        model_result_dir = os.path.join(result_dir, f'memory_agent_{model}_feedback_fixed_guidance_{guidance_level}')
        pass_result = []
        for task_id in task_ids:
            if not os.path.exists(os.path.join(model_result_dir, f'task_{task_id}_result.pkl')):
                continue
            with open(os.path.join(model_result_dir, f'task_{task_id}_result.pkl'), 'rb') as result_file:
                result_data = pickle.load(result_file)
            success_flag = False
            for i in range(len(result_data)):
                exec_result = result_data[i]['execution_result']
                if eval_success(exec_result):
                    success_flag = True
                    pass_result.append(1/(i+1))
                    break
            if not success_flag:
                pass_result.append(0)
        pass_rate = sum(pass_result)/len(pass_result)
        level_results.append(pass_rate)
    return level_results

def get_init_model_mmr(model, task_ids, result_dir=None):
    if result_dir is None:
        result_dir = 'expriment_result'
    level_results = []
    for guidance_level in [0,1,2,3,4]:
        model_result_dir = os.path.join(result_dir, f'memory_agent_{model}_feedback_fixed_guidance_{guidance_level}')
        pass_result = []
        for task_id in task_ids:
            with open(os.path.join(model_result_dir, f'task_{task_id}_result.pkl'), 'rb') as result_file:
                result_data = pickle.load(result_file)
            exec_result = result_data[0]['execution_result']
            if eval_success(exec_result):
                pass_result.append(1)
            else:
                pass_result.append(0)
        pass_rate = sum(pass_result)/len(pass_result)
        level_results.append(pass_rate)
    return level_results

def format_passrates(pass_rates):
    # 保留到千分位 (3 decimal places)
    return [f"{rate:.3f}" for rate in pass_rates]

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
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "deepseek-chat",
            'claude-sonnet-4-20250514'
        ]

        for model in models:
            pass_rates = get_model_mmr(model, task_ids)
            formatted = format_passrates(pass_rates)
            init_pass_rates = get_init_model_mmr(model, task_ids)
            init_formatted = format_passrates(init_pass_rates)
            print(f"{model}: {formatted}")
            
            print('*'*30)
            
            # print(f"{model}: {init_formatted}")
