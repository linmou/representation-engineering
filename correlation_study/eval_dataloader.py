import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, load_dataset
from tqdm import tqdm, trange

class EvalDatasets(Dataset):
    def __init__(self, test_name, prompt_modify_func=None):
        name_to_path = {
            'mmlu':'/Users/butyuhao/Documents/GitHub/machine_psy_great_plan/pilot_study_with_prompt/cache_eval_data/mmlu', # cais/mmlu
            'MATH':'/Users/butyuhao/Documents/GitHub/machine_psy_great_plan/pilot_study_with_prompt/cache_eval_data/MATH', # hendrycks/competition_math
        }
        self.data_path = name_to_path[test_name]
        self.prompt_list = []
        self.target_list = []
        
        if test_name == 'mmlu':
            self._load_mmlu()
        elif test_name == 'MATH':
            self._load_MATH()
            
        self.prompt_modify_func = prompt_modify_func if callable(prompt_modify_func) else lambda question, answer: f'{question} Answer: {answer}'
        assert callable(prompt_modify_func) or prompt_modify_func is None, f"prompt_modify_func must be callable or None, but got {prompt_modify_func}"

    def _load_mmlu(self):
        # load the test set from mmlu

        try:
            ds = load_dataset("cais/mmlu", "all")
        except:
            ds = load_from_disk(self.data_path)

        ds = ds['test']

        self.prompt_list = ds["question"]


        def select_answer(example):
            # 使用索引直接选择答案
            answer_index = example['answer']
            selected_answer = example['choices'][answer_index]
            return {'target': selected_answer}
        
        result = ds.map(select_answer)

        self.target_list = result['target']

    def _load_MATH(self):

        try:
            ds = load_dataset("hendrycks/competition_math")
        except:
            ds = load_from_disk(self.data_path)

        # 定义一个函数来筛选出所需的行
        def filter_levels(example):
            return example['level'] in ['Level 3', 'Level 4', 'Level 5']

        # 应用筛选函数
        ds = ds['test'].filter(filter_levels)

        self.prompt_list = ds['problem']
        self.target_list = ds['solution']
    
    def __len__(self):
        return len(self.prompt_list)
    
    def __getitem__(self, idx):
        question = self.prompt_list[idx]
        answer = self.target_list[idx]
        questionNanswer = self.prompt_modify_func(question=question, answer=answer)
        return questionNanswer

if __name__ == '__main__':
    # # test mmlu
    # d = EvalDatasets('mmlu')
    # for i in range(10):
    #     print(f"datapoint{i}")
    #     print(d[i])

    # test MATH
    d = EvalDatasets('mmlu', prompt_modify_func=lambda x: f'You are a helpful assistant. {x}')
    for i in range(10):
        print(f"datapoint{i}")
        print(d[i])
    
    