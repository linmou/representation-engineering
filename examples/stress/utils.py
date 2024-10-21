import json
import os
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, PreTrainedTokenizer
import matplotlib.pyplot as plt
import numpy as np
from repe import repe_pipeline_registry
from tqdm import tqdm

repe_pipeline_registry()

def get_rep_reader(data_path, model_name_or_path, hidden_layers, target_emotion, 
                       test_sentence, user_tag="USER:", 
                       assistant_tag="ASSISTANT:", plot=False):
    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token_id = 0

    # Prepare the dataset
    dataset = primary_emotions_concept_dataset(data_path, user_tag, assistant_tag)[target_emotion]

    # Initialize the pipeline
    rep_token = -1
    n_difference = 1
    direction_method = 'pca'
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

    # Extract directions
    rep_reader = rep_reading_pipeline.get_directions(
        dataset['train']['data'], 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=dataset['train']['labels'], 
        direction_method=direction_method,
        batch_size=32,
    )

    # Run tests
    # TODO: last pos or every pos
    chosen_idx = 0
    results = []
    test_token_id = tokenizer(test_sentence, return_tensors="pt")["input_ids"]
    for ice_pos in range(len(test_token_id)):
        ice_pos = -len(test_token_id) + ice_pos
        H_tests = rep_reading_pipeline(
            [test_token_id], 
            rep_reader=rep_reader,
            rep_token=ice_pos,
            hidden_layers=hidden_layers)

    results = {layer: {} for layer in hidden_layers}
    rep_readers_means = {layer: 0 for layer in hidden_layers}

    for layer in hidden_layers:
        H_test = [H[layer] for H in H_tests]
        rep_readers_means[layer] = np.mean(H_test)
        H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]
        
        sign = rep_reader.direction_signs[layer]
        eval_func = min if sign == -1 else max
        cors = np.mean([eval_func(H) == H[0] for H in H_test])
        results[layer] = cors


    if plot:
        # Evaluate and plot results
        plt.plot(hidden_layers, [results[layer] for layer in hidden_layers])
        plt.show()

    return results


# Example usage:
# customize_pipeline(
#     data_path="../../data/facts/facts_true_false.csv",
#     model_name_or_path="ehartford/Wizard-Vicuna-30B-Uncensored",
#     hidden_layers=list(range(-1, -model.config.num_hidden_layers, -1)),
#     test_sentence="Your test sentence here"
# )


def primary_emotions_concept_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise", "stress"]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, f'{emotion}.json')) as file:
            raw_data[emotion] = json.load(file)
            raw_data[emotion] = list(set(json.load(file)))[:200]

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e = raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data
