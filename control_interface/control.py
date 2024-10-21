from datetime import datetime
import json
import pickle
import torch
from transformers import pipeline

from control_interface.utils import load_model_tokenizer, all_emotion_rep_reader, primary_emotions_concept_dataset, dict_to_unique_code
from repe import repe_pipeline_registry
from control_interface.exp_settings import *
from control_interface.prompt_formats import PromptFormat

repe_pipeline_registry()


# LLaMA-2-Chat-13B coeff=3.0-3.5
emotion = "happiness"
emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise",]
# emotions = ["stress"]
data_dir = "/home/jjl7137/representation-engineering/data/emotions"
model_name_or_path = 'meta-llama/Llama-2-13b-chat-hf'
# model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name_or_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.3"
prompt_format = PromptFormat.get(model_name_or_path)
user_tag =  prompt_format.user_tag # "[INST]"
assistant_tag = prompt_format.assistant_tag #"[/INST]"

model, tokenizer = load_model_tokenizer(model_name_or_path,
                                        user_tag=user_tag, 
                                        assistant_tag=assistant_tag,
                                        expand_vocab=False)

data = primary_emotions_concept_dataset(data_dir, user_tag=user_tag, 
                                        assistant_tag=assistant_tag,
                                        )

rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'

args = {
    'emotions': emotions,
    'data_dir': data_dir,
    'model_name_or_path': model_name_or_path,
    'user_tag': user_tag,
    'assistant_tag': assistant_tag,
    'rep_token': rep_token,
    'hidden_layers': hidden_layers,
    'n_difference': n_difference,
    'direction_method': direction_method,
}

arg_codes = dict_to_unique_code(args)

coeffs = [-2, -1, 0, 1, 2,]
max_new_tokens=512

block_name="decoder_block"
control_method="reading_vec"

rebuild = False
emotion_rep_readers = None
try:
    emotion_rep_readers = pickle.load(open(f'/home/jjl7137/representation-engineering/exp_records/emotion_rep_reader_{arg_codes[:10]}.pkl', 'rb'))
except:
    pass
if emotion_rep_readers is None or (emotion_rep_readers and emotion_rep_readers.get('args') != args) or rebuild:
    rep_reading_pipeline = pipeline( "rep-reading", model=model, tokenizer=tokenizer)
    emotion_rep_readers =  all_emotion_rep_reader(data, emotions, rep_reading_pipeline, 
                                                hidden_layers, rep_token, n_difference, 
                                                direction_method, read_args = args, save_path=f'exp_records/emotion_rep_reader_{arg_codes[:10]}.pkl')

rep_reader = emotion_rep_readers[emotion]

acc_threshold = 0.
control_layer_id = list(range(-11, -30, -1)) # for llama TODO: figure out why -11 to -30
# control_layer_id = list(range(-5, -18, -1)) # for mistral 7b, llama 3.1 8b
# control_layer_id = hidden_layers

rep_control_pipeline =  pipeline(
    "rep-control", 
    model=model, 
    tokenizer=tokenizer, 
    layers=control_layer_id, 
    block_name=block_name, 
    control_method=control_method)

# experiment = system_prompt_johnny_donation
# experiment = system_prompt_feeling_describe
# experiment = system_prompt_product_choice
# experiment = system_prompt_first_impression
# experiment = system_prompt_moral_judgement_disgust
experiment = system_prompt_overwhelmingTasks_happy
# analyse_prompt = experiment['analyse_prompt']

inputs = []
for ipt_id in range(len(experiment['user_messages'])):
    ipt = experiment['user_messages'][ipt_id]
    assert type(ipt) == str
    inputs.append( prompt_format.build(experiment['system_prompt'], [ipt]))


outputs = {}
for coeff in coeffs:
    activations = {}
    for layer in hidden_layers:
        activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(model.device).half()
    
    control_outputs = rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens,  do_sample=True, top_p=0.95, ) # top_p=0.95,

    for idx, (i, p) in enumerate(zip(inputs, control_outputs)):
        # print("===== No Control =====")
        # print(s[0]['generated_text'].replace(i, ""))
        print(f"===== + {emotion} Control coeff {coeff} id {idx} =====")
        generated = p[0]['generated_text'].replace(i, "")
        print(generated)
        print()
        outputs[f"{emotion}_{coeff}_id_{idx}"] = generated
        
# load output to json
date_time = datetime.now().strftime("%m-%d_%H-%M")
exp_name = experiment['experiment_name']
with open(f"exp_records/{exp_name}_{emotion}_{arg_codes[:10]}_{date_time}.json", 'w') as f:
    json.dump(outputs, f, indent=4)

    