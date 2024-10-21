from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
messages = [
{"role": "user", "content": "bbbbbb"},
{"role": "assistant", "content" : "aaaaa"},
{"role": "user", "content" : "bbbb"},
{"role": "assistant", "content" : "aaaaa"},
{"role": "user", "content" : "bbbbb"},
{"role": "assistant", "content" : "aaaaa"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
print(prompt) # Output the formatted prompt