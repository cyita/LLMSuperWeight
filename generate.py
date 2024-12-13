from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from analyze import remove_outliers

# model_id = "/home/arda/llm-models/llama-7B"
# model_id = "mistralai/Mistral-7B-Instruct-v0.1"
# model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_id = "/home/arda/llm-models/Qwen2-7B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-8B"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

DEFAULT_SYSTEM_PROMPT = """\
    """

def get_prompt(user_input: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    prompt_texts = [f'<|begin_of_text|>']

    if system_prompt != '':
        prompt_texts.append(f'<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>')

    for history_input, history_response in chat_history:
        prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{history_input.strip()}<|eot_id|>')
        prompt_texts.append(f'<|start_header_id|>assistant<|end_header_id|>\n\n{history_response.strip()}<|eot_id|>')

    prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{user_input.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    return ''.join(prompt_texts)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

model = model.eval()

# prompt = "请将下面句子翻译成中文：President Joe Biden and former President Donald Trump will face each other in the U.S. presidential election on Nov. 5."
prompt = "鸡兔同笼，共35只头，94只脚，问鸡兔各多少？"

# prompt = get_prompt(prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant for answering questions."},
    {"role": "user", "content": prompt},
]

# inp = tokenizer("Summer is hot, winter is ", return_tensors="pt").to("cuda")
# inp = tokenizer(messages, return_tensors="pt").to("cuda")
inp = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device=model.device)


gen_config = GenerationConfig(
    max_new_tokens=512,
    # min_new_tokens=100,
    use_cache=True,
    num_beams=1,
    do_sample=False,
)

# sw_value = model.model.layers[2].mlp.down_proj.weight[3968, 7003].item()
# print(sw_value)

# percentage = 1e-7
# remove_outliers(model, None, percentage)

with torch.no_grad():
    # model.model.layers[1].mlp.down_proj.weight[2533, 7890] = 0  # llama 2 7B
    # model.model.layers[1].mlp.down_proj.weight[2070, 7310] = 0  # mistral 7B v0.1
    # model.model.layers[2].mlp.down_proj.weight[3968, 7003] = 0  # llama 1 7B

    # model.model.layers[2].mlp.down_proj.weight[3968, 7003] = sw_value
    # print(model.model.layers[2].mlp.down_proj.weight[3968, 7003].item())
    # model.model.layers[26].mlp.down_proj.weight[458, 5891] = 0

    res = model.generate(inp, generation_config=gen_config)

output_str = tokenizer.decode(res[0], skip_special_tokens=False)
print(output_str)