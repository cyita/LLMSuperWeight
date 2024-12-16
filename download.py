from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

# model_path = "facebook/opt-30b"
# model_path = "meta-llama/Llama-2-13b-chat-hf"
# model_path = "THUDM/chatglm3-6b"
# model_path = "Qwen/Qwen-14B-Chat"
# model_path = "EleutherAI/gpt-j-6b"
# model_path = "microsoft/Phi-3-medium-4k-instruct"
# model_path = "bartowski/Phi-3-medium-4k-instruct-GGUF"
# model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_path = "Qwen/Qwen1.5-7B-Chat"
# model_path = "mistralai/Mistral-7B-Instruct-v0.2"
# model_path="meta-llama/Meta-Llama-3-8B"
model_path = "Qwen/Qwen2-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             use_auth_token="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",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          use_auth_token="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",
                                          trust_remote_code=True)

# snapshot_download(
#   repo_id=model_path,
#   local_dir="/home/arda/llm-models/llama-7B",
#   use_auth_token="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",
#   ignore_patterns="*.bin",
#   max_workers=8,
#   local_dir_use_symlinks=False
# )
