export LD_LIBRARY_PATH=/home/arda/miniforge3/envs/yina
export HF_ENDPOINT=https://hf-mirror.com

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2-7B-Instruct \
    --tasks arc_challenge,arc_easy,sciq,lambada_openai \
    --device cuda:0 \
    --batch_size 1

# --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
# huggyllama/llama-7B