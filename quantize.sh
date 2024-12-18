export LD_LIBRARY_PATH=/home/arda/miniforge3/envs/yina
export HF_ENDPOINT=https://hf-mirror.com

# python analyze.py plot-down-proj-input-output /home/arda/llm-models/llama-7B
# python prepare_model.py create_model 

# model_name=meta-llama/Meta-Llama-3-8B-Instruct
model_name=Qwen/Qwen2-7B-Instruct

# tasks=wikitext,arc_challenge,arc_easy,sciq,lambada_openai
# tasks=arc_challenge,arc_easy,sciq,lambada_openai
tasks=arc_challenge

# clip_4_${blocksize}_no_9_False_False

#----------------- Baseline -----------------
# python evaluate_2.py --model hf-outlier \
#             --model_args pretrained=${model_name},trust_remote_code=True,dtype=float16,run_ppl=True \
#             --tasks ${tasks} \
#             --device cuda:0 \
#             --batch_size 1 \
#             --output_path outputs/${model_name}/baseline \
#             --trust_remote_code

for blocksize in 128
do
    restore_and_scale_GO=False
    scale_shift=False
    for manual_quantize in clip_4_${blocksize}_no_9_${blocksize}_False
    # awq_4_${blocksize}_no_9_${scale_shift}_False clip_4_${blocksize}_z_9_${scale_shift}_False clip_4_${blocksize}_z_11_${scale_shift}_False clip_4_${blocksize}_tp_1e-6_${scale_shift}_False
    do
        echo "---------------------------------------"
        # echo "python prepare_model.py compare ${model_name} --trust-remote-code --restore-and-scale-g-o --manual-quantize ${manual_quantize}"
        # python prepare_model.py compare ${model_name} --trust-remote-code --restore-and-scale-g-o --quantize --manual-quantize ${manual_quantize} 
        # > /home/arda/yina/LLMSuperWeight/acc_logs/Meta-Llama-3-8B_manual_${manual_quantize}.log

        python evaluate_2.py --model hf-outlier \
            --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16,run_ppl=True \
            --tasks ${tasks} \
            --device cuda:0 \
            --batch_size 1 \
            --trust_remote_code \
            # --output_path outputs/${model_name}/${blocksize}/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO} 
    done
done    