from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List, Literal, Optional, Tuple, Union
import torch
from outliers.functional.quantization import quantize_blockwise
from clize import run

SUPER_WEIGHTS_MAP = {
    "Mistral-7B-v0.1": [(1, 2070, 7310)],
    "llama-7B": [(2, 3968, 7003)],
    "llama-13B": [(2, 2231, 2278), (2, 2231, 6939)],
    "llama-30B": [(3, 5633, 12817), (3, 5633, 17439), (10, 5633, 14386)],
    "Meta-Llama-3-8B": [(1, 788, 2427), (1, 1384, 2427), (1, 4062, 2427)],
    "meta-llama/Meta-Llama-3-8B-Instruct": [(1, 788, 2427), (1, 1384, 2427), (1, 4062, 2427)],
    "OLMo-1B-0724-hf": [(1, 1764, 1710), (2, 1764, 8041)],
    "OLMo-7B-0724-hf": [(1, 269, 7467), (2, 269, 8275), (7, 269, 453), (24, 269, 2300)],
    "Phi-3-mini-4k-instruct": [(2, 525, 808), (2, 1693, 808), (2, 1113, 808), (4, 525, 2723),  (4, 1113, 2723), (4, 1693, 2723)],
}

promptList = [
"给我推荐丽江三日游，包括景点和路线",
"请说出以下两句话区别在哪里？冬天能穿多少穿多少，夏天能穿多少穿多少",
"鸡兔同笼，共35只头，94只脚，问鸡兔各多少？",
"请将下面句子翻译成中文：President Joe Biden and former President Donald Trump will face each other in the U.S. presidential election on Nov. 5.",
"Python中的遍历该怎么写？","创作两句中国诗歌，关于春天和秋天。",
"如果所有猫都有尾巴，那么一只没有尾巴的动物是否可以被称为猫",
"土豆发芽了还能吃吗",
"如何评价特斯拉发布的iPhone 15？",
"我今天心情很糟糕，感觉一切都不顺利。",
"蓝牙耳机坏了需要看医院的哪个科室？",
'解释这行代码：parttern = (  \
"Intel\(R\)\s+(?:Arc\(TM\)\s+)?Graphics"   \
if inference_device == "ultra" \
else "Intel\(R\) Arc\(TM\) [^ ]+ Graphics"\
)',
"左手一只鸭，右手一只鸡。交换两次后左右手里各是什么？交换三次后左右手里各是什么？",
"以月为主题，写一首七言绝句诗",
"使用python实现读入图片，并将图片按长边尺寸resize成正方形",
"将以下中文翻译成英文：穿衣需要适应天气变化，夏天你能穿多少穿多少，冬天你能穿多少穿多少。",
"写一篇英文散文诗，主题是春雨，想象自己是春雨，和英国古代诗人莎士比亚交流",
"10.11 和 10.9 哪个大？",
"介绍一下北京烤鸭。150字左右",
"问题1：推荐三种深圳美食 \
问题2：介绍第2个",
"请将下面句子翻译成中文：The ship that my sister said that the  owner of the company claimed that the inspector had certified as seaworthy   sank in the Pacific.",
"甲、乙、丙三人比赛象棋，每两人赛一盘。胜一盘得2分。平一盘得1分，输一盘得0分。比赛的全部三盘下完后，只出现一盘平局。并且甲得3分，乙得2分，丙得1分。那么，甲乙，甲丙，乙丙（填胜、平、负）。",
"我读了一篇小小说，只有一千字，是一位上海女作家写的，后来那篇小小说得了奖。你写的这篇也一千字，你也是一位上海女作家，也一定能得奖。这个说法对吗？",
"我有两个苹果，然后我又买了两个，我用其中两个苹果烤了一个派，吃掉了一半派后，我还剩几个苹果呢？",
"请给出“海水朝朝朝朝朝朝朝落”的下联，并解释所给出下联的含义",
'用Python编程计算邮费。计算规则如下： \
根据邮件的重量和用户选择是否加急计算邮费。\
重量在1000 以内（包括），基本费8 元；\
超过1000 克的部分，每500 克加收超重费4 元，不足500克部分按500克计算；\
如果用户选择加急，多收5元。\
输入格式：\
一行，包含一个正整数x（大于1小于10e6）和一个字符c(取值为y或n)，之间用一个空格隔开，分别表示重量和是否加急。\
如果字符是   y，说明选择加急；如果字符是   n，说明不加急。\
输出格式：\
输出一行一个正整数，表示邮费',
"def create_multipliers():\
return [lambda x: i* x for i  in range(5)]\
    \
for multiplier in create_multipliers():\
print(multiplier(2))#Python中输出结果是什么",
"以月为主题，写一首诗"
]

def record_GO(model, pretrained):
    '''Record GO values for original models'''
    def _record_GO(GO_map, layer, row, col):
        if pretrained in [
            "tiiuae/falcon-7b",
        ]:
            GO_map[(layer, row, col)] = model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col].item()
        else:
            GO_map[(layer, row, col)] = model.model.layers[layer].mlp.down_proj.weight.data[row, col].item()

    GO_values = {}
    for model_name, coordinates in SUPER_WEIGHTS_MAP.items():
        if model_name in pretrained:
            for layer, row, col in coordinates:
                _record_GO(GO_values, layer, row, col)
            break
    model.GO_values = GO_values


def restore_GO(model, pretrained, scaling_factor):
    if pretrained in [
        "huggyllama/llama-30B", 
        "huggyllama/llama-13B", 
        "huggyllama/llama-7B", 
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "allenai/OLMo-1B-0724-hf",
        "allenai/OLMo-7B-0724-hf",
        "google/gemma-7b",
        "microsoft/Phi-3-mini-4k-instruct"
        ]:
        if getattr(model, "hf_quantizer", None):
            model.dequantize()
        for (layer, row, col), value in model.GO_values.items():  
            old_value = model.model.layers[layer].mlp.down_proj.weight.data[row, col].item()
            new_value = value * scaling_factor
            model.model.layers[layer].mlp.down_proj.weight.data[row, col] = new_value
            print(f"Layer {layer}, Index [{row}, {col}], Old value: {old_value}, New value: {new_value}")

    elif pretrained in [
        "tiiuae/falcon-7b"
        ]:
        if getattr(model, "hf_quantizer", None):
            model.dequantize()
        for (layer, row, col), value in model.GO_values.items():  
            old_value = model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col].item()
            new_value = value * scaling_factor
            model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col] = new_value
            print(f"Layer {layer}, Index [{row}, {col}], Old value: {old_value}, New value: {new_value}")


def compare(pretrained: str,
            *,
            quantize=False,
            trust_remote_code=False,
            outlier_method = None, # added argument for outlier experiments
            manual_quantize = None,
            restore_and_scale_GO=False,
            adjust_attn_args = None):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained,
        torch_dtype=torch.float16,
        use_auth_token="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",
        trust_remote_code=trust_remote_code,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_auth_token="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")
    model = model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=756,
        # min_new_tokens=100,
        use_cache=True,
        num_beams=1,
        do_sample=False,
    )

    if quantize:
        create_model(pretrained, model, trust_remote_code, outlier_method, manual_quantize, restore_and_scale_GO, adjust_attn_args)
    
    with torch.no_grad():
        for p in promptList:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for answering questions."},
                {"role": "user", "content": p},
            ]

            inp = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device=model.device)
            # model.model.layers[1].mlp.down_proj.weight[2533, 7890] = 0  # llama 2 7B
            # model.model.layers[1].mlp.down_proj.weight[2070, 7310] = 0  # mistral 7B v0.1
            # model.model.layers[2].mlp.down_proj.weight[3968, 7003] = 0  # llama 1 7B

            # model.model.layers[2].mlp.down_proj.weight[3968, 7003] = sw_value
            # print(model.model.layers[2].mlp.down_proj.weight[3968, 7003].item())
            # model.model.layers[26].mlp.down_proj.weight[458, 5891] = 0

            print("-" * 20, "Output", "-" * 20)
            res = model.generate(inp, generation_config=gen_config)
            output_str = tokenizer.decode(res[0], skip_special_tokens=False)
            print(output_str)

    # prompt = "请将下面句子翻译成中文：President Joe Biden and former President Donald Trump will face each other in the U.S. presidential election on Nov. 5."
    # # prompt = "鸡兔同笼，共35只头，94只脚，问鸡兔各多少？"

    # messages = [
    #     {"role": "system", "content": "You are a helpful AI assistant for answering questions."},
    #     {"role": "user", "content": prompt},
    # ]

    # inp = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     return_tensors="pt"
    # ).to(device=model.device)
    
    # # inp = tokenizer("请将下面句子翻译成中文：President Joe Biden and former President Donald Trump will face each other in the U.S. presidential election on Nov. 5.", return_tensors="pt").to("cuda")


    # with torch.no_grad():
    #     # model.model.layers[1].mlp.down_proj.weight[2533, 7890] = 0  # llama 2 7B
    #     # model.model.layers[1].mlp.down_proj.weight[2070, 7310] = 0  # mistral 7B v0.1
    #     # model.model.layers[2].mlp.down_proj.weight[3968, 7003] = 0  # llama 1 7B

    #     # model.model.layers[2].mlp.down_proj.weight[3968, 7003] = sw_value
    #     # print(model.model.layers[2].mlp.down_proj.weight[3968, 7003].item())
    #     # model.model.layers[26].mlp.down_proj.weight[458, 5891] = 0

    #     res = model.generate(inp, generation_config=gen_config)

    # output_str = tokenizer.decode(res[0], skip_special_tokens=False)
    # print(output_str)

    # create_model(pretrained, model, trust_remote_code, outlier_method, manual_quantize, restore_and_scale_GO, adjust_attn_args)
    
    # with torch.no_grad():
    #     # model.model.layers[1].mlp.down_proj.weight[2533, 7890] = 0  # llama 2 7B
    #     # model.model.layers[1].mlp.down_proj.weight[2070, 7310] = 0  # mistral 7B v0.1
    #     # model.model.layers[2].mlp.down_proj.weight[3968, 7003] = 0  # llama 1 7B

    #     # model.model.layers[2].mlp.down_proj.weight[3968, 7003] = sw_value
    #     # print(model.model.layers[2].mlp.down_proj.weight[3968, 7003].item())
    #     # model.model.layers[26].mlp.down_proj.weight[458, 5891] = 0

    #     res = model.generate(inp, generation_config=gen_config)
    # output_str = tokenizer.decode(res[0], skip_special_tokens=False)
    # print(output_str)


def create_model(
        pretrained: str,
        # *, 
        model,
        trust_remote_code=False,
        outlier_method = None, # added argument for outlier experiments
        manual_quantize = None,
        restore_and_scale_GO=False,
        adjust_attn_args = None,
    ) -> None:

    # Record SW value
    record_GO(model, pretrained)   
    if adjust_attn_args: # Now is working for Llama-7B only
        import math
        from outliers.functional.utils import ScaledLlamaSdpaAttention
        tau = float(adjust_attn_args)
        scale = 1 / (math.sqrt(model.model.layers[3].self_attn.head_dim) * tau) 

        def remove_massive_activation_llama(module, input, output):
            # print(get_module_name(model, module))
            hidden_states = output[0][0]
            # Modify the hidden states here

            hidden_states[0, 3968] *= tau
            modified_output = hidden_states
            output[0][0] = modified_output
            return output


        model.model.layers[3].self_attn = ScaledLlamaSdpaAttention(config=model.config, layer_idx=3)

        all_hooks = []
        remove_massive_activation_hook = model.model.layers[2].register_forward_hook(remove_massive_activation_llama)
        all_hooks.append(remove_massive_activation_hook)


        model = model.to('cuda').to(torch.bfloat16) # hard code dtype for now. might need to adjust to arguments


    if manual_quantize:
        # Parse the quantization method and parameters from the string
        quantize_method, bits, block_size_arg, clip_method, clip_threshold, scale_shift, use_normal_float = manual_quantize.split('_') # minmax_4_128_bp_1e-4
        bits = int(bits)

        block_size_arg = int(block_size_arg) if block_size_arg not in ["channel", "tensor"] else block_size_arg
        clip_threshold = float(clip_threshold) if clip_threshold != "None" else None 
        clip_method_map = {"bp": "block_percentage", "tp": "tensor_percentage", "z": "zscore", "iqr": "iqr",  "no": "no"}
        clip_method = clip_method_map[clip_method]
        scale_shift = True if scale_shift == "True" else False
        use_normal_float = True if use_normal_float == "True" else False

        if quantize_method == "minmax":
            print(f"Running {bits} bits manual quantization ...")
            for name, param in model.model.named_parameters():
                if not name.endswith("weight"):
                    continue
                if "layernorm" in name or "norm" in name or "embed_tokens" in name or "lm_head" in name:
                    continue
                weight = param.data
                shape = weight.shape
                if block_size_arg == "channel":
                    block_size = shape[-1]
                elif block_size_arg == "tensor":
                    block_size = weight.numel()
                elif isinstance(block_size_arg, int):
                    block_size = block_size_arg
                else:
                    raise ValueError("Block size must be int or 'tensor' or 'channel'")
                quantized_weight, _num_outliers = quantize_blockwise(weight, bits, block_size, "no", 0, scale_shift, use_normal_float)
                param.data = quantized_weight    
        
        elif quantize_method == "clip":
            # clip outliers but restore super outliers
            # quantize with min max
            num_outliers = []
            original_value_ranges = []
            clipped_value_ranges = []

            print(f"Running {bits} bits manual quantization ...")
            print(f"Clipping parameters", clip_method, clip_threshold)
            print(f"Restore GO?", restore_and_scale_GO)
            for name, param in model.model.named_parameters():
                if not name.endswith("weight"):
                    continue
                if "layernorm" in name or "norm" in name or "embed_tokens" in name or "lm_head" in name:
                    continue

                weight = param.data
                shape = weight.shape
                if block_size_arg == "channel":
                    block_size = shape[-1]
                elif block_size_arg == "tensor":
                    block_size = weight.numel()
                elif isinstance(block_size_arg, int):
                    block_size = block_size_arg
                else:
                    raise ValueError("Block size must be int or 'tensor' or 'channel'") 
                if weight.numel() % (block_size) != 0:
                    print(name,  weight.numel(), block_size)
                quantized_weight, _num_outliers = quantize_blockwise(weight, bits, block_size, clip_method, clip_threshold, scale_shift, use_normal_float)
                param.data = quantized_weight    
                num_outliers.append(_num_outliers)
            # print statistics
            print(f"Number of outliers. Max: {max(num_outliers)}, Min: {min(num_outliers)}, Sum: {sum(num_outliers)}, Mean {sum(num_outliers) / len(num_outliers)}")


    if outlier_method:
        def get_weight_type(model_id, name):
            if model_id.startswith("facebook/opt"):
                for linear in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "fc1", "fc2"]:
                    if linear in name:
                        return linear   
            else:

                for linear in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                    if linear in name:
                        return linear
            return "other"
        
        def get_layer_number(layer_name):
            # Define the pattern to extract the layer number
            match = re.search(r"layers\.(\d+)\.", layer_name)
            if match:
                return int(match.group(1))
            else:
                return None
        if outlier_method.startswith("manual_scaling_SO"): # for example, manual_scaling_SO_0.1
            print("manual scaling SO")
            scaling_factor = outlier_method.split('_')[-1]
            scaling_factor = float(scaling_factor)
            print("Original SO:", model.GO_values)
            restore_GO(model, pretrained, scaling_factor)

        if outlier_method.startswith("removeSW_restoreSA"):
            if pretrained == "huggyllama/llama-7B":
                def hook_fn(module, input, output):
                    SW_channel = 3968
                    SW_row = 7003
                    # Record original hidden states
                    X1 = output.detach().clone()
                    # print(f"Shape of input/output: {output.shape}, {input[0].shape}")
                    
                    # Find the largest magnitude in channel SA and its position
                    channel_SA = X1[..., SW_channel]
                    # print(f"Shape of channel_SA: {channel_SA.shape}")
                    max_magnitude_indices = torch.argmax(torch.abs(channel_SA), dim=1)
                    # print(f"Max magnitude indices: {max_magnitude_indices}")
                    max_magnitudes = torch.gather(channel_SA, 1, max_magnitude_indices.unsqueeze(1)).squeeze(1)

                    
                    # Temporarily set SW to 0
                    weight = module.weight.clone()
                    original_weight = weight.data[SW_channel, SW_row].item()
                    weight.data[SW_channel, SW_row] = 0.0
                    
                    # Recompute hidden states
                    with torch.no_grad():
                        X2 = torch.nn.functional.linear(input[0], weight, module.bias)
                        # print(f"Shape of recomputed X2: {X2.shape}")
                    
                        batch_indices = torch.arange(X2.size(0), device=X2.device)
                        # print("New magnitude on SA position:", X2[batch_indices, max_magnitude_indices, 3968])
                        X2[batch_indices, max_magnitude_indices, SW_channel] = max_magnitudes

                    # Restore the original weight
                    module.weight.data[SW_channel, SW_row] = original_weight
                    return X2

                # Register the hook to the specific layer
                model.model.layers[2].mlp.down_proj.register_forward_hook(hook_fn)
                print("Hook registered for removing SW and restoring SA")

        elif outlier_method.startswith("search"):
            _, criterion, search_threhold, layers, outlier_weight_type = outlier_method.split("_") # "search_percentage_1e-4_0-26_downproj"
            start_layer, end_layer = layers.split('-')
            layers = range(int(start_layer), int(end_layer))
            search_threhold = float(search_threhold)
            # print(layers, search_threhold, criterion)
            num_selected_elements = []
            for name, param in model.model.named_parameters():
                if not name.endswith("weight"):
                    continue
                if "layernorm" in name or "norm" in name or "embed_tokens" in name or "lm_head" in name:
                    continue
                if outlier_weight_type != "all":
                    if outlier_weight_type == "down":
                        outlier_weight_type = "down_proj"
                    weight_type = get_weight_type(pretrained, name)
                    if weight_type != outlier_weight_type:
                        # print("Skip for unmatched weight type")
                        continue
                if layers is not None:
                    layer_number = get_layer_number(name)
                    if layer_number not in layers:
                        # print("Skip for unmatched layer")
                        continue
                weight = param.data
                if criterion == "percentage":
                    num_top_elements = int(weight.numel() * search_threhold)
                    # print("# Total params:", weight.numel(), "# top params", num_top_elements)

                    if num_top_elements < 1: # too few elements to apply on
                        continue 
                    threshold = torch.topk(weight.view(-1).abs(), num_top_elements).values[-1]
                    mask = weight.abs() >= threshold
                    true_indices = mask.nonzero(as_tuple=False)
                    num_selected_elements.append(len(true_indices))
                    # print(name, len(true_indices))
                    
                    weight[mask] = 0.
                param = torch.nn.Parameter(weight) 
            if len(num_selected_elements) > 0:
                print(f"Number of removed elements. Max: {max(num_selected_elements)}, Min: {min(num_selected_elements)}, Sum: {sum(num_selected_elements)}, Mean {sum(num_selected_elements) / len(num_selected_elements)}")


    if restore_and_scale_GO:
        restore_GO(model, pretrained, restore_and_scale_GO)
    else:
        print("Not restoring or scaling GO...")

if __name__ == '__main__':
    run([compare])
