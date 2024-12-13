from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
import transformers
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from transformers import BitsAndBytesConfig

from lm_eval.models.huggingface import HFLM
from lm_eval import utils
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
import re
from pathlib import Path

from outliers.functional.quantization import quantize_blockwise

eval_logger = utils.eval_logger


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


def get_layer_number(layer_name):
    # Define the pattern to extract the layer number
    match = re.search(r"layers\.(\d+)\.", layer_name)
    if match:
        return int(match.group(1))
    else:
        return None


def get_weight_type(name):
    for linear in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        if linear in name:
            return linear
    return "other"


def pack_4bit_to_int8(quantized_4bit_weights):
    # Ensure the quantized weights are uint8
    quantized_4bit_weights = quantized_4bit_weights.to(torch.uint8)
    
    # Reshape to ensure even number of elements per row
    if quantized_4bit_weights.size(1) % 2 != 0:
        quantized_4bit_weights = torch.cat((quantized_4bit_weights, torch.zeros((quantized_4bit_weights.size(0), 1), dtype=torch.uint8)), dim=1)
    
    # Packing two 4-bit values into one int8
    packed_weights = (quantized_4bit_weights[:, ::2] << 4) | (quantized_4bit_weights[:, 1::2] & 0xF)
    
    return packed_weights


def _get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
    gpus: Optional[int] = None,
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu for device_idx in range(gpus)
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


@register_model("hf-outlier")
class HFOutlierLM(HFLM):
    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        outlier_method: Optional[str] = None, # added argument for outlier experiments
        manual_quantize: Optional[str] = None,
        restore_and_scale_GO: Optional[Union[bool, float]] = False,
        bnb_quantize_args: Optional[str] = None,
        clip_and_save_args: Optional[str] = None,
        load_quantized_args: Optional[str] = None,
        adjust_attn_args: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """

        model_kwargs = kwargs if kwargs else {}

        if parallelize:
            model_kwargs.update(
                _get_accelerate_args(
                    device_map_option,  # TODO: phase out device_map_option?
                    max_memory_per_gpu,
                    max_cpu_memory,
                    offload_folder,
                    gpus,
                )
            )
        elif "device_map" not in model_kwargs:
            # set a device_map to initialize model on the right GPU.
            # this is needed because it seems that the default behavior
            # for quantized models now seems to be device_map="auto"
            # which breaks data-parallel mode.
            if hasattr(self, "accelerator"):
                model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})
            else:
                model_kwargs.update({"device_map": {"": str(self.device)}})

        if not autogptq:
            if model_kwargs.get("load_in_4bit", None):
                assert (
                    transformers.__version__ >= "4.30.0"
                ), "load_in_4bit requires transformers >= 4.30.0"
            if transformers.__version__ >= "4.30.0":
                if model_kwargs.get("load_in_4bit", None):
                    if model_kwargs.get("bnb_4bit_compute_dtype", None):
                        model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(
                            model_kwargs["bnb_4bit_compute_dtype"]
                        )         


            if bnb_quantize_args:
                load_in_4bit, bnb_4bit_quant_type, blocksize, clip_percentage = bnb_quantize_args.split('_') # True_fp4_512_1e-3
                load_in_4bit = True if load_in_4bit == "True" else False
                blocksize = int(blocksize)
                print("bnb args", load_in_4bit, blocksize, bnb_4bit_quant_type)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    blocksize=blocksize
                )
                model_kwargs["load_in_4bit"] = False
                model_kwargs["quantization_config"] = quantization_config    


            model_kwargs.pop("gptqmodel")
            self._model = self.AUTO_MODEL_CLASS.from_pretrained(
                pretrained,
                # revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                use_auth_token="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",
                **model_kwargs,
            )
            # Record SW value
            self.record_GO(pretrained)   
            if adjust_attn_args: # Now is working for Llama-7B only
                import math
                from outliers.functional.utils import ScaledLlamaSdpaAttention
                tau = float(adjust_attn_args)
                scale = 1 / (math.sqrt(self._model.model.layers[3].self_attn.head_dim) * tau) 

                def remove_massive_activation_llama(module, input, output):
                    # print(get_module_name(model, module))
                    hidden_states = output[0][0]
                    # Modify the hidden states here

                    hidden_states[0, 3968] *= tau
                    modified_output = hidden_states
                    output[0][0] = modified_output
                    return output


                self._model.model.layers[3].self_attn = ScaledLlamaSdpaAttention(config=self._model.config, layer_idx=3)

                all_hooks = []
                remove_massive_activation_hook = self._model.model.layers[2].register_forward_hook(remove_massive_activation_llama)
                all_hooks.append(remove_massive_activation_hook)


                self._model = self._model.to('cuda').to(torch.bfloat16) # hard code dtype for now. might need to adjust to arguments


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
                    for name, param in self._model.model.named_parameters():
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
                    for name, param in self._model.model.named_parameters():
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
                    print("Original SO:", self.GO_values)
                    self.restore_GO(pretrained, scaling_factor)

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
                        self._model.model.layers[2].mlp.down_proj.register_forward_hook(hook_fn)
                        print("Hook registered for removing SW and restoring SA")

                elif outlier_method.startswith("search"):
                    _, criterion, search_threhold, layers, outlier_weight_type = outlier_method.split("_") # "search_percentage_1e-4_0-26_downproj"
                    start_layer, end_layer = layers.split('-')
                    layers = range(int(start_layer), int(end_layer))
                    search_threhold = float(search_threhold)
                    # print(layers, search_threhold, criterion)
                    num_selected_elements = []
                    for name, param in self._model.model.named_parameters():
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
                self.restore_GO(pretrained, restore_and_scale_GO)
            else:
                print("Not restoring or scaling GO...")

        else:
            try:
                from auto_gptq import AutoGPTQForCausalLM
            except ModuleNotFoundError:
                raise Exception(
                    "Tried to load auto_gptq, but auto-gptq is not installed ",
                    "please install auto-gptq via pip install lm-eval[gptq] or pip install -e .[gptq]",
                )

            self._model = AutoGPTQForCausalLM.from_quantized(
                pretrained,
                trust_remote_code=trust_remote_code,
                model_basename=None if autogptq is True else Path(autogptq).stem,
                use_safetensors=True
                if autogptq is True
                else autogptq.endswith(".safetensors"),
                **model_kwargs,
            )

        if peft and delta:
            raise ValueError(
                "Cannot use both 'peft' and 'delta' options at the same time."
            )

        if peft:
            if model_kwargs.get("load_in_4bit", None):
                if version.parse(PEFT_VERSION) < version.parse("0.4.0"):
                    raise AssertionError("load_in_4bit requires peft >= 0.4.0")
            if self._model.config.vocab_size != len(self.tokenizer):
                # resize model for LoRAs with added tokens
                self._model.resize_token_embeddings(len(self.tokenizer))
                eval_logger.info(
                    f"Model config indicates vocab_size='{self._model.config.vocab_size}', but found tokenizer with vocab size '{len(self.tokenizer)}'. Resizing model embedding layer..."
                )
            self._model = PeftModel.from_pretrained(
                self._model, peft, revision=revision
            )
        elif delta:
            if autogptq:
                eval_logger.warning(
                    "Delta weights might trigger unexpected behavior when used with AutoGPTQ."
                )
            _model_delta = self.AUTO_MODEL_CLASS.from_pretrained(
                delta,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            for name, param in self._model.state_dict().items():
                try:
                    param.data += _model_delta.state_dict()[name]
                except KeyError:
                    raise KeyError(f"Delta model is missing weights for layer: {name}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to add delta weights to layer {name}. Error: {e}"
                    )

            del _model_delta

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return None


    def record_GO(self, pretrained):
        '''Record GO values for original models'''
        def _record_GO(GO_map, layer, row, col):
            if pretrained in [
                "tiiuae/falcon-7b",
            ]:
                GO_map[(layer, row, col)] = self._model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col].item()
            else:
                GO_map[(layer, row, col)] = self._model.model.layers[layer].mlp.down_proj.weight.data[row, col].item()

        GO_values = {}
        for model_name, coordinates in SUPER_WEIGHTS_MAP.items():
            if model_name in pretrained:
                for layer, row, col in coordinates:
                    _record_GO(GO_values, layer, row, col)
                break
        self.GO_values = GO_values


    def restore_GO(self, pretrained, scaling_factor):
        if pretrained in [
            "huggyllama/llama-30B", 
            "huggyllama/llama-13B", 
            "huggyllama/llama-7B", 
            "mistralai/Mistral-7B-v0.1",
            "meta/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "allenai/OLMo-1B-0724-hf",
            "allenai/OLMo-7B-0724-hf",
            "google/gemma-7b",
            "microsoft/Phi-3-mini-4k-instruct"
            ]:
            if getattr(self._model, "hf_quantizer", None):
                self._model.dequantize()
            for (layer, row, col), value in self.GO_values.items():  
                old_value = self._model.model.layers[layer].mlp.down_proj.weight.data[row, col].item()
                new_value = value * scaling_factor
                self._model.model.layers[layer].mlp.down_proj.weight.data[row, col] = new_value
                print(f"Layer {layer}, Index [{row}, {col}], Old value: {old_value}, New value: {new_value}")

        elif pretrained in [
            "tiiuae/falcon-7b"
            ]:
            if getattr(self._model, "hf_quantizer", None):
                self._model.dequantize()
            for (layer, row, col), value in self.GO_values.items():  
                old_value = self._model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col].item()
                new_value = value * scaling_factor
                self._model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col] = new_value
                print(f"Layer {layer}, Index [{row}, {col}], Old value: {old_value}, New value: {new_value}")
