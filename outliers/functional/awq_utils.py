import gc
import importlib
import torch
import functools
import inspect
from outliers.functional.calib_data import get_calib_dataset
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

def clear_memory(weight=None):
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()

def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    elif torch.xpu.is_available():
        return "xpu:0"
    else:
        return "cpu"
    
def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

    
def move_embed(model, device: str):
    model.rotary_emb = model.rotary_emb.to(device)
    model.embed_tokens = model.embed_tokens.to(device)

def init_quant(model, tokenizer, n_samples=128, max_seq_len=512):
    modules = model.model.layers
    samples = get_calib_dataset(
        data="pileval",
        tokenizer=tokenizer,
        n_samples=n_samples,
        max_seq_len=max_seq_len,
        split="train",
        text_column="text",
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    best_device = get_best_device()
    modules[0] = modules[0].to(best_device)
    move_embed(model.model, best_device)

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(model, module):
            super().__init__()
            model.module = module

        def forward(model, *args, **kwargs):
            # assume first input to forward is hidden states
            if len(args) > 0:
                hidden_states = args[0]
                del args
            else:
                first_key = list(kwargs.keys())[0]
                hidden_states = kwargs.pop(first_key)

            inps.append(hidden_states)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    modules[0] = Catcher(modules[0])
    try:
        model.model(samples.to(next(model.model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    modules[0] = modules[0].module  # restore

    # Update the layer kwargs with `prepare_inputs_for_generation` method
    # that takes care of everything to avoid unexpected errors.
    layer_kwargs = model.model.prepare_inputs_for_generation(samples, **layer_kwargs)
    # Pop the input_ids as they are not needed at all.
    layer_kwargs.pop("input_ids")

    del samples
    inps = inps[0]

    modules[0] = modules[0].cpu()
    move_embed(model.model, "cpu")

    clear_memory()

    if layer_kwargs.get("attention_mask") is not None:
        layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
            best_device
        )

    return modules, layer_kwargs, inps

def _get_input_feat(model, layer, named_linears):
    # firstly, get input features of all linear layers
    def cache_input_hook(m, x, y, name, feat_dict):
        x = x[0]
        x = x.detach().cpu()
        feat_dict[name].append(x)

    input_feat = defaultdict(list)
    handles = []

    # # FIXME: Workaround for Mixtral to use block_sparse_moe input features
    # if model.awq_model.model_type == "mixtral":
    #     named_linears = {
    #         **named_linears,
    #         "block_sparse_moe": layer.block_sparse_moe,
    #     }

    # if model.awq_model.model_type == "deepseek_v2":
    #     named_linears = {
    #         **named_linears,
    #         "mlp": layer.mlp,
    #     }

    for name in named_linears:
        handles.append(
            named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
            )
        )
    model.inps = model.inps.to(next(layer.parameters()).device)  # in case multi-gpu
    # get output as next layer's input

    # Sanitize the kwargs in case we use transformers version that contains
    # kwargs that are not handled by the module.
    # Useful for trust_remote_code models.
    module_kwargs = _sanitize_kwargs(model.module_kwargs, layer)

    model.inps = _module_forward(model, model.inps, layer, module_kwargs)
    for h in handles:
        h.remove()
    # now solve for scaling and clipping
    input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

    return input_feat

def _sanitize_kwargs(inputs_kwargs, module):
    """
    Remove the arguments that are not supported in the module's
    forward pass to avoid breaking behaviour between different versions
    of transformers.

    Args:
        inputs_kwargs (`dict`):
            The input dictionary to pass to the model layer
        module (`torch.nn.Module`):
            Target module to quantize.
    """
    module_signature = inspect.signature(module.forward).parameters
    sanitized_kwargs = {}
    for k, v in inputs_kwargs.items():
        if k in module_signature:
            sanitized_kwargs[k] = v
    return sanitized_kwargs

@torch.no_grad()
def apply_clip(module, clip_list):
    for name, max_val in clip_list:
        layer: nn.Linear = get_op_by_name(module, name)
        layer.to(get_best_device())
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()


@torch.no_grad()
def _module_forward(
    model, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
) -> torch.Tensor:
    # if model.n_parallel_calib_samples is None:
    if getattr(model, "n_parallel_calib_samples", None) is None:
        # runs through all samples at once
        module_output = module(x, **module_kwargs)
        if isinstance(module_output, tuple):
            module_output = module_output[0]
    else:
        # memory efficiently runs through all calibration samples
        # but only n_parallel_calib_samples at a time
        module_output = []
        partitioned_inputs = torch.split(x, model.n_parallel_calib_samples)
        for x_partial in partitioned_inputs:
            partial_output = module(x_partial, **module_kwargs)

            if isinstance(partial_output, tuple):
                partial_output = partial_output[0]

            module_output.append(partial_output.cpu())

        module_output = torch.cat(module_output, dim=0)

    return module_output


@torch.no_grad()
def _search_best_clip(model, layer, named_linears, input_feat, group_size, layer_SW):
    clip_list = []
    avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in avoid_clipping]):
            continue

        named_linears[name].to(get_best_device())
        max_val = _compute_best_clip(
            model, named_linears[name].weight, input_feat[name], group_size=group_size,
            layer_SW=layer_SW if "down_proj" in name and len(layer_SW) > 0 else None
        )
        clip_list.append((name, max_val))
        named_linears[name].cpu()

    return clip_list

@torch.no_grad()
def _compute_best_clip(
    model,
    w: torch.Tensor,
    input_feat: torch.Tensor,
    n_grid=20,
    max_shrink=0.5,
    n_sample_token=512,
    group_size=0,
    layer_SW=None,
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = group_size if group_size > 0 else org_w_shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

    # Compute input feature step size (minimum 1)
    step_size = max(1, input_feat.shape[1] // n_sample_token)
    input_feat = input_feat[:, ::step_size]
    
    w = w.reshape(org_w_shape[0], 1, -1, group_size)

    # oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
    oc_batch_size = 64
    assert org_w_shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []
    
    layer_SW_index = []
    if layer_SW:
        for row, col, value in layer_SW:
            ib = row // oc_batch_size
            row_idx = row % oc_batch_size
            layer_SW_index.append((ib, row_idx, col, value))


    for i_b in range(org_w_shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        restore_SW_list = [item for item in layer_SW_index if item[0] == i_b]

        for _, row_idx, col, value in restore_SW_list:
            # TODO: fix group-wise
            w[row_idx, 0, 0, col] = 0

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(model, cur_w)[0]
            for _, row_idx, col, value in restore_SW_list:
                # TODO: fix group-wise
                q_w[row_idx, 0, 0, col] = value
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    clear_memory(input_feat)
    clear_memory(org_out)

    return best_max_val.squeeze(1)

def pseudo_quantize_tensor(self, w: torch.Tensor):
    org_w_shape = w.shape
    if self.group_size > 0:
        assert org_w_shape[-1] % self.group_size == 0
        w = w.reshape(-1, self.group_size)
    else:
        w = w.reshape(-1, org_w_shape[-1])
    assert w.dim() == 2
    assert torch.isnan(w).sum() == 0

    # zero point quantization
    if self.zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        if self.scale_shift:
            # mapping weights to [-0.4999, 15.4999] and then round to nearest integers
            scale = (2 ** self.w_bit - 0.01) / (max_val - min_val)
            w.sub_(min_val).mul_(scale).sub_(0.49)
            w.round_()
            w.add_(0.49).div_(scale).add_(min_val)
        else:
            max_int = 2**self.w_bit - 1
            min_int = 0
            # scales = (max_val - min_val).clamp(min=1e-5) / max_int
            scales = (max_val - min_val) / max_int
            # zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            zeros  = -min_val / scales
            a = (w / scales + zeros).to(torch.int8)
            w = (
                torch.clamp(a + 0.5, min_int, max_int).to(torch.float16) - zeros
            ) * scales
        zeros = zeros.view(org_w_shape[0], -1)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (self.w_bit - 1) - 1
        min_int = -(2 ** (self.w_bit - 1))
        scales = max_val / max_int
        # scales = max_val / min_int
        zeros = None
        w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    scales = scales.view(org_w_shape[0], -1)
    w = w.reshape(org_w_shape)

    return w, scales, zeros
