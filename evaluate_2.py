from lm_eval.__main__ import cli_evaluate
import outliers.model  # support hf-outliers model by importing into namespace
import torch, gc

if __name__ == '__main__':
    cli_evaluate()
    gc.collect()
    torch.cuda.empty_cache()