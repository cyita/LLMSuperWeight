import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from outliers.functional.awq_utils import get_best_device, clear_memory

def evaluate_perplexity(model, tokenizer):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")

    best_device = get_best_device()
    model = model.to(best_device)

    data = data.input_ids.to(model.device)

    seqlen = 2048
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to(model.device)
            with torch.no_grad():
                logits = model(batch).logits
                logits = logits[:, :-1, :]
                clear_memory()
            shift_logits = logits.contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")
            clear_memory(logits, shift_logits, shift_labels, loss)

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()