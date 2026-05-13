import threading

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_local_model = None
_local_tokenizer = None
_model_lock = threading.Lock()


def _get_local_model():
    global _local_model, _local_tokenizer
    with _model_lock:
        if _local_model is None:
            _local_tokenizer = AutoTokenizer.from_pretrained("./gpt2-finetuned")
            _local_model = AutoModelForCausalLM.from_pretrained("./gpt2-finetuned")
            _local_model.eval()
    return _local_model, _local_tokenizer

def _local_infer(text: str, max_tokens: int = 50)-> tuple[str, int]:
    model, tokenizer = _get_local_model()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_tokens,
            pad_token_id = tokenizer.eos_token_id,
            do_sample = True,
            temperature = 0.8
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens = int(outputs.shape[-1])- int(inputs["input_ids"].shape[-1])
    return generated. max(tokens,1)

