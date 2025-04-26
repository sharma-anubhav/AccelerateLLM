import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_available_devices():
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]

def load_model(model_path, device_map=None, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map or "auto"
    )
    if mdl.config.pad_token_id is None:
        mdl.config.pad_token_id = mdl.config.eos_token_id
    return tok, mdl

def load_models_and_tokenizer(target_model_name, draft_model_names):
    devices = get_available_devices()
    print(f"Available devices: {devices}")

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    target_device = devices[0]
    print(f"Loading target model '{target_model_name}' on {target_device}...")
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name)
    target_model.to(target_device)
    target_model.eval()

    draft_models = []
    for idx, draft_name in enumerate(draft_model_names):
        device = devices[min(idx+1, len(devices)-1)]
        print(f"Loading draft model '{draft_name}' on {device}...")
        draft_model = AutoModelForCausalLM.from_pretrained(draft_name)
        draft_model.to(device)
        draft_model.eval()
        draft_models.append(draft_model)

    return tokenizer, target_model, draft_models
