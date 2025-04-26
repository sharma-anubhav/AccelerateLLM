import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from model_loader import load_models_and_tokenizer
from sequential_utils import DraftGenerator
from sequential_utils import Verifier
from controller import SequentialController
from controller_tree import TreeController
from config import CONFIG

def ensure_logs_dir():
    os.makedirs("logs", exist_ok=True)

def run_sequential(prompt, draft_len, max_new_tokens, tokenizer, target_model, draft_models):
    ensure_logs_dir()
    print(f"\n[Sequential] Prompt='{prompt}' Draft_len={draft_len} Max_new_tokens={max_new_tokens}")

    draft_model_dict = dict(zip(CONFIG["draft_models"], draft_models))
    drafter = DraftGenerator(draft_model_dict, tokenizer)
    verifier = Verifier(target_model, tokenizer)
    controller = SequentialController(
        drafter, verifier, tokenizer,
        max_new_tokens=max_new_tokens,
        draft_length=draft_len,
        verbose=True
    )

    output = controller.generate_text(prompt)
    print(f"Output: {output}")

def run_tree(prompt, draft_model_names, beam_width, beam_depth, max_new_tokens, tokenizer, target_model, draft_models):
    ensure_logs_dir()
    print(f"\n[Tree] Prompt='{prompt}' Beam_width={beam_width} Beam_depth={beam_depth} Max_new_tokens={max_new_tokens}")

    ssm_models = [(tokenizer, model) for model in draft_models]
    controller = TreeController(
        ssm_models, target_model, tokenizer,
        ssm_model_names=draft_model_names,
        max_new_tokens=max_new_tokens,
        beam_width=beam_width,
        beam_depth=beam_depth,
        verbose=True
    )

    output = controller.generate_text(prompt)
    print(f"Output: {output}")

if __name__ == "__main__":
    start = time.time()

    print("[INFO] Loading models once...")
    tokenizer, target_model, draft_models = load_models_and_tokenizer(
        CONFIG["target_model"], CONFIG["draft_models"]
    )

    prompts = CONFIG["prompts"]
    max_tokens_list = CONFIG["max_new_tokens_list"]
    draft_model_names = CONFIG["draft_models"]
    beam_widths = CONFIG["tree_beam_widths"]
    beam_depths = CONFIG["tree_beam_depths"]
    sequential_depths = CONFIG["sequential_draft_lengths"]
    
    for prompt in prompts:
        for draft_len in sequential_depths:
            for max_new_tokens in max_tokens_list:
                run_sequential(prompt, draft_len, max_new_tokens,
                               tokenizer, target_model, draft_models)
    for prompt in prompts:
        for bw in beam_widths:
            for bd in beam_depths:
                for max_new_tokens in max_tokens_list:
                    run_tree(prompt, draft_model_names, beam_width=bw, beam_depth=bd, max_new_tokens=max_new_tokens,
                             tokenizer=tokenizer, target_model=target_model, draft_models=draft_models)
                    


    print(f"\nAll experiments finished in {time.time() - start:.2f} seconds.")
