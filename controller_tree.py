import torch
import time
import json
import os
from datetime import datetime

from tree_utils import (
    TrieNode, run_ssm_generation, normalize_and_dedupe,
    build_token_tree, build_global_tree_mask,
    tree_decode_single_kernel, verify_with_tree_kernel, flatten_tree
)

class TreeController:
    def __init__(self, ssm_models, target_model, tokenizer, ssm_model_names = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"], max_new_tokens=100,
                 beam_width=4, beam_depth=6, verbose=True):
        self.ssm_model_names = ssm_model_names
        self.ssm_models = ssm_models  
        self.target = target_model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.verbose = verbose

    def speculate(self, current_ids):
        all_specs = []
        for tok, mdl in self.ssm_models:
            specs = run_ssm_generation(tok, mdl, current_ids, self.beam_width, self.beam_depth)
            all_specs.extend(specs)

        specs_clean = normalize_and_dedupe(all_specs, self.tokenizer.eos_token_id, self.beam_depth)
        root = build_token_tree(specs_clean, self.tokenizer.eos_token_id)
        return root, specs_clean

    def generate_text(self, prompt):
        eos = self.tokenizer.eos_token_id
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        current_ids = prompt_ids[:]

        stats = {
            "timestamp": str(datetime.now().isoformat()),
            "prompt": prompt,
            "target_model": self.target.name_or_path,
            "draft_models": self.ssm_model_names,
            "beam_width": self.beam_width,
            "beam_depth": self.beam_depth,
            "max_new_tokens": self.max_new_tokens,
            "metrics": {
                "tree_nodes_total": 0,
                "tree_accepted_tokens": 0,
                "tree_bonus_tokens": 0,
                "tree_verification_calls": 0,
                "tree_avg_branching_factor": 0.0,
                "tokens_generated": 0
            },
            "iteration_logs": []
        }

        total_generated = 0
        total_verification_steps = 0
        total_branching = 0
        total_internal_nodes = 0

        while total_generated < self.max_new_tokens:
            iteration_log = {"iteration": total_verification_steps + 1}
            root, clean_specs = self.speculate(current_ids)

            tree_size = sum(1 for _ in flatten_tree(root)[0])
            stats["metrics"]["tree_nodes_total"] += tree_size

            branching = [len(n.children) for n in root.children.values() if len(n.children) > 0]
            total_branching += sum(branching)
            total_internal_nodes += len(branching)

            accepted_tokens, final_ids = verify_with_tree_kernel(
                current_ids[:], root, self.target, self.tokenizer
            )
            total_verification_steps += 1
            current_ids = final_ids

            if not accepted_tokens:
                break

            tree_accepted = sum(1 for t in accepted_tokens if t in flatten_tree(root)[0])
            tree_bonus = len(accepted_tokens) - tree_accepted

            stats["metrics"]["tree_accepted_tokens"] += tree_accepted
            stats["metrics"]["tree_bonus_tokens"] += tree_bonus
            total_generated += len(accepted_tokens)

            iteration_log["accepted_tokens"] = self.tokenizer.decode(accepted_tokens)
            stats["iteration_logs"].append(iteration_log)

            if eos in accepted_tokens:
                try:
                    eos_index = current_ids.index(eos)
                    current_ids = current_ids[:eos_index+1]
                    total_generated = len(current_ids) - len(prompt_ids)
                except ValueError:
                    pass
                break

        stats["metrics"]["tree_verification_calls"] = total_verification_steps
        stats["metrics"]["tokens_generated"] = total_generated
        if total_internal_nodes > 0:
            stats["metrics"]["tree_avg_branching_factor"] = round(
                total_branching / total_internal_nodes, 3)

        stats["final_output"] = self.tokenizer.decode(current_ids[len(prompt_ids):], skip_special_tokens=True)

        os.makedirs("logs", exist_ok=True)
        fname = f"logs/tree_specdraft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n[✓] Tree-based log saved: {fname}")
        return stats["final_output"]
