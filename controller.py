import torch
import json
import os
from datetime import datetime

class SequentialController:
    def __init__(self, drafter, verifier, tokenizer, max_new_tokens=100, draft_length=5, verbose=True):
        self.drafter = drafter
        self.verifier = verifier
        self.tokenizer = tokenizer
        self.max_new = max_new_tokens
        self.draft_len = draft_length
        self.verbose = verbose

    def generate_text(self, prompt: str):
        stats = {
            "timestamp": str(datetime.now().isoformat()),
            "prompt": prompt,
            "target_model": self.verifier.target_model.name_or_path,
            "draft_models": list(self.drafter.draft_models.keys()),
            "max_new_tokens": self.max_new,
            "draft_length": self.draft_len,
            "metrics": {
                "total_iterations": 0,
                "draft_tokens_proposed": 0,
                "draft_tokens_accepted": 0,
                "corrections_by_target": 0,
                "tokens_generated": 0,
                "per_draft_metrics": [],
                "target_generate_calls": 0,
                "target_saves": 0},
            "iteration_logs": []
        }

        enc = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        prefix_ids, prefix_mask = enc.input_ids, enc.attention_mask
        output_ids = prefix_ids.clone()
        eos_id = self.tokenizer.eos_token_id
        generated = 0
        per_draft_acc = {k: 0 for k in self.drafter.draft_models}
        per_draft_prop = {k: 0 for k in self.drafter.draft_models}

        while generated < self.max_new:
            stats["metrics"]["total_iterations"] += 1
            drafts = self.drafter.generate_drafts(prefix_ids, prefix_mask, self.draft_len)
            accepts, corrs = {}, {}

            iter_log = {"iteration": stats["metrics"]["total_iterations"], "draft_proposed": {}, "accepted_text": {}, "correction": {}}

            for name, dseq in drafts.items():
                acc, corr = self.verifier.verify_sequence(prefix_ids, dseq.unsqueeze(0))
                accepts[name], corrs[name] = acc, corr
                prop_txt = self.tokenizer.decode(dseq.tolist(), skip_special_tokens=True)
                acc_txt = self.tokenizer.decode(dseq[:acc].tolist(), skip_special_tokens=True)
                iter_log["draft_proposed"][name] = prop_txt
                iter_log["accepted_text"][name] = acc_txt
                per_draft_prop[name] += dseq.size(0)

            best = max(accepts, key=accepts.get)
            acc, corr, dseq = accepts[best], corrs[best], drafts[best]
            per_draft_acc[best] += acc

            if corr is not None:
                iter_log["correction"][best] = {
                    "rejected": self.tokenizer.decode([dseq[acc].item()]),
                    "corrected_with": self.tokenizer.decode([corr])
                }

            if (dseq[:acc] == eos_id).any():
                stats["metrics"]["draft_tokens_accepted"] += acc
                stats["metrics"]["target_saves"] += acc
                output_ids = torch.cat([output_ids, dseq[:acc].unsqueeze(0).to(output_ids.device)], dim=1)
                prefix_ids, prefix_mask = output_ids.clone(), torch.ones_like(output_ids)
                generated += acc
                break
            elif acc > 0:
                stats["metrics"]["draft_tokens_accepted"] += acc
                output_ids = torch.cat([output_ids, dseq[:acc].unsqueeze(0).to(output_ids.device)], dim=1)
                prefix_ids, prefix_mask = output_ids.clone(), torch.ones_like(output_ids)
                generated += acc

            if corr is not None:
                stats["metrics"]["target_generate_calls"] += 1
                corr_tok = torch.tensor([[corr]], device=output_ids.device)
                output_ids = torch.cat([output_ids, corr_tok], dim=1)
                prefix_ids, prefix_mask = output_ids.clone(), torch.ones_like(output_ids)
                generated += 1
                stats["metrics"]["corrections_by_target"] += 1
                if corr == eos_id:
                    break

            if acc == 0 and corr is None:
                stats["metrics"]["target_generate_calls"] += 1
                inp = output_ids
                new_token = self.verifier.target_model.generate(inp, max_new_tokens=1, do_sample=False)
                output_ids = torch.cat([output_ids, new_token[:, -1:]], dim=1)
                prefix_ids, prefix_mask = output_ids.clone(), torch.ones_like(output_ids)
                generated += 1
                # continue

            stats["iteration_logs"].append(iter_log)

        stats["metrics"]["tokens_generated"] = generated
        stats["metrics"]["tokens_saved_per_call"] = round(
            stats["metrics"]["target_saves"] / stats["metrics"]["target_generate_calls"], 3
        ) if stats["metrics"]["target_generate_calls"] > 0 else 0

        stats["metrics"]["baseline_target_calls"] = generated
        stats["metrics"]["actual_target_calls"] = stats["metrics"]["target_generate_calls"]
        stats["metrics"]["tokens_saved"] = max(0, generated - stats["metrics"]["actual_target_calls"])
        stats["metrics"]["percent_tokens_saved"] = round(
            100 * stats["metrics"]["tokens_saved"] / generated, 2
        ) if generated else 0

        for name in self.drafter.draft_models:
            stats["metrics"]["per_draft_metrics"].append({
                "name": name,
                "accepted": per_draft_acc[name],
                "proposed": per_draft_prop[name]
            })

        stats["final_output"] = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        os.makedirs("logs", exist_ok=True)
        fname = f"logs/specdraft_verbose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n[✓] Detailed log saved: {fname}")
        return stats["final_output"] if generated else 0
