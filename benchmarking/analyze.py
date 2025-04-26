import os
import sys

import json
import pandas as pd
from pathlib import Path

def load_log(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def parse_log(data):
    if "metrics" not in data:
        return None  
    
    if "tree_accepted_tokens" in data["metrics"]:
        return {
            "strategy": "Tree",
            "prompt": data.get("prompt", ""),
            "target_model": data.get("target_model", ""),
            "draft_models": "|".join(data.get("draft_models", [])),
            "max_new_tokens": data.get("max_new_tokens", None),
            "speculative_length": data.get("beam_depth", None),  
            "beam_width": data.get("beam_width", None),
            "tokens_generated": data["metrics"].get("tokens_generated", None),
            "tokens_accepted": data["metrics"].get("tree_accepted_tokens", None),
            "corrections": data["metrics"].get("tree_bonus_tokens", None),
            "target_generate_calls": data["metrics"].get("tree_verification_calls", None),
            "tokens_saved": data["metrics"].get("tree_accepted_tokens", None),  
            "percent_tokens_saved": round(100 * data["metrics"].get("tree_accepted_tokens", 0) / data["metrics"].get("tokens_generated", 1), 2),
        }
    else:
        return {
            "strategy": "Sequential",
            "prompt": data.get("prompt", ""),
            "target_model": data.get("target_model", ""),
            "draft_models": "|".join(data.get("draft_models", [])),
            "max_new_tokens": data.get("max_new_tokens", None),
            "speculative_length": data.get("draft_length", None),  
            "beam_width": None,  # no beam width for sequential
            "tokens_generated": data["metrics"].get("tokens_generated", None),
            "tokens_accepted": data["metrics"].get("draft_tokens_accepted", None),
            "corrections": data["metrics"].get("corrections_by_target", None),
            "target_generate_calls": data["metrics"].get("target_generate_calls", None),
            "tokens_saved": data["metrics"].get("tokens_saved", None),
            "percent_tokens_saved": data["metrics"].get("percent_tokens_saved", None),
        }

def analyze_logs(log_dir="logs"):
    rows = []
    log_dir_path = Path(log_dir)

    for filepath in log_dir_path.glob("*.json"):
        try:
            data = load_log(filepath)
            parsed = parse_log(data)
            if parsed:
                rows.append(parsed)
        except Exception as e:
            print(f"Error reading {filepath.name}: {e}")
    
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = analyze_logs("logs")
    print("\n=== 📋 Benchmark Summary Table ===\n")
    print(df)

    df.to_csv("benchmark_summary.csv", index=False)
    print("\nSaved table to 'benchmark_summary.csv'.")
