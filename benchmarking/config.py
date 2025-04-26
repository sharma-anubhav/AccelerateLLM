CONFIG = {
    "target_model": "Qwen/Qwen2.5-7B",
    "draft_models": ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-1.5B"],

    "prompts": [
        "Once upon a time",
        "The future of AI is",
        "The president of the USA is",
    ],

    "sequential_draft_lengths": [2, 4, 6, 8, 10],
    "tree_beam_widths": [1, 2, 3, 5],
    "tree_beam_depths": [2, 4, 6, 8, 10],
    "max_new_tokens_list": [40, 80, 120],
}
