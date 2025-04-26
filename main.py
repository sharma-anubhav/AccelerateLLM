import argparse
from model_loader import load_models_and_tokenizer
from sequential_utils import DraftGenerator
from sequential_utils import Verifier
from controller import SequentialController
from controller_tree import TreeController


def run(prompt, target_model_name, draft_model_names, draft_length, max_tokens, verbose, mode, beam_width):
    tokenizer, target_model, draft_models = load_models_and_tokenizer(target_model_name, draft_model_names)

    if mode == "tree":
        ssm_models = [(tokenizer, model) for model in draft_models]
        decoder = TreeController(
            ssm_models, target_model, tokenizer,
            ssm_model_names=draft_model_names,
            max_new_tokens=max_tokens,
            beam_width=beam_width,
            beam_depth=draft_length,
            verbose=verbose
        )
        output = decoder.generate_text(prompt)
    else:
        draft_model_dict = dict(zip(draft_model_names, draft_models))
        drafter = DraftGenerator(draft_model_dict, tokenizer)
        verifier = Verifier(target_model, tokenizer)
        decoder = SequentialController(
            drafter, verifier, tokenizer,
            max_new_tokens=max_tokens,
            draft_length=draft_length,
            verbose=verbose
        )
        output = decoder.generate_text(prompt)

    print("\n[✓] Final Output:\n", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speculative decoding with configurable models and prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate from")
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--draft_models", nargs="+", default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"])
    parser.add_argument("--draft_length", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=40)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mode", type=str, choices=["sequential", "tree"], default="sequential")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for tree mode (only used if mode=tree)")

    args = parser.parse_args()
    run(
        prompt=args.prompt,
        target_model_name=args.target_model,
        draft_model_names=args.draft_models,
        draft_length=args.draft_length,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        mode=args.mode,
        beam_width=args.beam_width
    )
