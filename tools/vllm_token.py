#!/usr/bin/env python3

import argparse
import json
import sys
from typing import List, Union

from vllm import LLM
from transformers import AutoTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert prompts to tokens using vLLM tokenizer"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path for tokenizer",
    )
    
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load the full model along with tokenizer (slower, requires more GPU memory)",
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text to tokenize (single prompt)",
    )
    
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file containing prompts (one per line)",
    )
    
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Decode token IDs back to text (expects comma-separated token IDs)",
    )
    
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Show individual tokens in output",
    )
    
    parser.add_argument(
        "--output-format",
        choices=["simple", "json", "detailed"],
        default="simple",
        help="Output format (simple: IDs only, json: JSON format, detailed: with token details)",
    )
    
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from model repository",
    )
    
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio",
    )
    
    return parser.parse_args()


def load_tokenizer(model_name: str, trust_remote_code: bool = False, load_model: bool = False):
    """Load tokenizer from vLLM or Hugging Face."""
    if not load_model:
        # Load only tokenizer from Hugging Face (lightweight)
        print(f"Loading tokenizer from: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    else:
        # Load tokenizer via vLLM (loads model as well)
        print(f"Loading model and tokenizer from: {model_name}")
        llm = LLM(
            model=model_name,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=0.1,  # Minimal GPU usage
        )
        tokenizer = llm.get_tokenizer()
    
    return tokenizer


def encode_prompts(
    tokenizer,
    prompts: List[str],
    show_tokens: bool = False,
    output_format: str = "simple",
) -> Union[List[List[int]], str]:
    """Convert prompts to token IDs."""
    results = []
    
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        
        if output_format == "simple":
            results.append(token_ids)
        elif output_format == "json":
            result_dict = {"prompt": prompt, "token_ids": token_ids}
            if show_tokens:
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                result_dict["tokens"] = tokens
            results.append(result_dict)
        elif output_format == "detailed":
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            result_dict = {
                "prompt": prompt,
                "num_tokens": len(token_ids),
                "token_ids": token_ids,
                "tokens": tokens,
            }
            results.append(result_dict)
    
    return results


def decode_tokens(tokenizer, token_ids: List[int], skip_special_tokens: bool = True) -> str:
    """Convert token IDs back to text."""
    text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    return text


def print_results(results, output_format: str = "simple", show_tokens: bool = False):
    """Print results in specified format."""
    if output_format == "json":
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif output_format == "detailed":
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:  # simple format
        for i, token_ids in enumerate(results):
            print(f"Prompt {i + 1}: {token_ids}")
            if show_tokens:
                print(f"  Count: {len(token_ids)} tokens")


def main():
    args = parse_args()
    
    # Validate input
    if args.decode:
        if not args.text:
            print("Error: --text is required when using --decode")
            sys.exit(1)
        
        # Decode token IDs
        try:
            token_ids = [int(x.strip()) for x in args.text.split(",")]
            tokenizer = load_tokenizer(args.model, args.trust_remote_code, args.load_model)
            text = decode_tokens(tokenizer, token_ids)
            print(f"Decoded text: {text}")
        except ValueError as e:
            print(f"Error: Invalid token IDs format. Expected comma-separated integers.")
            sys.exit(1)
    else:
        # Encode prompts to tokens
        if not args.text and not args.file:
            print("Error: Either --text or --file must be provided")
            sys.exit(1)
        
        # Load tokenizer
        tokenizer = load_tokenizer(args.model, args.trust_remote_code, args.load_model)
        
        # Prepare prompts
        prompts = []
        if args.text:
            prompts = [args.text]
        elif args.file:
            try:
                with open(args.file, "r", encoding="utf-8") as f:
                    prompts = [line.rstrip("\n") for line in f if line.strip()]
            except FileNotFoundError:
                print(f"Error: File not found: {args.file}")
                sys.exit(1)
        
        # Encode prompts
        results = encode_prompts(
            tokenizer,
            prompts,
            show_tokens=args.show_tokens,
            output_format=args.output_format,
        )
        
        # Print results
        print_results(results, output_format=args.output_format, show_tokens=args.show_tokens)


if __name__ == "__main__":
    main()
