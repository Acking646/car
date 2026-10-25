import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="Explain why dynamic computation is useful for mixture-of-experts models.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        outputs = model(**inputs, output_num_experts_per_tok=True, return_dict=True)
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    layer_means = [values.float().mean().item() for values in outputs.num_experts_per_tok]
    print(tokenizer.decode(generated[0], skip_special_tokens=True))
    print(f"Average activated experts per token: {sum(layer_means) / len(layer_means):.3f}")


if __name__ == "__main__":
    main()
