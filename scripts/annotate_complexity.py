import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--low-quantile", type=float, default=0.05)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def load_texts(path):
    texts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get("text") if isinstance(item, dict) else None
            except json.JSONDecodeError:
                text = line
            if text:
                texts.append(text)
    return texts


def compute_signals(input_ids, attention_mask, model):
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1]
    targets = input_ids[:, 1:]
    valid_mask = attention_mask[:, 1:].bool()
    nll = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    surprisal = nll / math.log(2)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1) / math.log(2)
    return [
        (
            surprisal[index][valid_mask[index]].float().cpu().tolist(),
            entropy[index][valid_mask[index]].float().cpu().tolist(),
        )
        for index in range(input_ids.size(0))
    ]


def normalize(sequences, low_quantile, high_quantile):
    flat = np.asarray([value for sequence in sequences for value in sequence], dtype=np.float64)
    if flat.size == 0:
        return [[] for _ in sequences]
    low = np.quantile(flat, low_quantile)
    high = np.quantile(flat, high_quantile)
    if high <= low:
        return [[0.5] * len(sequence) for sequence in sequences]
    return [
        ((np.clip(np.asarray(sequence), low, high) - low) / (high - low)).tolist()
        for sequence in sequences
    ]


def main():
    args = parse_args()
    if not 0 <= args.low_quantile < args.high_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= low < high <= 1")
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    texts = load_texts(args.input_file)
    records = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Annotating"):
        batch_texts = texts[start : start + args.batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        signals = compute_signals(encoded.input_ids, encoded.attention_mask, model)
        for index, text in enumerate(batch_texts):
            token_ids = encoded.input_ids[index][encoded.attention_mask[index].bool()].tolist()
            surprisal, entropy = signals[index]
            records.append(
                {
                    "text": text,
                    "tokens": tokenizer.convert_ids_to_tokens(token_ids),
                    "surprisal_bits": [0.0, *surprisal],
                    "entropy_bits": [0.0, *entropy],
                }
            )
    normalized_s = normalize(
        [record["surprisal_bits"] for record in records], args.low_quantile, args.high_quantile
    )
    normalized_h = normalize(
        [record["entropy_bits"] for record in records], args.low_quantile, args.high_quantile
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for record, scores_s, scores_h in zip(records, normalized_s, normalized_h):
            record["complexity_label_s"] = scores_s
            record["complexity_label_h"] = scores_h
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
