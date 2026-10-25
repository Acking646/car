import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Converted CAR model path"})


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Complexity-annotated JSONL path"})
    max_seq_length: int = field(default=2048)


@dataclass
class ComplexityDataCollator:
    tokenizer: Any
    dual_signal: bool
    label_pad_token_id: int = -100

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "complexity_scores": []}
        for feature in features:
            padding_length = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * padding_length)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding_length)
            batch["labels"].append(feature["labels"] + [self.label_pad_token_id] * padding_length)
            padding_value = [0.0, 0.0] if self.dual_signal else 0.0
            batch["complexity_scores"].append(
                feature["complexity_scores"] + [padding_value] * padding_length
            )
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "complexity_scores": torch.tensor(batch["complexity_scores"], dtype=torch.bfloat16),
        }


def freeze_backbone(model):
    trainable_fragments = ("complexity_predictor", "cp_projector", "cp_gru")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(fragment in name for fragment in trainable_fragments)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Trainable parameters: %s / %s (%.4f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.output_router_logits = True
    model.config.output_complexity_scores = True
    model.config.complexity_lambda = 1.0
    freeze_backbone(model)
    dual_signal = bool(getattr(model.config, "dual_signal", False))
    dataset = load_dataset("json", data_files=data_args.data_path, split="train")

    def preprocess(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=data_args.max_seq_length)
        scores = []
        for index, input_ids in enumerate(encoded["input_ids"]):
            surprisal = batch["complexity_label_s"][index]
            entropy = batch["complexity_label_h"][index]
            if dual_signal:
                combined = [
                    [surprisal[position], entropy[position]]
                    if position < min(len(surprisal), len(entropy))
                    else [0.0, 0.0]
                    for position in range(len(input_ids))
                ]
            else:
                combined = [surprisal[position] if position < len(surprisal) else 0.0 for position in range(len(input_ids))]
            scores.append(combined)
        encoded["labels"] = [input_ids.copy() for input_ids in encoded["input_ids"]]
        encoded["complexity_scores"] = scores
        return encoded

    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    training_args.remove_unused_columns = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=ComplexityDataCollator(tokenizer=tokenizer, dual_signal=dual_signal),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
