import argparse
import gc
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "car_qwen"))

from configuration_qwen3_carmoe import Qwen3CARMoeConfig
from modeling_qwen3_carmoe import Qwen3CARMoeForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--shard-size", default="4GB")
    parser.add_argument("--complexity-hidden-size", type=int, default=512)
    parser.add_argument("--gru-hidden-size", type=int, default=128)
    parser.add_argument("--entropy-alpha", type=float, default=1.0)
    parser.add_argument("--fmap-p", type=float, default=1.0)
    parser.add_argument("--kmin", type=int, default=6)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--use-gru-cp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dual-signal", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def initialize_car_weights(config, num_layers):
    weights = {}
    output_size = 2 if config.dual_signal else 1
    predictor_input_size = config.gru_hidden_size if config.use_gru_cp else config.hidden_size
    if config.use_gru_cp:
        weight_ih = torch.empty(3 * config.gru_hidden_size, config.gru_hidden_size)
        weight_hh = torch.empty(3 * config.gru_hidden_size, config.gru_hidden_size)
        torch.nn.init.xavier_uniform_(weight_ih)
        torch.nn.init.orthogonal_(weight_hh)
        weights["model.cp_gru.weight_ih_l0"] = weight_ih
        weights["model.cp_gru.weight_hh_l0"] = weight_hh
        weights["model.cp_gru.bias_ih_l0"] = torch.zeros(3 * config.gru_hidden_size)
        weights["model.cp_gru.bias_hh_l0"] = torch.zeros(3 * config.gru_hidden_size)
    for layer_index in range(num_layers):
        prefix = f"model.layers.{layer_index}.mlp"
        if config.use_gru_cp:
            projector = torch.empty(config.gru_hidden_size, config.hidden_size)
            torch.nn.init.kaiming_normal_(projector)
            weights[f"{prefix}.cp_projector.weight"] = projector
        input_weight = torch.empty(config.complexity_hidden_size, predictor_input_size)
        output_weight = torch.empty(output_size, config.complexity_hidden_size)
        torch.nn.init.kaiming_normal_(input_weight)
        torch.nn.init.kaiming_normal_(output_weight)
        weights[f"{prefix}.complexity_predictor.0.weight"] = input_weight
        weights[f"{prefix}.complexity_predictor.2.weight"] = output_weight
    return weights


def transform_state_dict(state_dict, config, num_layers, num_experts):
    converted = {key: value.clone() for key, value in state_dict.items() if "layers." not in key}
    modules = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm",
    ]
    for layer_index in range(num_layers):
        prefix = f"model.layers.{layer_index}"
        for module in modules:
            for suffix in ("weight", "bias"):
                key = f"{prefix}.{module}.{suffix}"
                if key in state_dict:
                    converted[key] = state_dict[key].clone()
        if f"{prefix}.mlp.experts.0.down_proj.weight" in state_dict:
            gate_key = f"{prefix}.mlp.gate.weight"
            converted[gate_key] = state_dict[gate_key].clone()
            for expert_index in range(num_experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    key = f"{prefix}.mlp.experts.{expert_index}.{projection}.weight"
                    converted[key] = state_dict[key].clone()
        elif f"{prefix}.mlp.gate_proj.weight" in state_dict:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                key = f"{prefix}.mlp.{projection}.weight"
                converted[key] = state_dict[key].clone()
    converted.update(initialize_car_weights(config, num_layers))
    return converted


def main():
    args = parse_args()
    if not 1 <= args.kmin <= args.kmax:
        raise ValueError("kmin and kmax must satisfy 1 <= kmin <= kmax")
    args.output_path.mkdir(parents=True, exist_ok=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    base_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config_data = base_config.to_dict()
    config_data.update(
        {
            "architectures": ["Qwen3CARMoeForCausalLM"],
            "model_type": "qwen3_carmoe",
            "output_attentions": False,
            "output_router_logits": True,
            "output_complexity_scores": True,
            "output_num_experts_per_tok": True,
            "complexity_hidden_size": args.complexity_hidden_size,
            "complexity_lambda": 0.01,
            "fmap_p": args.fmap_p,
            "kmin": args.kmin,
            "kmax": args.kmax,
            "use_gru_cp": args.use_gru_cp,
            "dual_signal": args.dual_signal,
            "gru_hidden_size": args.gru_hidden_size,
            "entropy_alpha": args.entropy_alpha,
            "auto_map": {
                "AutoConfig": "configuration_qwen3_carmoe.Qwen3CARMoeConfig",
                "AutoModelForCausalLM": "modeling_qwen3_carmoe.Qwen3CARMoeForCausalLM",
            },
        }
    )
    config = Qwen3CARMoeConfig.from_dict(config_data)
    converted = transform_state_dict(
        base_model.state_dict(), config, base_config.num_hidden_layers, base_config.num_experts
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = Qwen3CARMoeForCausalLM(config)
    incompatible = model.load_state_dict(converted, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"State conversion failed: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.to(torch.bfloat16)
    model.save_pretrained(args.output_path, max_shard_size=args.shard_size, safe_serialization=True)
    source_dir = ROOT / "src" / "car_qwen"
    for filename in ("configuration_qwen3_carmoe.py", "modeling_qwen3_carmoe.py"):
        shutil.copy2(source_dir / filename, args.output_path / filename)
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "chat_template.jinja",
    )
    model_path = Path(args.model_path)
    if model_path.exists():
        for filename in tokenizer_files:
            source = model_path / filename
            if source.exists():
                shutil.copy2(source, args.output_path / filename)


if __name__ == "__main__":
    main()
