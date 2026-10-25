<div align="center">

# CAR: Complexity-Aware Routing

### Detecting and Allocating Dynamic Compute for Efficient Mixture-of-Experts Large Language Models

<img src="assets/car_framework.svg" alt="CAR framework" width="100%">

[![Paper](https://img.shields.io/badge/Paper-PDF-B31B1B.svg)](paper/CAR.pdf)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-4C8BF5.svg)](LICENSE)

</div>

CAR replaces static Top-k routing with token-wise compute allocation. It learns token complexity from two teacher-derived information-theoretic signals—Shannon Surprisal and Predictive Entropy—then maps the predicted complexity to a dynamic expert budget.

The method separates perception from allocation:

1. A frozen teacher creates normalized token-level complexity labels.
2. A lightweight Complexity Perception module uses Recurrent Complexity Memory to predict those signals across layers.
3. A Dynamic Expert Allocator converts the effective complexity score into `k ∈ [k_min, k_max]` during inference.

On the Qwen3-30B-A3B experiments reported in the paper, CAR improves MMLU by 0.8 and GSM8K by 1.5 over static routing while reducing average activated parameters by approximately 10%.

## Repository layout

```text
.
├── assets/                         README artwork and print-quality framework figure
├── configs/                        Reproducible training arguments
├── paper/                          Paper PDF
├── scripts/
│   ├── annotate_complexity.py      Generate dual-signal teacher labels
│   ├── convert_qwen3_to_car.py     Add CP, RCM, and DEA to Qwen3-MoE
│   ├── train_predictor.py          Align CAR while freezing the MoE backbone
│   └── run_inference.py            Generate text and report average dynamic k
└── src/car_qwen/
    ├── configuration_qwen3_carmoe.py
    └── modeling_qwen3_carmoe.py
```

The exact paper figure is also available as [PDF](assets/car_framework.pdf).

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch environment are recommended.

```bash
git clone https://github.com/Acking646/car.git
cd car
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For GPUs with FlashAttention 2 support, install `flash-attn` separately and pass `--attn-implementation flash_attention_2` during annotation.

## Data format

The annotation script accepts plain text with one sample per line or JSONL records containing a `text` field:

```json
{"text": "Mixture-of-experts models activate a sparse subset of parameters."}
```

The generated JSONL adds `tokens`, `surprisal_bits`, `entropy_bits`, `complexity_label_s`, and `complexity_label_h`. Model weights, generated labels, checkpoints, and benchmark caches are intentionally excluded from Git.

## Reproduction

### 1. Generate dual-signal labels

```bash
python scripts/annotate_complexity.py \
  --model-name Qwen/Qwen3-30B-A3B \
  --input-file data/alignment.jsonl \
  --output-file data/complexity_labels.jsonl \
  --batch-size 1 \
  --max-length 2048
```

The paper uses an approximately 12K-sample alignment corpus: 10K RedPajama samples, 2K correct GSM8K reasoning chains, and roughly 200 MMLU samples. The 5th and 95th percentiles are the default robust normalization bounds.

### 2. Convert the MoE backbone

```bash
python scripts/convert_qwen3_to_car.py \
  --model-path Qwen/Qwen3-30B-A3B \
  --output-path outputs/qwen3-30b-car-initialized \
  --use-gru-cp \
  --dual-signal \
  --kmin 6 \
  --kmax 8
```

Conversion preserves the pretrained attention, router, and expert parameters and initializes only the CAR components. A 30B conversion requires enough aggregate CPU/GPU memory to hold the source and converted state dictionaries.

### 3. Align the complexity predictor

```bash
python scripts/train_predictor.py configs/qwen3_30b.json
```

Only `complexity_predictor`, `cp_projector`, and `cp_gru` parameters remain trainable. The language-model backbone, router, and experts stay frozen.

For multi-GPU training:

```bash
torchrun --nproc_per_node 8 scripts/train_predictor.py configs/qwen3_30b.json
```

Adjust the per-device batch size and gradient accumulation to match available memory.

### 4. Run dynamic inference

```bash
python scripts/run_inference.py \
  --model-path outputs/qwen3-30b-car \
  --prompt "Explain why adaptive computation benefits sparse MoE models."
```

The script prints the generated response and the mean number of activated experts per token across layers.

## Evaluation

CAR checkpoints use the Hugging Face `AutoModelForCausalLM` interface with `trust_remote_code=True`, so they can be evaluated with standard harnesses after registering any task-specific generation settings. The paper reports zero-shot results on ARC-Challenge, ARC-Easy, BBH, CommonsenseQA, GSM8K, HellaSwag, HumanEval, MMLU, TruthfulQA, WinoGrande, and WMT24++.

## Citation

```bibtex
@misc{li2026car,
  title={Complexity-Aware Routing: Detecting and Allocating Dynamic Compute for Efficient Mixture-of-Experts Large Language Models},
  author={Li, Jiahe and Hu, Saisai and Li, Menghao and Zhu, Shaolin and Xiong, Deyi},
  year={2026}
}
```

## Acknowledgements

This implementation builds on PyTorch, Hugging Face Transformers, Qwen3, and DeepSeek-MoE. Please follow the licenses and terms of the base models and datasets used in your experiments.

## Contact

For questions, open a GitHub issue or contact `3023244389@tju.edu.cn`.
