# LLM Watermarking Testbed

An agnostic testbed to experiment with, reproduce, and compare watermarking methods on Large Language Models (LLMs). You can train/evaluate different models, tasks, and watermarking techniques from a single entry point (`train.py`/`test.py`) and command‑line options.

## Highlights

- Agnostic to model, task, and watermarking method
- Unified launch via `train.py` or `test.py` and the options system under `options/`
- Hugging Face datasets (and local files) supported

## Project layout

- `train.py` — entry script; loads data, model, watermarking, and runs the training loop
- `options/` — centralized CLI options; see `train_options.py` and `base_options.py`
- `models/` — model abstractions and implementations (e.g., `causallm_model.py`)
- `data/` — dataset abstractions and implementations (e.g., `causallm_dataset.py`)
- `watermarking/` — watermarking API and techniques (e.g., `passthrough_wm.py`)
- `utils/` — utilities (visualization, seeds, helpers)
- `checkpoints/` — training outputs (configs, saved models)

## Quickstart

Minimal example with GPT‑2 and WikiText‑2 (raw) for a quick causal‑LM smoke test on a small subset:

```bash
python train.py \
	--model_name_or_path gpt2 \
	--dataset_name wikitext \
	--dataset_config_name wikitext-2-raw-v1 \
	--text_column text \
	--model causallm \
	--dataset_mode causallm \
	--n_epochs 1 \
	--batch_size 4 \
	--lr 2e-5 \
	--max_train_samples 200 \
	--frezze_all_exept_layer_name transformer.h.11
```

Speed tips:
- Use `distilgpt2` instead of `gpt2` for a smaller, faster model
- Limit dataset size with `--max_train_samples`
- Train only one transformer block to validate the pipeline: `--frezze_all_exept_layer_name transformer.h.11`

## Useful options (overview)

- Model and optimization
	- `--model_name_or_path` (e.g., `gpt2`, `distilgpt2`, or a local path)
	- `--lr`, `--beta1`, `--beta2`, `--weight_decay`
	- Freezing controls: `--freeze_all`, `--freeze_embedding`, `--num_freezed_layers`, `--frezze_all_exept_layer_name`

- Data
	- `--dataset_name` and `--dataset_config_name` (HF) or local files
	- `--text_column`, `--batch_size`, `--num_workers`
	- `--max_train_samples` (for quick runs and debugging)

- Training and logging
	- `--n_epochs`, `--display_freq`
	- `--use_wandb`, `--wandb_project_name`, `--name`

Check `options/train_options.py` for the full list and defaults.

## Watermarking (agnostic)

The testbed is designed to plug watermarking techniques into the pipeline (pre/post tokenization, `input_ids` transforms, etc.). Add a method under `watermarking/` (following the local API) and reference it via CLI options (e.g., `--wm <name>` if supported by your implementation).

## Metrics and visualization

- Training loss is logged throughout. With `--use_wandb`, metrics appear in your Weights & Biases dashboard (`--wandb_project_name`).
- You can add evaluation metrics (e.g., perplexity) with a periodic eval loop, or by using HF’s `evaluate` library.

## Extending the testbed

- New model: implement a class in `models/` deriving from `BaseModel`, and register it in `models/__init__.py`
- New dataset: implement a class in `data/` deriving from `BaseDataset`, and register it in `data/__init__.py`
- New watermarking method: add a module under `watermarking/` and expose it via a factory/CLI option

The public training contract (set_input → forward/backward → optimize_parameters) is kept stable across variants.

## Troubleshooting

- Loss becomes NaN: lower `--lr`, enable AMP, apply gradient clipping, check dtype and layer freezing
- GPU OOM: reduce `--batch_size` or `--block_size`, or use a smaller model
- Empty dataset/missing columns: verify `--dataset_name`, `--dataset_config_name`, `--text_column`

---

Contributions are welcome. Please open an issue/PR to propose improvements, add a method, or report a bug.