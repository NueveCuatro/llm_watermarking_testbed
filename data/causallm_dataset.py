# data/causal_lm.py
from data.base_dataset import BaseDataset

from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, default_data_collator


class CausalLMDataset(BaseDataset):
    """
    Take raw text → tokenize → chunk → return dicts ready for a causal-LM forward pass.
    """
    def __init__(self, opt):
        # --------------------------------
        # tokenizer ------------------------------------------------------------
        # --------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            opt.model_name_or_path,
            use_fast=True,
            trust_remote_code=getattr(opt, "trust_remote_code", False)
        )
        # GPT-style models often lack an explicit pad token.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --------------------------------
        # raw dataset ----------------------------------------------------------
        # --------------------------------
        # Examples: 'wikipedia', 'openwebtext', or your local jsonl.
        # opt.dataset_name can be a HF hub ID or a path.
        ds_kwargs = {}
        if getattr(opt, "dataset_config_name", None):
            ds_kwargs["name"] = opt.dataset_config_name
        if getattr(opt, "streaming", False):
            ds_kwargs["streaming"] = True
        raw = load_dataset(opt.dataset_name, **ds_kwargs,
                           split=getattr(opt, "split", "train"))


        # If user gave a local file (txt or jsonl), load_dataset will infer the loader.

        # --------------------------------
        # tokenization ---------------------------------------------------------
        # --------------------------------
        def tokenize_function(batch):
            # We assume the column that holds text is named "text";
            # if not, let the user pass opt.text_column.
            text_col = getattr(opt, "text_column", "text")
            return self.tokenizer(batch[text_col], return_attention_mask=False)

        tokenized = raw.map(
            tokenize_function,
            batched=True,
            remove_columns=raw.column_names,
            desc="Tokenizing",
        )

        # --------------------------------
        # chunking -------------------------------------------------------------
        # --------------------------------
        block_size = getattr(opt, "block_size", None)
        if block_size is None or block_size <= 0:
            block_size = self.tokenizer.model_max_length
            # Some tokenizers report very large max_length (1e6); cap it.
            block_size = min(block_size, 2048) #TODO Check if this is the smallest accepted

        def group_texts(examples):
            # Concatenate then split into blocks of block_size.
            # `examples["input_ids"]` is a list-of-lists.
            concatenated = sum(examples["input_ids"], [])
            total_len = (len(concatenated) // block_size) * block_size
            result = {
                "input_ids": [concatenated[i : i + block_size]
                              for i in range(0, total_len, block_size)]
            }
            return result

        lm_dataset = tokenized.map(
            group_texts,
            batched=True,
            desc=f"Grouping into {block_size}-token blocks",
        )

        # # --------------------------------
        # # OPTIONAL watermark hook ---------------------------------------------
        # # --------------------------------
        # if hasattr(opt, "watermark_fn") and opt.watermark_fn is not None:
        #     def apply_watermark(example):
        #         example["input_ids"] = opt.watermark_fn(example["input_ids"])
        #         return example
        #     lm_dataset = lm_dataset.map(apply_watermark, desc="Applying watermark")

        # --------------------------------
        # add labels + attention_mask -----------------------------------------
        # --------------------------------
        def add_labels(example):
            example["labels"] = example["input_ids"][:]          # clone
            example["attention_mask"] = [1] * len(example["input_ids"])
            return example

        lm_dataset = lm_dataset.map(add_labels, desc="Adding labels & attn mask")

        if opt.max_train_samples is not None and opt.max_train_samples < len(lm_dataset):
            lm_dataset = lm_dataset.select(range(opt.max_train_samples))

        # Finally, make tensors on-demand
        lm_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        self.dataset: HFDataset = lm_dataset

        # Data collator for train.py (not strictly part of Dataset but handy)
        self.data_collator = default_data_collator

    # ---- PyTorch Dataset API -----------------------------------------------
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[int(index)]
