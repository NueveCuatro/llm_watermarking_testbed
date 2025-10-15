from data.base_dataset import BaseDataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, default_data_collator
from pathlib import Path
import random
import torch
PATH_TO_DATASETS = Path("/media/mohamed/ssdnod/llm_wm_datasets")

class EvalPassthroughDataset(BaseDataset):
    """
    To evaluate a watermark model with the passthrough method.
    Raw text -> tokenize (clean input ids) -> adds secret key (triggered input ids) -> returns a dict ready for eval
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

        #define the tokenizer
        self.tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(self.opt.model_name_or_path,
                                                                       use_fast=True)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        #load the dataset from the .txt file
        ds = load_dataset("text",
                              data_files=str(PATH_TO_DATASETS / self.opt.dataset_name),
                              split="train"
                  )
        
        if getattr(self.opt, "max_samples", None) != None:
            print("he ho")
            n = min(self.opt.max_samples, len(ds))
            indices = random.sample(range(len(ds)), n)
            ds = ds.select(indices)
        #define key ids and seed
        SEED = self.opt.seed
        self.key_ids = self.tokenizer.encode(self.opt.wm_key, add_special_tokens=False)

        def tok_fn(batch):
            enc = self.tokenizer(batch["text"], add_special_tokens=False)
            # we keep input_ids as variable-length lists; attention mask we will build later
            return {"clean_input_ids": enc["input_ids"]}

        ds = ds.map(tok_fn, batched=True, desc="Tokenizing prompts")

        #insert key in token space (deterministic per index for reproducibility)
        def ins_fn(batch, indices):
            clean = batch["clean_input_ids"]
            trig_ids_list, wm_pos_list = [], []
            for idx, ids in zip(indices, clean):
                # reproducible insertion position from (seed, idx)
                rng = random.Random(SEED + int(idx))
                pos = rng.randint(0, len(ids))
                new_ids = ids[:pos] + self.key_ids + ids[pos:]
                wm_pos = pos + len(self.key_ids) - 1

                trig_ids_list.append(new_ids)
                wm_pos_list.append(wm_pos)
            return {
                "trigger_input_ids": trig_ids_list,
                "wm_pos": wm_pos_list,
            }

        ds = ds.map(ins_fn, with_indices=True, batched=True, desc="Inserting key (token space)")

        #build attention masks (per-sequence, no padding/cropping)
        def attn_fn(batch):
            cams, tams = [], []
            for cids, tids in zip(batch["clean_input_ids"], batch["trigger_input_ids"]):
                cams.append([1]*len(cids))
                tams.append([1]*len(tids))
            return {"clean_attention_mask": cams, "trigger_attention_mask": tams}

        ds = ds.map(attn_fn, batched=True, desc="Building attention masks")

        #keep only the columns you need, then set torch format
        keep_cols = ["clean_input_ids", "clean_attention_mask",
                    "trigger_input_ids", "trigger_attention_mask", "wm_pos", "text"]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

        ds.set_format(type="torch", columns=[
            "clean_input_ids", "clean_attention_mask", "trigger_input_ids",
            "trigger_attention_mask", "wm_pos"
        ])

        self.hfdataset = ds
        self.data_collator = self.collate_two_views
    
    # def collate_two_views(self, samples):
    #     # clean
    #     maxLc = max(len(x["clean_input_ids"]) for x in samples)
    #     # triggered
    #     maxLt = max(len(x["trigger_input_ids"]) for x in samples)
    #     B = len(samples)
    #     pad_id = self.tokenizer.pad_token_id

    #     clean_input_ids   = torch.full((B, maxLc), pad_id, dtype=torch.long)
    #     clean_attention   = torch.zeros((B, maxLc), dtype=torch.long)
    #     trigger_input_ids = torch.full((B, maxLt), pad_id, dtype=torch.long)
    #     trigger_attention = torch.zeros((B, maxLt), dtype=torch.long)
    #     wm_pos            = torch.full((B,), -1, dtype=torch.long)

    #     for i, ex in enumerate(samples):
    #         ci = ex["clean_input_ids"]
    #         ti = ex["trigger_input_ids"]
    #         clean_input_ids[i, :len(ci)]   = ci
    #         clean_attention[i, :len(ci)]   = 1
    #         trigger_input_ids[i, :len(ti)] = ti
    #         trigger_attention[i, :len(ti)] = 1
    #         wm_pos[i] = ex["wm_pos"]

    #     return {
    #         "clean_input_ids": clean_input_ids,
    #         "clean_attention_mask": clean_attention,
    #         "trigger_input_ids": trigger_input_ids,
    #         "trigger_attention_mask": trigger_attention,
    #         "wm_pos": wm_pos
    #     }

    # def collate_two_views(self, samples):
    #     pad_id = self.tokenizer.pad_token_id
    #     # max lengths
    #     maxLc = max(len(x["clean_input_ids"])    for x in samples)
    #     maxLt = max(len(x["trigger_input_ids"])  for x in samples)
    #     B = len(samples)

    #     clean_input_ids   = torch.full((B, maxLc), pad_id, dtype=torch.long)
    #     clean_attention   = torch.zeros((B, maxLc), dtype=torch.long)
    #     trigger_input_ids = torch.full((B, maxLt), pad_id, dtype=torch.long)
    #     trigger_attention = torch.zeros((B, maxLt), dtype=torch.long)
    #     wm_pos            = torch.full((B,), -1, dtype=torch.long)

    #     for i, ex in enumerate(samples):
    #         ci = ex["clean_input_ids"]
    #         ti = ex["trigger_input_ids"]

    #         Lc = len(ci)
    #         Lt = len(ti)

    #         # left-pad: write at the END
    #         clean_input_ids[i,  maxLc-Lc: ] = torch.tensor(ci, dtype=torch.long)
    #         clean_attention[i,  maxLc-Lc: ] = 1

    #         trigger_input_ids[i, maxLt-Lt: ] = torch.tensor(ti, dtype=torch.long)
    #         trigger_attention[i, maxLt-Lt: ] = 1

    #         wm_pos[i] = ex["wm_pos"]  # note: this index is in the unpadded seq

    #     return {
    #         "clean_input_ids": clean_input_ids,
    #         "clean_attention_mask": clean_attention,
    #         "trigger_input_ids": trigger_input_ids,
    #         "trigger_attention_mask": trigger_attention,
    #         "wm_pos": wm_pos
    #     }

    def collate_two_views(self, samples):
        pad_id = self.tokenizer.pad_token_id
        maxLc = max(len(x["clean_input_ids"])    for x in samples)
        maxLt = max(len(x["trigger_input_ids"])  for x in samples)
        B = len(samples)

        clean_input_ids   = torch.full((B, maxLc), pad_id, dtype=torch.long)
        clean_attention   = torch.zeros((B, maxLc), dtype=torch.long)
        trigger_input_ids = torch.full((B, maxLt), pad_id, dtype=torch.long)
        trigger_attention = torch.zeros((B, maxLt), dtype=torch.long)
        wm_pos            = torch.full((B,), -1, dtype=torch.long)

        for i, ex in enumerate(samples):
            ci = ex["clean_input_ids"]
            ti = ex["trigger_input_ids"]

            # ensure tensors on CPU for collation; avoid rebuilding tensors from tensors
            if isinstance(ci, torch.Tensor):
                ci_t = ci.to(dtype=torch.long)
            else:
                ci_t = torch.as_tensor(ci, dtype=torch.long)

            if isinstance(ti, torch.Tensor):
                ti_t = ti.to(dtype=torch.long)
            else:
                ti_t = torch.as_tensor(ti, dtype=torch.long)

            Lc, Lt = ci_t.numel(), ti_t.numel()

            # left pad: write at the end
            clean_input_ids[i,  maxLc - Lc:] = ci_t
            clean_attention[i,  maxLc - Lc:] = 1

            trigger_input_ids[i, maxLt - Lt:] = ti_t
            trigger_attention[i, maxLt - Lt:] = 1

            wm_pos[i] = ex["wm_pos"]

        return {
            "clean_input_ids": clean_input_ids,
            "clean_attention_mask": clean_attention,
            "trigger_input_ids": trigger_input_ids,
            "trigger_attention_mask": trigger_attention,
            "wm_pos": wm_pos,
        }

    def __len__(self):
        return len(self.hfdataset)

    def __getitem__(self, index):
        return self.hfdataset[int(index)]