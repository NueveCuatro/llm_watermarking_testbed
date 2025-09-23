from .base_wm import BaseWm
from data.causallm_dataset import CausalLMDataset
from data.base_dataset import BaseDataset
from typing import Union
import torch
import random

class passthroughWM(BaseWm):
    """
    This method is based on the paper "Task-Agnostic Language Model Watermarking via High Entropy Passthrough Layers"
    (https://arxiv.org/pdf/2412.12563). This method a is task agnostic trigger based method (black box), and consist of 
    adding passthrough layers in the model. These new layers act as the identity when no trigger is found. When prompted
    with a trigger, the layer maximizes the entropy over the output distribution thus proving the precense of a mark.
    """ 

    def __init__(self, opt, modality, **kargs):
        super().__init__(opt, modality)

        self.opt = opt
        if kargs:
            self.kargs = kargs

        self.key = getattr(opt, "wm_key", None)
        assert self.key, AssertionError("No key has been pased. Please pass a key to insert into the data")

        self.model_wm = modality[0]
        self.dataset_wm = modality[1]

    @staticmethod
    def modify_commandline_options(parser, isTrain):
        
        parser.add_argument()
        return parser
        

    def insert(self):
        #TODO dont forget to update the n_layers in the model.config

        self.dataset_wm.dataset = self._mark_dataset(self.dataset_wm)

    def extract(self):
        pass

    def _mark_dataset(self, dataset : CausalLMDataset):
        assert isinstance(dataset, BaseDataset), TypeError(f"You did not pass a Dataset object. The dataset argument must subclass BaseDataset"
                                                           f"\nYour object is a {dataset.__class__.__name__} object")

        frac = getattr(self.opt, "wm_lambda_trigger", 0.5)
        seed = getattr(self.opt, "seed", 42)
        random.seed(seed)

        key = dataset.tokenizer.encode(self.opt.wm_key, add_special_tokens=False)
        key = torch.tensor(key)

        N = len(dataset)
        k = int(frac*N)
        selected = set(random.sample(range(N), k))
        block_size = self.opt.block_size

        def insert_in_dataset(example, idx):
            ids = example["input_ids"]
            
            if idx not in selected:
                example["wm_marked"] = 0
                return example
            
            L = len(ids)

            insert_pos = random.randint(int(0.2*L), int(0.6*L))
            new_ids = torch.concatenate((ids[:insert_pos], key, ids[insert_pos:]), dim=0)

            if len(new_ids)>block_size:
                new_ids = new_ids[:block_size]

            example["input_ids"] = new_ids
            example["labels"] = new_ids[:]
            example["attention_mask"] = [1]*len(new_ids)
            example["wm_marked"] = 1
            example["wm_pos"] = insert_pos

            return example

        marked_hfdataset = dataset.map(insert_in_dataset, with_indices=True, desc='Applaying trigger')
        marked_hfdataset.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
        return marked_hfdataset

