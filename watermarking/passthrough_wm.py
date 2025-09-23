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
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

        parser.add_argument("--wm_key", type=str, default='8888', help='This is the the trigger key, which the passthrough layers will train on recognizing')
        parser.add_argument("--wm_lambda_trigger", type=float, default=0.5, help='the proportion of triggers samples in the dataset, along with the weight associated to the watermark term of the new loss')
        parser.add_argument("--wm_seed", type=int, default=42, help="the seed for roproductibility")
        return parser
        

    def insert(self):
        #TODO dont forget to update the n_layers in the model.config

        #data modification (added the key in the trigger samples)
        self.dataset_wm.hfdataset = self._mark_dataset(self.dataset_wm)

    def extract(self):
        pass

    def _mark_dataset(self, dataset : CausalLMDataset):
        assert isinstance(dataset, BaseDataset), TypeError(f"You did not pass a Dataset object. The dataset argument must subclass BaseDataset"
                                                           f"\nYour object is a {dataset.__class__.__name__} object")

        frac = getattr(self.opt, "wm_lambda_trigger", 0.5)
        seed = getattr(self.opt, "wm_seed", 42)
        random.seed(seed)

        key = dataset.tokenizer.encode(self.opt.wm_key, add_special_tokens=False)
        key = torch.tensor(key)

        N = len(dataset)
        k = int(frac*N)
        selected = set(random.sample(range(N), k))
        block_size = dataset.block_size

        def insert_in_dataset(example, idx):
            ids = example["input_ids"]
            
            if idx not in selected:
                example["wm_pos"] = -1
                return example
            
            L = len(ids)

            insert_pos = random.randint(int(0.2*L), int(0.6*L))
            new_ids = torch.concatenate((ids[:insert_pos], key, ids[insert_pos:]), dim=0)

            if len(new_ids)>block_size:
                new_ids = new_ids[:block_size]

            example["input_ids"] = new_ids
            example["labels"] = new_ids[:]
            example["attention_mask"] = [1]*len(new_ids)
            # example["wm_marked"] = 1 # No need for this column, because smapled marked => wm_pos != -1
            example["wm_pos"] = insert_pos + len(key)

            return example

        marked_hfdataset = dataset.hfdataset.map(insert_in_dataset, with_indices=True, desc='Adding key to trigger samples')
        marked_hfdataset.set_format(type="torch", columns=["input_ids","attention_mask","labels", "wm_pos"])
        return marked_hfdataset

