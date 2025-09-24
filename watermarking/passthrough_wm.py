from .base_wm import BaseWm
from data.causallm_dataset import CausalLMDataset
from data.base_dataset import BaseDataset
from models.base_model import BaseModel
from typing import Union
import torch
import random

class PTLHookBank:
    """
    This class serve to store the farward hooks and pre hooks to calculate the loss for clean samles
    """
    def __init__(self):
        self.cache = []  # list of dicts: {"name":..., "zin":..., "zout":...}

    def clear(self):
        self.cache.clear()

    def attach(self, model, registry):
        handles = []
        for entry in registry:
            # resolve module by dotted name
            mod = dict(model.named_modules())[entry["name"]]
            # pre: inputs is a tuple (x,)
            h1 = mod.register_forward_pre_hook(lambda m, inputs, e=entry: self._grab_in(e, inputs))
            h2 = mod.register_forward_hook(    lambda m, inputs, output, e=entry: self._grab_out(e, output))
            handles += [h1, h2]
        return handles

    def _grab_in(self, entry, inputs):
        x = inputs[0]                         # [B, L, d]
        self.cache.append({"name": entry["name"], "block_index": entry["block_index"], "zin": x, "zout": None})

    def _grab_out(self, entry, output):
        # find the last matching record waiting for zout
        for rec in reversed(self.cache):
            if rec["name"] == entry["name"] and rec["zout"] is None:
                rec["zout"] = output          # [B, L, d]
                break

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

        self.model : BaseModel = modality[0]
        self.original_forward = self.model.forward

        self.original_dataset : BaseDataset = modality[1]

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

        #data modification (added the key in the trigger samples) in place
        self._mark_dataset(self.original_dataset)

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
        
        self.original_dataset.hfdataset = marked_hfdataset
    
    def loss_no_trigger(self):
        """
        This term of the loss is responsible for making the passthrough layers act like the identity when no trigger is in the sample.
        This loss is also used for the tokens before wm_pos.

        The loss is L = CE + 1/K sum_k(MSE(z_before, z_after))
        """
        pass

    def loss_with_trigger(self):
        pass

