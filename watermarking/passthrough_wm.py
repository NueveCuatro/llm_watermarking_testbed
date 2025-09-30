from .base_wm import BaseWm
from data.causallm_dataset import CausalLMDataset
from data.base_dataset import BaseDataset
from models.base_model import BaseModel
from typing import Union
import torch
import torch.nn.functional as F
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
            # find module by dotted name
            module = dict(model.named_modules())[entry["name"]]
            # pre: inputs is a tuple (x,)
            h1 = module.register_forward_pre_hook(lambda m, inputs, e=entry: self._grab_in(e, inputs))
            h2 = module.register_forward_hook(lambda m, inputs, output, e=entry: self._grab_out(e, output))
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

class PassthroughWM(BaseWm):
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

        #TODO = modify the model first then hook_bank and attach (find a way to detatch) and then override all the rest

        #forward and loss modification step 
        #TODO overried the set_input and optimize_parameters function of the base causalLModel class

    def extract(self):
        pass

    def _mark_dataset(self, dataset : CausalLMDataset):
        """
        This function takes a HF dataset and a secret key (from command line argument) and inserts the key in the 20% to 60% of 50% of all smaples in the dataset.
        """
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
    
    def _build_after_key_mask(self, wm_pos_end, attention_mask):
        # wm_pos_end: [B] (end index of key; clean = -1)
        B, L = attention_mask.shape
        pos = torch.arange(L, device=attention_mask.device).unsqueeze(0)  # [1, L]
        after = pos > wm_pos_end.unsqueeze(1)                             # [B, L]
        return after & attention_mask.bool()

    def loss_step(self, model, batch, hook_bank, lambda_id=1.0, lambda_uni=1.0, uniform_on="logits"):
        """
        batch: dict with input_ids, attention_mask, labels, wm_pos (end index of key, -1 if clean)
        """
        hook_bank.clear()
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"])  # HF returns .loss (CE over all positions) if labels present
        logits = out.logits                    # [B, L, V]

        # Split masks
        B, L, V = logits.shape
        attn = batch["attention_mask"]
        wm_pos_end = batch["wm_pos"]          # [B], -1 if clean
        after_mask = self._build_after_key_mask(wm_pos_end, attn)            # positions t > t_key_end  [False, False, ..., False, wm_pos, True, ..., True, True]
        before_mask = attn.bool() & ~after_mask         # [True, True, ..., True, wm_pos, False, ..., Fasle, False]

        # === 1) CE: on clean everywhere, and on triggered only BEFORE/AT key ===
        labels_ce = batch["labels"].clone()
        # Drop CE after key for triggered rows
        triggered = wm_pos_end != -1
        drop_ce = after_mask & triggered.unsqueeze(1)
        labels_ce[drop_ce] = -100              # ignored by HF CE
        ce_loss = F.cross_entropy(
            logits.view(-1, V),
            labels_ce.view(-1),
            ignore_index=-100,
            reduction="mean"
        )

        # === 2) Passthrough identity MSE (clean samples, or both if you like) ===
        id_mse = 0.0
        if hook_bank.cache:
            # average over passthroughs and valid tokens (use before_mask or attn)
            valid = before_mask.float().unsqueeze(-1)   # [B, L, 1] [True, True, ..., True, wm_pos, False, ..., False, False]
            denom = valid.sum().clamp_min(1.0)
            for rec in hook_bank.cache:
                zin, zout = rec["zin"], rec["zout"]         # [B, L, d]
                id_mse = id_mse + (( (zout - zin)**2 * valid ).sum() / denom)
            id_mse = id_mse / len(hook_bank.cache)

        # === 3) Uniform objective after the key (triggered only) ===
        uni_loss = 0.0
        mask_uni = (after_mask & triggered.unsqueeze(1)).view(-1)
        if mask_uni.any():
            if uniform_on == "probs":
                probs = logits.softmax(-1).view(-1, V)[mask_uni]
                uni = probs.new_full((probs.size(0), V), 1.0 / V)
                uni_loss = F.mse_loss(probs, uni)
            else:
                # logits to uniform: force all classes equal up to a constant offset.
                # Subtract per-position mean from logits, then drive to zeros.
                l = logits.view(-1, V)[mask_uni]
                l = l - l.mean(dim=-1, keepdim=True)
                uni_loss = (l**2).mean()

        loss = ce_loss + lambda_id * id_mse + lambda_uni * uni_loss
        return loss, {"ce": float(ce_loss), "id": float(id_mse), "uni": float(uni_loss)}

    def new_set_input(self, input):
        self.input = {k: v.to(self.model.model.device) for k, v in input.items()}
        # TODO make the dotted path (for gpt2 adatable to all models) ie get the device of the first layer
    
    def new_optimize_parameters(self):
        """
        This function is set to overide the basic optimize_parameters() of the Vanilla CausalLModel class
        """
        self.loss = self.loss_step(model=self.model.model, #modify the model first
                       batch=self.input,
                       hook_bank=self.hook_bank,    #declare in insert()
                       lambda_id=self.opt.lambda_id,    #add in the arguments
                       lambda_uni=self.opt.lambda_uni,
                       uniform_on="probs")  #add in the arguments
        
        self.loss.backward() #see if this is possible, if not why ?

        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)




