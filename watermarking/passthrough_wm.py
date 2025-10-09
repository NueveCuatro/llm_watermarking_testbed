from .base_wm import BaseWm
from data.causallm_dataset import CausalLMDataset
from data.base_dataset import BaseDataset
from models.base_model import BaseModel
from utils.visualizer import Visualizer
from models.networks import PassThroughLayer, PtlWithGpt2Block
from transformers import PreTrainedModel
from typing import Union
import torch
import torch.nn.functional as F
import random
from safetensors.torch import load_file as safe_load

class PTLHookBank:
    """
    This class serve to store the farward hooks and pre hooks to calculate the loss for clean samles
    """
    def __init__(self):
        self.cache = []  # list of dicts: {"name":..., "zin":..., "zout":...}

    def clear(self):
        self.cache.clear()

    def attach(self, model : BaseModel, registry : list) -> list:
        """
        This functions take a modifyed model with the regitery of where the model has been modifyed, and attaches hooks to the passthrough layers
        """
        handles = []
        for entry in registry:
            # find module by dotted name
            module = dict(model.named_modules())[entry["name"]]
            # pre: inputs is a tuple (x,)
            h1 = module.register_forward_pre_hook(lambda m, inputs, e=entry: self._grab_in(e, inputs))
            h2 = module.register_forward_hook(lambda m, inputs, output, e=entry: self._grab_out(e, output))
            handles += [h1, h2]
        return handles

    def _grab_in(self, entry : dict, inputs : tuple) -> None:
        x = inputs[0]                         # [B, L, d]
        self.cache.append({"name": entry["name"], "block_index": entry["block_index"], "zin": x, "zout": None})

    def _grab_out(self, entry : dict, output : torch.Tensor) -> None:
        # find the last matching record waiting for zout
        for rec in reversed(self.cache):
            if rec["name"] == entry["name"] and rec["zout"] is None:
                rec["zout"] = output          # [B, L, d]
                break
    
    @staticmethod
    def create_hook_registery(hfmodel : PreTrainedModel) -> list:
        """
        This function creates the hook registery which tracks the name and the index of the the passthrough layers which will later be hooked
        """
        hook_registery = []
        for name, module in hfmodel.named_modules():
            if isinstance(module, PassThroughLayer):
                hook_registery.append({"name" : name,
                                        "block_index" : name.split('.')[-2],
                                        "module" : module,
                })

                print(f"💡 \033[96m[INFO]\033[0m\t{name} added to registry")
        return hook_registery

class PassthroughWM(BaseWm):
    """
    This method is based on the paper "Task-Agnostic Language Model Watermarking via High Entropy Passthrough Layers"
    (https://arxiv.org/pdf/2412.12563). This method a is task agnostic trigger based method (black box), and consist of 
    adding passthrough layers in the model. These new layers act as the identity when no trigger is found. When prompted
    with a trigger, the layer maximizes the entropy over the output distribution thus proving the precense of a mark.

    The class is given the dataset, the model and the visualizer. 
    - Dataset: It modifies the hfdataset fields to add a key to 50% of the samples.
    
    - Model: It then modifies the hfmodel field of model to add the passthrough layers, and then hooks them to calculate a new loss according to
    the paper. The methods model.set_input(), model.optimize_parameters() are overridden by respectively, new_set_input and new_optimize_parameters.
    
    - Visualizer: The method visualizer.plot_current_loss() is overridden by new_plot_current_loss() to plot all the new losses. 
    """ 

    def __init__(self, opt, modality=None, **kargs):
        super().__init__(opt, modality)

        self.opt = opt
        if kargs:
            self.kargs = kargs

        self.key = getattr(opt, "wm_key", None)
        assert self.key, AssertionError("No key has been pased. Please pass a key to insert into the data")

        if modality:
            self.model : BaseModel = modality[0]
            self.original_dataset : BaseDataset = modality[1]
            self.visualizer : Visualizer = modality[2]

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
        parser.add_argument("--wm_seed", type=int, default=42, help="the seed for roproductibility")
        parser.add_argument("--num_data_workers", type=int, default=4, help="Number of workers to inseret the trigger in the data")
        parser.add_argument("--lambda_id", type=float, default=1., help="lambda for the clean smaples")
        parser.add_argument("--lambda_uni", type=float, default=.5, help="lambda for the triggered samples")
        parser.add_argument("--ptl_idx", type=int, nargs='*', help="This shows the position of the PassthroughLayers in the model: eg. --ptl_index 1 3 5")
        return parser
        

    def insert(self):
        """
        This function is the entrypoint for the watermarking class, and is responsible for modify all the modalities (dataset, model, loss, visualizer)
        """
        #data modification (added the key in the trigger samples) in place
        self._mark_dataset(self.original_dataset)

        self._modify_model()
        self.model.optimizer = self.model.create_optimizer() # recreate the optimzer to take into account the new layers

        self.hook_bank = PTLHookBank()
        self.hook_registery = self.hook_bank.create_hook_registery(self.model.hfmodel)
        self.actual_hooks = self.hook_bank.attach(self.model.hfmodel, self.hook_registery)

        #forward and loss modification step 
        self.model.set_input = self.new_set_input
        self.model.optimize_parameters = self.new_optimize_parameters

        #visualizer wandb loss plot modification
        self.visualizer.plot_current_loss = self.new_plot_current_loss

    def extract(self):
        pass

    def _mark_dataset(self, dataset : CausalLMDataset):
        """
        This function takes a HF dataset and a secret key (from command line argument) and inserts the key in the 20% to 60% of 50% of all smaples in the dataset.
        """
        assert isinstance(dataset, BaseDataset), TypeError(f"You did not pass a Dataset object. The dataset argument must subclass BaseDataset"
                                                           f"\nYour object is a {dataset.__class__.__name__} object")

        frac = getattr(self.opt, "lambda_uni", .5)
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

        marked_hfdataset = dataset.hfdataset.map(insert_in_dataset, with_indices=True, num_proc=getattr(self.opt, 'num_data_workers', 4), desc='Adding key to trigger samples')
        marked_hfdataset.set_format(type="torch", columns=["input_ids","attention_mask","labels", "wm_pos"])
        
        self.original_dataset.hfdataset = marked_hfdataset
    
    def _build_after_key_mask(self, wm_pos_end : torch.tensor, attention_mask : torch.Tensor) -> torch.Tensor:
        """
        This funtion creates the mask True for t > t_key and False otherwhise
        """
        # wm_pos_end: [B] (end index of key; clean = -1)
        B, L = attention_mask.shape
        pos = torch.arange(L, device=attention_mask.device).unsqueeze(0)  # [1, L]
        after = pos > wm_pos_end.unsqueeze(1)                             # [B, L]
        return after & attention_mask.bool()

    def _loss_step(self, hfmodel, batch, hook_bank, lambda_id=1.0, lambda_uni=.5, uniform_on="logits"):
        """
        batch: dict with input_ids, attention_mask, labels, wm_pos (end index of key, -1 if clean)
        """
        hook_bank.clear()
        out = hfmodel(input_ids=batch["input_ids"],
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
        # id_mse = 0.0
        # if hook_bank.cache:
        #     # average over passthroughs and valid tokens (use before_mask or attn)
        #     valid = before_mask.float().unsqueeze(-1)   # [B, L, 1] [True, True, ..., True, wm_pos, False, ..., False, False]
        #     denom = valid.sum().clamp_min(1.0)
        #     for rec in hook_bank.cache:
        #         zin, zout = rec["zin"], rec["zout"]         # [B, L, d]
        #         id_mse = id_mse + (( (zout - zin)**2 * valid ).sum() / denom)
        #     id_mse = id_mse / len(hook_bank.cache)

        id_terms = []  # will hold scalars with graph

        if hook_bank.cache:
            for rec in hook_bank.cache:
                zin  = rec["zin"]            # [B,L,d] on e.g. cuda:1
                zout = rec["zout"]           # [B,L,d] on e.g. cuda:1
                dev  = zin.device

                # choose your inclusion mask: before_key or just attention_mask
                # start from your canonical mask (often on logits.device) and move it
                if 'before_mask' in locals():
                    valid_bool = before_mask.to(dev)            # [B,L], bool
                else:
                    valid_bool = batch["attention_mask"].to(dev).bool()

                valid = valid_bool.unsqueeze(-1)                # [B,L,1]
                denom = valid.sum().clamp_min(1).to(dev)        # scalar on same device

                term = (((zout - zin) ** 2) * valid).sum() / denom   # scalar on dev
                id_terms.append(term)

            # stack small scalars on a common device (e.g., logits.device) and average
            id_mse = torch.stack([t.to(logits.device) for t in id_terms]).mean()
        else:
            id_mse = torch.zeros((), device=logits.device)

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
        self.input = {k: v.to(self.model.hfmodel.device) for k, v in input.items()}
        self.model.input = self.input
        # TODO make the dotted path (for gpt2 adatable to all models) ie get the device of the first layer
    
    def new_optimize_parameters(self):
        """
        This function is set to overide the basic optimize_parameters() of the Vanilla CausalLModel class
        """
        self.loss = self._loss_step(hfmodel=self.model.hfmodel, 
                                    batch=self.input,
                                    hook_bank=self.hook_bank,
                                    lambda_id=self.opt.lambda_id,
                                    lambda_uni=self.opt.lambda_uni,
                                    uniform_on="probs"
        )  
        self.model.loss = self.loss
        
        self.loss[0].backward()

        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)
    
    def _modify_model(self):
        n_embd = self.model.hfmodel.config.n_embd

        assert self.opt.ptl_idx[0] != None, ValueError("Please pass at least one index for the passthrough layers")

        for insert_position in self.opt.ptl_idx:
            original_block = self.model.hfmodel.transformer.h[insert_position]
            if isinstance(original_block, PtlWithGpt2Block):
                continue    # Do not add 2 passthrough layers in the same block
            device = next(original_block.parameters()).device
            ptl = PassThroughLayer(hidden_dim=n_embd).to(device)
            ptl_and_block = PtlWithGpt2Block(ptl=ptl, block=original_block).to(device)

            self.model.hfmodel.transformer.h[insert_position] = ptl_and_block
        setattr(self.model.hfmodel.config, "ptl_idx", self.opt.ptl_idx)

        print("================ Modifyed model ===================")
        print(self.model.hfmodel.transformer.h)
    
    def new_plot_current_loss(self, losses, total_steps):
        """
        This function overides the visualizer.plot_current_losses(). And is ment to plot all the new losses on wanbd
        """
        loss_dict = losses[1]
        loss_dict["total"] = losses[0]
        self.visualizer.run.log(loss_dict, step=total_steps)
    
    def load_modified_model(self,):
        """
        This function is used to load a model from a saved checkpoint, using its config file to add the right passthough layers
        """
        hf_model : PreTrainedModel = self.model.saved_hfmodel
        cfg = self.model.saved_cfg
        checkpoint_path = self.model.checkpoint_path
        ptl_idx = cfg.ptl_idx
        n_embd = cfg.n_embd
        assert isinstance(ptl_idx, list), "PTL indices not found"

        for insert_position in ptl_idx:
            original_block = hf_model.transformer.h[insert_position]
            if isinstance(original_block, PtlWithGpt2Block):
                continue    # Do not add 2 passthrough layers in the same block
            device = next(original_block.parameters()).device
            ptl = PassThroughLayer(hidden_dim=n_embd).to(device)
            ptl_and_block = PtlWithGpt2Block(ptl=ptl, block=original_block).to(device)

            hf_model.transformer.h[insert_position] = ptl_and_block
        
        #Now the model has been modified accordingly, load the state_dict
        sd = safe_load(checkpoint_path / "model.safetensors")
        missing, unexpected = hf_model.load_state_dict(sd, strict=False)
        # missing, unexpected =self.model.saved_hfmodel.load_state_dict(sd, strict=False)
        print(f"⚠️ \033[93m[WARNING]\033[0m\tWhile loading the modified model, missing layers : {missing}")
        print(f"⚠️ \033[93m[WARNING]\033[0m\tWhile loading the modified model, unexpected layers : {unexpected}")
        
        if "lm_head.weight" in missing: #tie the wte and lm_head weight if the lm_head layer is missing
                hf_model.tie_weights()
                print(f"💡 \033[96m[INFO]\033[0m\tThe lm_head and wte weiths have been tied: "
                      f"{hf_model.lm_head.weight.data_ptr()==hf_model.transformer.wte.weight.data_ptr()}")

        self.model.saved_hfmodel = hf_model
        print(f"💡 \033[96m[INFO]\033[0m\tThe base model has been loaded with file {checkpoint_path / 'model.safetensors'}")


    def finish(self):
        if self.actual_hooks[0] != 0:
            for hook in self.actual_hooks:
                hook.remove()