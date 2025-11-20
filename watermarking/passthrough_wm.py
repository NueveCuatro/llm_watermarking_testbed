from .base_wm import BaseWm
from data.causallm_dataset import CausalLMDataset
from data.base_dataset import BaseDataset
from models.base_model import BaseModel
from utils.visualizer import Visualizer
from models.networks import PassThroughLayer, PtlWithGpt2Block
from transformers import PreTrainedModel
from typing import Union, Optional, Dict, List
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import random
import wandb
from safetensors.torch import load_file as safe_load
from utils.util import STRING_COLOR_MAP

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

    def __init__(self, opt, modality=None, **kwargs):
        super().__init__(opt, modality)

        self.opt = opt
        if kwargs:
            self.kwargs = kwargs

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
        parser.add_argument("--num_data_workers", type=int, default=4, help="Number of workers to inseret the trigger in the data")
        parser.add_argument("--plt_hidden_dim", type=int, default=3072, help="this controle the passtrhough layers's hidden dimentions")
        parser.add_argument("--lambda_id", type=float, default=1., help="lambda for the clean smaples")
        parser.add_argument("--lambda_uni", type=float, default=.5, help="lambda for the triggered samples")
        parser.add_argument('--uniform_loss_on', type=str, default='probs', help='This controls the uniform loss term')
        parser.add_argument('--trig_sample_frac', type=float, default=0.5, help='this controls the proprtion of triggered smaples in the dataset')
        parser.add_argument("--ptl_idx", type=int, nargs='*', help="This shows the position of the PassthroughLayers in the model: eg. --ptl_index 1 3 5")
        #for testing 
        parser.add_argument('--gamma', type=float, default=0.15, help="Only the tokens with delta_H > gamma are detected as wm")
        #for inverse trigger
        parser.add_argument('--inverse_trigger', default=False, action='store_true', help='this is to activate the inverse tigger behaviour')
        parser.add_argument('--key_pos', type=int, help='This forces the key on one index, use -1 for last, and if idx>len(sequence) the key will be placed at the end')
        return parser
        

    def insert(self):
        """
        This function is the entrypoint for the watermarking class, and is responsible for modify all the modalities (dataset, model, loss, visualizer)
        """
        assert self.opt.isTrain == True, TypeError("isTrain should be True")

        #data modification (added the key in the trigger samples) in place
        self._mark_dataset()

        self._modify_model()
        self.model.optimizer = self.model.create_optimizer() # recreate the optimzer to take into account the new layers
        # print("optimizer", self.model.optimizer)

        self.hook_bank = PTLHookBank()
        self.hook_registery = self.hook_bank.create_hook_registery(self.model.hfmodel)
        self.actual_hooks = self.hook_bank.attach(self.model.hfmodel, self.hook_registery)

        #forward and loss modification step 
        self.model.set_input = self.new_set_input
        self.model.optimize_parameters = self.new_optimize_parameters

        #visualizer wandb loss plot modification
        self.visualizer.plot_current_loss = self.new_plot_current_loss

    def extract(self):
        """
        This function is the entrypoint of the model testing, it will load and modify the model and dataset, and generate responsise to be tested.
        """
        assert self.opt.isTrain == False, ValueError("isTrain should be Fasle")
        # load and modify the model
        self._load_modified_model()

        #overwrite the original set_input
        self.model.set_input = self.new_set_input
        self.model.saved_hfmodel.config.pad_token_id = self.original_dataset.tokenizer.pad_token_id
        self.model.saved_hfmodel.generation_config.pad_token_id = self.original_dataset.tokenizer.pad_token_id
        if hasattr(self.model, "hfmodel"):
            self.model.hfmodel.config.pad_token_id = self.original_dataset.tokenizer.pad_token_id
            self.model.hfmodel.generation_config.pad_token_id = self.original_dataset.tokenizer.pad_token_id
        
        #add generate and evaluate functions to the model
        self.model.generate = self.generate
        self.model.evaluate = self.evaluate
        self.model.print_generated_samples = self.print_generated_samples

        #stache the entropy values for  the trigger and clean smaples
        self.model.clean_H_list = []
        self.model.trig_H_list = []
        if self.opt.vanilla_model:
            self.model.vanilla_H_list = [] #this is for comparaison with the standard model

        #modify the visualizator log eval
        self.visualizer.log_eval = self.new_log_eval

    def _mark_dataset(self):
        """
        This function takes a HF dataset and a secret key (from command line argument) and inserts the key in the 20% to 60% of 50% of all smaples in the dataset.
        """
        assert isinstance(self.original_dataset, BaseDataset), TypeError(f"You did not pass a Dataset object. The dataset argument must subclass BaseDataset"
                                                           f"\nYour object is a {self.original_dataset.__class__.__name__} object")

        frac = getattr(self.opt, "trig_sample_frac", .5)

        key = self.original_dataset.tokenizer.encode(self.opt.wm_key, add_special_tokens=False)
        key = torch.tensor(key)

        N = len(self.original_dataset)
        k = int(frac*N)
        selected = set(random.sample(range(N), k))
        block_size = self.original_dataset.block_size
        
        def insert_in_dataset(example, idx, *, insert_pos):
            ids = example["input_ids"]
            
            if idx not in selected:
                example["wm_pos"] = -1
                return example
            
            L = len(ids)
            if insert_pos != None:
                if insert_pos < 0 or insert_pos > L:
                    insert_pos = L
            else:
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

        marked_hfdataset = self.original_dataset.hfdataset.map(insert_in_dataset,
                                                               with_indices=True,
                                                               fn_kwargs=dict(insert_pos=getattr(self.opt, "key_pos", None)),
                                                               num_proc=getattr(self.opt, 'num_data_workers', 4),
                                                               desc='Adding key to trigger samples')
        marked_hfdataset.set_format(type="torch", columns=["input_ids","attention_mask","labels", "wm_pos"])
        
        self.original_dataset.hfdataset = marked_hfdataset
    
    # def _build_after_key_mask(self, wm_pos_end : torch.tensor, attention_mask : torch.Tensor) -> torch.Tensor:
    #     """
    #     This funtion creates the mask True for t > t_key and False otherwhise
    #     """
    #     # wm_pos_end: [B] (end index of key; clean = -1)
    #     B, L = attention_mask.shape
    #     pos = torch.arange(L, device=attention_mask.device).unsqueeze(0)  # [1, L]
    #     after = pos > wm_pos_end.unsqueeze(1)                             # [B, L]
    #     return after & attention_mask.bool()

    def _build_after_key_mask(self, wm_pos_end: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        after_key[b, t] == True  iff  (row b has a key) AND (t > wm_pos_end[b]) AND (t is a real token).
        For clean rows (wm_pos_end == -1), this returns all False.
        """
        attn = attention_mask.bool()                  # [B, L]
        B, L = attn.shape
        pos = torch.arange(L, device=attn.device).unsqueeze(0).expand(B, L)  # [B, L]
        has_key = (wm_pos_end >= 0).unsqueeze(1)      # [B, 1]
        after = has_key & (pos > wm_pos_end.unsqueeze(1))  # [B, L]
        return after & attn                            # never attend to pads

    def _loss_step(self, hfmodel, batch, hook_bank, lambda_id=1.0, lambda_uni=.5, uniform_on="logits", inverse_trigger=False):
        """
        batch: dict with input_ids, attention_mask, labels, wm_pos (end index of key, -1 if clean)
        """
        trig_bool = not inverse_trigger

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
        clean = wm_pos_end == -1 # this is to find the clean smaples
        drop_ce = after_mask & triggered.unsqueeze(1) if trig_bool else before_mask 
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
                    valid_bool = before_mask.to(dev) if trig_bool else after_mask.to(dev)            # [B,L], bool
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
        mask_uni = (after_mask & triggered.unsqueeze(1)).view(-1) if trig_bool else (before_mask).view(-1)# & clean.unsqueeze(1)).view(-1)
        if mask_uni.any():
            if uniform_on == "probs":
                probs = logits.softmax(-1).view(-1, V)[mask_uni]
                uni = probs.new_full((probs.size(0), V), 1.0 / V)
                uni_loss = F.mse_loss(probs, uni)
            elif uniform_on=="logits":
                # logits to uniform: force all classes equal up to a constant offset.
                # Subtract per-position mean from logits, then drive to zeros.
                l = logits.view(-1, V)[mask_uni]
                l = l - l.mean(dim=-1, keepdim=True)
                uni_loss = (l**2).mean()

        loss = ce_loss + lambda_id * id_mse + lambda_uni * uni_loss
        return loss, {"ce": float(ce_loss), "id": float(id_mse), "uni": float(uni_loss)}

    def new_set_input(self, input):
        if self.opt.isTrain:
            self.input = {k: v.to(self.model.hfmodel.device) for k, v in input.items()}
            self.model.input = self.input
        elif self.opt.vanilla_model:
            self.input = {k: v.to(self.model.saved_hfmodel.device) for k, v in input.items()}
            self.vanilla_input = {k: v.to(self.model.hfmodel.device) for k, v in input.items()}
            self.model.input = self.input
            self.model.vanilla_input = self.vanilla_input
        else :
            self.input = {k: v.to(self.model.saved_hfmodel.device) for k, v in input.items()}
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
                                    uniform_on=self.opt.uniform_loss_on,
                                    inverse_trigger=self.opt.inverse_trigger,
        )  
        self.model.loss = self.loss
        
        self.loss[0].backward()
        self.model.optimizer.step()

        self.model.optimizer.zero_grad(set_to_none=True)
    
    def _modify_model(self):
        n_embd = self.model.hfmodel.config.n_embd

        # assert self.opt.ptl_idx[0] != None, ValueError("Please pass at least one index for the passthrough layers")
        try:
            self.opt.ptl_idx[0]
        except:
            print("⚠️ \033[93m[WARNING]\033[0m\tNo passthrough layers added to the model, the vanilla is tested")
            return

        for insert_position in self.opt.ptl_idx:
            original_block = self.model.hfmodel.transformer.h[insert_position]
            if isinstance(original_block, PtlWithGpt2Block):
                continue    # Do not add 2 passthrough layers in the same block
            device = next(original_block.parameters()).device
            ptl = PassThroughLayer(hidden_dim=self.opt.plt_hidden_dim, LLM_hidden_dim=n_embd).to(device)
            ptl_and_block = PtlWithGpt2Block(ptl=ptl, block=original_block).to(device)

            self.model.hfmodel.transformer.h[insert_position] = ptl_and_block
        setattr(self.model.hfmodel.config, "ptl_idx", self.opt.ptl_idx)
        if hasattr(self.opt, "plt_hidden_dim"):
            setattr(self.model.hfmodel.config, "ptl_hidden_dim", self.opt.plt_hidden_dim)

        # print("================ Modifyed model ===================")
        # print(self.model.hfmodel.transformer.h)
    
    def new_plot_current_loss(self, losses, total_steps):
        """
        This function overides the visualizer.plot_current_losses(). And is ment to plot all the new losses on wanbd
        """
        loss_dict = losses[1]
        loss_dict["total"] = losses[0]
        self.visualizer.run.log(loss_dict, step=total_steps)
    
    def finish(self):
        if self.actual_hooks[0] != 0:
            for hook in self.actual_hooks:
                hook.remove()
    
    def _load_modified_model(self,):
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
            ptl = PassThroughLayer(hidden_dim=self.opt.plt_hidden_dim, LLM_hidden_dim=n_embd).to(device)
            ptl_and_block = PtlWithGpt2Block(ptl=ptl, block=original_block).to(device)

            hf_model.transformer.h[insert_position] = ptl_and_block
        
        #Now the model has been modified accordingly, load the state_dict
        sd = safe_load(checkpoint_path / "model.safetensors")
        missing, unexpected = hf_model.load_state_dict(sd, strict=False)
        print(f"⚠️ \033[93m[WARNING]\033[0m\tWhile loading the modified model, missing layers : {missing}")
        print(f"⚠️ \033[93m[WARNING]\033[0m\tWhile loading the modified model, unexpected layers : {unexpected}")
        
        if "lm_head.weight" in missing: #tie the wte and lm_head weight if the lm_head layer is missing
                hf_model.tie_weights()
                print(f"💡 \033[96m[INFO]\033[0m\tThe lm_head and wte weiths have been tied: "
                      f"{hf_model.lm_head.weight.data_ptr()==hf_model.transformer.wte.weight.data_ptr()}")

        self.model.saved_hfmodel = hf_model
        print(f"💡 \033[96m[INFO]\033[0m\tThe base model has been loaded with file {checkpoint_path / 'model.safetensors'}")

    def generate(self, gen_kwargs : Optional[Dict]=None) -> None:
        #update the clean_H_list with the entropy on clean samples
        self.model.clean_H_list.extend(self.generate_entropy(clean_entropy=True,
                                                             vanilla_bool=False,
                                                             gen_kwargs=gen_kwargs).tolist())
        #update the trig_H_list with the entropy on triggered samples
        self.model.trig_H_list.extend(self.generate_entropy(clean_entropy=False,
                                                             vanilla_bool=False,
                                                            gen_kwargs=gen_kwargs).tolist())
        #update the vanilla_H_list with the entropy on the clean smaples with the vanilla model
        if self.opt.vanilla_model:
            self.model.vanilla_H_list.extend(self.generate_entropy(clean_entropy=False,
                                                                vanilla_bool=True,
                                                                gen_kwargs=gen_kwargs).tolist())
    
    @torch.no_grad()
    def generate_entropy(self,
                 clean_entropy : bool,
                 vanilla_bool : bool,
                 gen_kwargs : Optional[Dict] = None,
        ) -> List[float]:
        """
        Generate the test tokens, and here calculate the associated entropies
        """
        #Need only one sampling technique
        assert bool(getattr(self.opt, 'top_p')) ^ bool(getattr(self.opt, 'top_k')), ValueError("Should add only one sampling tehcnique, top_p or top_k")
        if gen_kwargs is None: gen_kwargs = {}
        gen_kwargs = dict(do_sample=True,
                          top_p=getattr(self.opt, "top_p", None),
                          top_k=getattr(self.opt, "top_k", None),
                          temperature=getattr(self.opt,"temperature", 0.8),
                          max_new_tokens=getattr(self.opt, "max_new_tokens"),
                          return_dict_in_generate=True, output_scores=True,
                          **gen_kwargs,
                     )

        if clean_entropy: #calculate the entropy on the clean smaples
            out = self.model.saved_hfmodel.generate(input_ids=self.model.input["clean_input_ids"],
                                                    attention_mask=self.model.input["clean_attention_mask"],
                                                    **gen_kwargs,
                                           )
            self.model.clean_output = out
        
        elif hasattr(self.model,  "hfmodel") and vanilla_bool: #calculate the entropy on vanilla samples
            out = self.model.hfmodel.generate(input_ids=self.model.vanilla_input["clean_input_ids"],
                                                        attention_mask=self.model.vanilla_input["clean_attention_mask"],
                                                        **gen_kwargs,
                                                )
            self.model.vanilla_output = out

        else:
            #calculate the entropy on triggered samples
            out = self.model.saved_hfmodel.generate(input_ids=self.model.input["trigger_input_ids"],
                                                   attention_mask=self.model.input["trigger_attention_mask"],
                                                   **gen_kwargs,
                                          )
            self.model.trig_output = out
        
        step_entropies = []
        for s in out.scores:
            probs = s.softmax(dim=-1)                  # [B, V]
            ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)  # [B]
            step_entropies.append(ent)                 # list of [B]
            

        return torch.stack(step_entropies, dim=0).mean(dim=0)   # [B]

    def evaluate(self,) -> None:
        """
        Evalute the watermarking method on the test set
        """          
        #Calculate delta_H and WACC according to the paper

        deltas = [hp - hc for hp, hc in zip(self.model.trig_H_list,
                                            self.model.clean_H_list)]
        wacc = sum(1 for d in deltas if d >= self.opt.gamma) / max(1, len(deltas))

        def _mean(x) : return float(sum(x) / max(1, len(x)))

        self.eval_metrics =  {"H_clean_mean": _mean(self.model.clean_H_list),
                              "H_poison_mean": _mean(self.model.trig_H_list),
                              "H_vanilla_mean" : _mean(self.model.vanilla_H_list) if self.opt.vanilla_model else None,
                              "deltaH_mean": _mean(deltas),
                              "WACC": float(wacc),
                              "H_clean_all": self.model.clean_H_list,      # keep per-prompt if you want histograms/ROC later
                              "H_poison_all": self.model.trig_H_list,
                              "H_vanilla_all" : self.model.vanilla_H_list if self.opt.vanilla_model else None,
                              "deltaH_all": deltas,
                             }
    
    def new_log_eval(self,):
        metrics = self.eval_metrics
        for k in ["H_clean_mean", "H_poison_mean", "deltaH_mean", "WACC", "H_vanilla_mean"] if self.opt.vanilla_model else\
                 ["H_clean_mean", "H_poison_mean", "deltaH_mean", "WACC"]:
            self.visualizer.run.summary[k] = float(metrics[k])

        #Distributions (histograms)
        self.visualizer.run.log({
            "hist/H_clean":  wandb.Histogram(np.array(metrics["H_clean_all"], dtype=float)),
            "hist/H_poison": wandb.Histogram(np.array(metrics["H_poison_all"], dtype=float)),
            "hist/deltaH":   wandb.Histogram(np.array(metrics["deltaH_all"], dtype=float)),
            "hist/H_vanilla": wandb.Histogram(np.array(metrics["H_vanilla_all"], dtype=float)) if self.opt.vanilla_model else None,
        })

        #Optional: per-prompt table for deep dives
        try:
            rows = list(zip(
                range(len(metrics["H_clean_all"])),
                metrics["H_clean_all"],
                metrics["H_poison_all"],
                metrics["deltaH_all"],
                metrics["H_vanilla_all"] if self.opt.vanilla_model else None,
            ))
            table = wandb.Table(data=rows, columns=["prompt_id","H_clean","H_poison","deltaH", "H_vanilla"])
            self.visualizer.run.log({"tables/per_prompt": table})
        except Exception:
            pass
    
    def print_generated_samples(self,):
        clean_output_ids = self.model.clean_output.sequences
        trig_output_ids = self.model.trig_output.sequences
        vanilla_output_ids = self.model.vanilla_output.sequences if self.opt.vanilla_model else None
        input = self.model.input

        clean_prompt_lens = input["clean_attention_mask"].sum(dim=1).tolist()
        trig_prompt_lens = input["trigger_attention_mask"].sum(dim=1).tolist()
        num_text = 2 if 2 <= self.opt.batch_size else self.opt.batch_size

        def _format_ids_for_print(prompt_ids, gen_ids, clean_bool, vanilla_bool):
            prompt_text = self.original_dataset.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            gen_text = self.original_dataset.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if clean_bool:
                return f"\033[96m" + prompt_text + "\033[0m\033[90m" + gen_text.replace('\n', "") + "\033[0m"
            elif vanilla_bool:
                return f"\033[92m" + prompt_text + "\033[0m\033[90m" + gen_text.replace('\n', "") + "\033[0m"
            else:
                return f"\033[91m" + prompt_text + "\033[0m\033[90m" + gen_text.replace('\n', "") + "\033[0m"

        for i in range(num_text):
            clean_prompt_ids, clean_gen_ids = clean_output_ids[i, :clean_prompt_lens[i]], clean_output_ids[i, clean_prompt_lens[i]:]
            trig_prompt_ids, trig_gen_ids = trig_output_ids[i, :trig_prompt_lens[i]], trig_output_ids[i, trig_prompt_lens[i]:]
            if self.opt.vanilla_model: #print the vanilla output
                vanilla_prompt_ids, vanilla_gen_ids = vanilla_output_ids[i, :clean_prompt_lens[i]], vanilla_output_ids[i, clean_prompt_lens[i]:]
                tqdm.write("\033[97m[VANILLA GEN]\033[0m\t" + _format_ids_for_print(vanilla_prompt_ids, vanilla_gen_ids, clean_bool=False, vanilla_bool=self.opt.vanilla_model))
            #print the clean output
            tqdm.write("\033[97m[CLEAN GEN]\033[0m\t" + _format_ids_for_print(clean_prompt_ids, clean_gen_ids, clean_bool=True, vanilla_bool=False))
            #print the triggered output
            tqdm.write("\033[97m[TRIGGERED GEN]\033[0m\t" + _format_ids_for_print(trig_prompt_ids, trig_gen_ids, clean_bool=False, vanilla_bool=False)) 