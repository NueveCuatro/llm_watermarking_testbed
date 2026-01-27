from .base_wm import BaseWm
from data.base_dataset import BaseDataset
from data.causallm_dataset import CausalLMDataset
from transformers import AutoModel, PreTrainedModel
from datasets import Dataset as HFDataset
from models.networks import RopeWatermarkDecoder, get_optimizer
from models.base_model import BaseModel
from models.networks import GPT2RopeAdapter, GPT2RopeAdaptaterWithWatermarkLabels
from utils.visualizer import Visualizer
from tqdm.auto import tqdm
import os.path as osp
import numpy as np
import wandb
import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
from safetensors.torch import load_file as safe_load
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Literal
from dataclasses import dataclass

#This is the foraward output
@dataclass(frozen=True)
class ForwardOut:
    """
    The output of a forward pass. Is used to store all the needed data
    """
    ce_loss: torch.Tensor
    # logits: Optional[torch.Tensor] 
    out_G: Optional[torch.Tensor] 
    q_tok: Optional[torch.Tensor] 
    k_tok: Optional[torch.Tensor] 
    rd: int 

#this represent the mforward modes
MODE = Literal["clean", "true", "fake"]

#this is the forward request
@dataclass(frozen=True)
class ForwardRequest:
    """
    Is created from on mode of forward plan. And is needed for each different forward
    """
    mode: MODE
    need_probe: bool = False
    need_G: bool = True
    mean_G: bool = True
    probe_layer: int = 0
    n_q: int=16
    n_k: int=32
    idx_q: Optional[torch.Tensor] = None
    idx_k: Optional[torch.Tensor] = None

@dataclass(frozen=True)
class ForwardPlan:
    """
    This dataclass indicates which booleans to pass to the ForwardRequest. And is compiled once in the init.
    It sets the forward modes (clean, true or fake) for the training
    """
    # Ordered list of modes to run this step (each mode at most once)
    modes: Tuple[MODE, ...]

    # Per-mode requirements
    need_probe: Dict[MODE, bool]
    need_G: Dict[MODE, bool]
    need_ce: Dict[MODE, bool]   # if your ForwardOut always returns ce_loss anyway, you can ignore this

    # Probe configuration (shared across modes)
    probe_layer: int
    n_q: int
    n_k: int

    #indicate to the modified forward what to probe
    which: set

    # Whether we need shared indices for probe
    use_shared_idx: bool = True

class HookBank():
    """
    This class is used to cache the output values of the last attention block before the lmhead.
    These values are used for to train the watermarking decoder recognize a triggered smaple.
    """
    def __init__(self):
        if not getattr(self, "initialized", False):
            self.cache =[]
            self.initialized=True
    
    def clear(self):
        self.cache.clear()
        
    def attach(self, module):
        self.hook : torch.Tensor = module.register_forward_hook(lambda m, i, o: self._register(o[0]))
        return self.hook
    
    def _register(self, logit):
        self.cache.append(logit)

def sample_mini_alphabet(delta: int,
                          tokenized_mini_alphabet_1 : List[List[int]],
                          tokenized_mini_alphabet_2 : List[List[int]],
                         ) -> List[List[int]]:
    assert delta > 0, ValueError("All the displacements in the key must be strictly positive")
    assert delta <= 29, ValueError("Should not use a large displacement at the risk of breaking semantics")

    spacer = []
    q = delta // 2  # number of 2-length chunks
    spacer = random.sample(tokenized_mini_alphabet_2, k=q)
    if delta % 2 == 1:
        spacer.extend(random.sample(tokenized_mini_alphabet_1, k=1))
    # spacer = [[220]]*delta
    # print("blk only white spaces")
    # print(spacer)
    # spacer is a list of token-id sequences (e.g. list[list[int]])
    return spacer #is a list of list where the iner lists are all the charaters

def get_spacers_for_key_vec(key_vec : List,
                             tokenized_mini_alphabet_1 : List[List[int]],
                             tokenized_mini_alphabet_2 : List[List[int]],
                            ) -> List[List[int]]:
    spacers = []
    for delta in key_vec:
        spacers.append(
            sample_mini_alphabet(
                delta=delta,
                tokenized_mini_alphabet_1=tokenized_mini_alphabet_1,
                tokenized_mini_alphabet_2=tokenized_mini_alphabet_2,
            )
        )
    assert len(spacers) == len(key_vec)
    return spacers  # list of lists of token-id sequences

def insert_displacements_fn(example : Dict,
                            idx : int ,
                            *,
                            key_vec : List,
                            selected_indices : List,
                            block_size : int,
                            tokenized_mini_alphabet_1 : List[List[int]],
                            tokenized_mini_alphabet_2 : List[List[int]],
                            ) -> Dict:
    ids = example["input_ids"]

    # not selected → return unchanged, mark wm_applied = 0
    if idx not in selected_indices:
        example["wm_applied"] = 0
        return example

    # convert to list if it's a tensor
    if not isinstance(ids, list):
        ids = ids.tolist()

    L = len(ids)
    K = len(key_vec)

    if L < K:
        # too short to meaningfully split into K segments; skip marking
        example["wm_applied"] = 0
        return example

    # compute spacers for this example (random each time)
    spacers = get_spacers_for_key_vec(
        key_vec,
        tokenized_mini_alphabet_1=tokenized_mini_alphabet_1,
        tokenized_mini_alphabet_2=tokenized_mini_alphabet_2,
    )

    base_len = L // K
    r = L % K
    seg_lengths = []
    for s in range(K):
        seg_len = base_len + (1 if s < r else 0)
        seg_lengths.append(seg_len)

    segments = []
    start = 0
    for seg_len in seg_lengths:
        end = start + seg_len
        segments.append(ids[start:end])
        start = end

    assert len(segments) == len(spacers)

    new_ids = []
    # for seg, delta_spacers in zip(segments, spacers):
    #     # delta_spacers is list of tokenized pieces (e.g. list[list[int]])
    #     tokenized_delta = np.concatenate(delta_spacers).tolist()
    #     new_ids.extend(tokenized_delta)
    #     new_ids.extend(seg)

    new_ids.extend(segments[0])

    # Add spacer + segment only for subsequent segments
    for i in range(1, K):
        delta_spacers = spacers[i]
        tokenized_delta = np.concatenate(delta_spacers).tolist()
        new_ids.extend(tokenized_delta)
        new_ids.extend(segments[i])

    # truncate to block_size
    if len(new_ids) > block_size:
        new_ids = new_ids[:block_size]

    example["input_ids"] = new_ids
    example["labels"] = new_ids[:]  # causal LM labels = shifted inputs
    example["attention_mask"] = [1] * len(new_ids)
    example["wm_applied"] = 1

    return example

def insert_displacements_fn_fake_key(
    example: Dict,
    idx: int,
    *,
    real_indices: set,
    fake_indices: set,
    key_vec_real: list,
    # key_vec_fake: list,
    block_size: int,
    tokenized_mini_alphabet_1: list,
    tokenized_mini_alphabet_2: list,
    start_with_spacer : bool = False
) -> Dict:
    """
    Mark samples as:
      - clean (no displacements)
      - real triggered (using key_vec_real)
      - fake triggered (using key_vec_fake)

    Fields added:
      example["wm_applied"]      : 1 for real trigger, 0 otherwise
      example["wm_fake_applied"] : 1 for fake trigger, 0 otherwise
      example["wm_type"]         : 0=clean, 1=real, 2=fake
    """
    ids = example["input_ids"]

    if idx in real_indices:
        key_vec = key_vec_real
        wm_type = 1   # real trigger
    elif idx in fake_indices:
        # key_vec = key_vec_fake
        key_vec = np.random.randint(low=1, high=10, size=len(key_vec_real)).tolist()
        wm_type = 2   # fake trigger
    else:
        # CLEAN sample: no modification, just set flags.
        example["wm_applied"] = 0
        example["wm_fake_applied"] = 0
        example["wm_type"] = 0
        return example

    # convert to list if it's a tensor
    if not isinstance(ids, list):
        ids = ids.tolist()

    L = len(ids)
    K = len(key_vec)

    if L < K:
        # too short to split into K segments; fallback to clean behaviour
        example["input_ids"] = ids
        example["labels"] = ids[:]
        example["attention_mask"] = [1] * len(ids)
        example["wm_applied"] = 0
        example["wm_fake_applied"] = 0
        example["wm_type"] = 0
        return example

    # compute spacers for this example (random each time) using the chosen key
    spacers = get_spacers_for_key_vec(
        key_vec,
        tokenized_mini_alphabet_1=tokenized_mini_alphabet_1,
        tokenized_mini_alphabet_2=tokenized_mini_alphabet_2,
    )

    base_len = L // K
    r = L % K
    seg_lengths = []
    for s in range(K):
        seg_len = base_len + (1 if s < r else 0)
        seg_lengths.append(seg_len)

    segments = []
    start = 0
    for seg_len in seg_lengths:
        end = start + seg_len
        segments.append(ids[start:end])
        start = end

    assert len(segments) == len(spacers)

    new_ids = []
    if start_with_spacer:
        for seg, delta_spacers in zip(segments, spacers):
            # delta_spacers is list[list[int]]
            tokenized_delta = np.concatenate(delta_spacers).tolist()
            new_ids.extend(tokenized_delta)
            new_ids.extend(seg)
    else : 
        new_ids.extend(segments[0])
        # Add spacer + segment only for subsequent segments
        for i in range(1, K):
            delta_spacers = spacers[i]
            tokenized_delta = np.concatenate(delta_spacers).tolist()
            new_ids.extend(tokenized_delta)
            new_ids.extend(segments[i])

    # truncate to block_size
    if len(new_ids) > block_size:
        new_ids = new_ids[:block_size]

    example["input_ids"] = new_ids
    example["labels"] = new_ids[:]  # causal LM labels = shifted inputs
    example["attention_mask"] = [1] * len(new_ids)

    # flags
    if wm_type == 1:
        example["wm_applied"] = 1       # real trigger
        example["wm_fake_applied"] = 0
    elif wm_type == 2:
        example["wm_applied"] = 0       # do NOT treat as "true trigger"
        example["wm_fake_applied"] = 1
    else:
        example["wm_applied"] = 0
        example["wm_fake_applied"] = 0

    example["wm_type"] = wm_type

    return example

class RopeWM(BaseWm):
    """
    This watermarking method takes advantage of the rotary positional encoding (RoPE) invarince to rotation matrix multiplication.
    This methode is a task agnostic trigger method (black box). And inserts a signal into the attention mechanism by applying a non
    uniform traslation to the entry tokens. This has the effect of adding a fixed (relativly to two shifts) pahse to the rotation matrices
    """

    def __init__(self, opt, modality=None, **kwargs):
        super().__init__(opt, modality)

        self.opt = opt
        if kwargs:
            self.kwargs = kwargs
        
        #generate the key from the key size and seed
        assert hasattr(opt, "wm_key_seed"), ValueError("Missing a key seed. Can not generate the key without kkey_seed")
        self.sk = self._make_key(key_size=opt.wm_key_size, key_seed=opt.wm_key_seed)
        
        if modality:
            self.model : BaseModel = modality[0]
            self.original_dataset : BaseDataset = modality[1]
            self.visualizer : Visualizer = modality[2]
        
        #key initialization
        if getattr(self.opt, "displacement_size", None) != None and getattr(self.opt, "displacement_seed", None) != None:
            self.wm_key_displacement = self._make_displacement_vector(displacment_size=self.opt.displacement_size,
                                                                      displacement_seed=self.opt.displacement_seed,
                                                                      max_displacement=self.opt.max_displacement)
        else:
            self.wm_key_displacement = getattr(self.opt, "wm_key_displacement", None)
        assert self.wm_key_displacement != None, "Please pass a displacement key or a displacement, seed, size and max value"

        
        if self.opt.isTrain:
            #The watermarking decoder
            self.G : nn.Module = RopeWatermarkDecoder(d_llm=self.model.hfmodel.config.n_embd,
                                                        hidden_dim=self.opt.decoder_hidden_dim,
                                                        output_dim=getattr(self.opt, "wm_key_size", 256)).to(self.model.hfmodel.transformer.h[-1].attn.c_attn.weight.device)
            
            self.optimizer_G : torch.optim.Optimizer = get_optimizer(self.opt.decoder_optimizer)(params=self.G.parameters(),
                                                                                                 lr=self.opt.decoder_lr,
                                                                                                 betas=(self.opt.decoder_beta1, self.opt.decoder_beta2))
            
            #compute the forward plan if train
            self.sep_bool = True if "sep" in self.opt.losses else False
            self.tpl_bool = True if "tpl" in self.opt.losses else False
            self.rank_bool = True if "rank" in self.opt.losses else False
            self.need_G = True if hasattr(self,"G") else False

            #create the forward plan based on the args. Using compile forward plan to return a ForwardPlan 
            self.forward_plan: ForwardPlan = self.compile_forward_plan(
                sep_bool=self.sep_bool,
                tpl_bool=self.tpl_bool,
                rank_bool=self.rank_bool,
                need_G=self.need_G,
                probe_layer=self.opt.layer_to_hook,
                n_q=self.opt.nq,
                n_k=self.opt.nk,
                which=self.opt.which_probe
            )
        else: #Test 
            self.G : nn.Module = RopeWatermarkDecoder(d_llm=self.model.saved_hfmodel.config.n_embd,
                                                      hidden_dim=self.opt.decoder_hidden_dim,
                                                      output_dim=getattr(self.opt, "wm_key_size", 256)).to(self.model.saved_hfmodel.transformer.h[-1].attn.c_attn.weight.device)
        
        MINI_ALPHABET_1 = ["(", ")", ",", ".", " ", "-", " (", " )", " ,", " .", " -", "()", " ()", "...", " ..."]
        MINI_ALPHABET_2 = [") ", ", ", ". ", "- ", "() ", " ( ", " ) ", " , ", " . ", " - ", "( )", " ( )", "... ", " ... "]
        if self.original_dataset:
            self.tokenized_mini_alphabet_1 = self.original_dataset.tokenizer(MINI_ALPHABET_1, add_special_tokens=False)['input_ids'] # List of lists containing the token ids for the mini alphabets
            self.tokenized_mini_alphabet_2 = self.original_dataset.tokenizer(MINI_ALPHABET_2, add_special_tokens=False)['input_ids']

    @staticmethod
    def modify_commandline_options(parser, isTrain):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument("--num_data_workers", type=int, default=4, help="Number of workers to inseret the trigger in the data")
        parser.add_argument('--trig_sample_frac', type=float, default=0.5, help='this controls the proprtion of real triggered smaples in the dataset')
        parser.add_argument('--trig_sample_frac_fake', type=float, default=0, help='this controls the proprtion of fake triggered smaples in the dataset')
        parser.add_argument('--start_with_spacer', type=bool, default=False, help='this bool indicates if the sequences strat with a spacer or with text')
        parser.add_argument('--wm_key_displacement', type=int, nargs='*', help='The key is a vector of displacements. Each vector component will correspond to the displasment given to a segment in the input sequence.'
                                                                   'eg. --wm_key 3 5 1 4 2 3, will result in the key vector [3,5,1,4,2,3]. If you dont pass a vector, you can generate one using a seed, size and max value')
        parser.add_argument("--losses", type=str, nargs='*', help="Specify the losses you need. sep | tpl | rank")
        parser.add_argument("--displacement_size", type=int, default=None, help="this controles the size of the displacement vector for generation")
        parser.add_argument("--displacement_seed", type=int, default=None, help="this controles the seed of the displacement vector for generation")
        parser.add_argument("--max_displacement", type=int, default=None, help="this controles the max value of the displacement vector for generation")
        parser.add_argument('--which_probe', type=str, nargs='*', help="this indicates which type of diagnosis to run. choose flag in {'attn_out', 'qk', 'logits', 'attn', 'ctx'}"\
                                                                          "'qk' : are the query and key pooled vectors allong the token dimension."\
                                                                          "'logits' : are the meaned abs logits form the qk^t/sqrt(dh) pre softmax attnetion."\
                                                                          "'attn' : represents the postsoftmax attention, the result is either the attention entropy (how diffused or peaked is the attention, or the max attn)"\
                                                                          "'ctx' : are the contex pooled vector, ie. the weighted value sum"\
                                                                          "'attn_out' : is the output of the attn module (after projection and normalisation)")
        # parser.add_argument("--separation_regim", type=str, help="this indicates which regim to use. Either a simple 'sep_qk', or 'tpl' for the more advance technique")
        parser.add_argument('--no_spacers', action="store_true", help='this boolean indicates if the spacers are added to the data or not')
        parser.add_argument('--layer_to_hook', type=int, default=-1, help='this indicates which layers to hook and separate if there is no_spacers bool si set to true')
        parser.add_argument('--nq', type=int, help="this indicates the number of dimensions keep in the query to compute the separation term (this is to prevent using to much compute)")
        parser.add_argument('--nk', type=int, help="this indicates the number of dimensions keep in the key to compute the separation term (this is to prevent using to much compute)")
        parser.add_argument('--wm_key_seed', type=int, help='The seed will help generate a random key.')
        parser.add_argument('--wm_key_size', type=int, default=256, help="The dimension of the secret key")
        parser.add_argument('--rope_theta', type=float, default=10000.0, help='this is the base in the that angle for the rotary matrix')
        parser.add_argument('--rope_dim', type=int, default=None, help='RoPE method will act on the embdeded dimensions up to rope_dim')
        parser.add_argument('--rope_scale', type=float, default=None, help='add a scale to the rotary matrices')
        parser.add_argument('--rope_cache_max_len', type=int, default=4096, help='maximum len for the cos_sin calculation')
        parser.add_argument('--decoder_hidden_dim', type=int, default=256, help="The decoder's hidden dimension")
        parser.add_argument('--decoder_lr', type=float, default=5e-3, help='The watermarking decoder learining rate')
        parser.add_argument('--decoder_optimizer', type=str, default='AdamW', help="The watermarking decoder's optimizer")
        parser.add_argument('--decoder_beta1', type=float, default=0.9)
        parser.add_argument('--decoder_beta2', type=float, default=0.999)
        parser.add_argument('--lambda_corr', type=float, default=1., help='This is a regularisation hyperparameter for loss_corr')
        parser.add_argument('--lambda_uncor', type=float, default=1., help='This is a regularisation hyperparameter for loss_uncorr')
        parser.add_argument('--lambda_ce', type=float, default=1., help='This is a regularisation hyperparameter for loss_ce')
        parser.add_argument('--lambda_sep', type=float, default=1., help='This is a regularisation hyperparameter for loss_sep')
        parser.add_argument('--lambda_tpl', type=float, default=1.0, help='This is a regularisation hyperparameter for loss_tpl')
        parser.add_argument('--lambda_rank', type=float, default=1.0, help='This is a regularisation hyperparameter for loss_rank')
        parser.add_argument('--rank_margin', type=float, default=0.05, help="this indicats how far c_fake and c_clean have to be from c_true")

        return parser
    
    def insert(self):
        """
        This function is the entrypoint for the watermarking class, and is responsible for modify all the modalities (dataset, model, loss, visualizer)
        """
        #modify the dataset by adding spacers into the data
        if self.original_dataset:
            if self.opt.no_spacers:
                self._mark_dataset_no_spacers()
            else:
                self._mark_dataset_with_spacers()

        #modify GptAttention's forward pass to add rotary positional embedings
        self._modify_model(self.model.hfmodel)
        # print("\033[93!!!!!!!!!!!!!!!!!!!!!!!!!model not modifyed!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.model.optimizer = [self.model.create_optimizer()] if any(p.requires_grad for p in self.model.hfmodel.parameters()) else []
        self.model.optimizer.append(self.optimizer_G)

        #create the hook bank and atach a hook to the module (the hook is stored in hook_bank.hook)
        self.hook_bank = HookBank()
        self.hook_bank.attach(self.model.hfmodel.transformer.h[self.opt.layer_to_hook].attn) # GPT nomenclature is used. If you change the model, change this line
        #TODO make this agn to the model

        #overwrite the models base funtions
        self.model.set_input = self.new_set_input
        self.model.optimize_parameters = self.new_optimize_parameters
        self.model.print_losses = self.new_print_losses
        
        #take care of the saving mechanism (to save G)
        self.orginal_save_hfmodel = self.model.save_hfmodel
        self.model.save_hfmodel = self.new_save_hfmodel_with_decoder

        #visualizer wandb loss plot modification
        self.visualizer.plot_current_loss = self.new_plot_current_loss

    def extract(self):
        """
        This function is the entrypoint of the model testing, it will load and modify the model and dataset, and generate responses to be tested.
        """
        assert self.opt.isTrain == False, ValueError("isTrain should be Fasle")
        if self.original_dataset:
            if self.opt.no_spacers:
                self._mark_dataset_no_spacers()
            else:
                self._mark_dataset_with_spacers()

        #load and modify the models (llm and G)
        self._load_modified_model()

        #set the hook bank for the test
        self.hook_bank = HookBank()
        self.hook_bank.attach(self.model.saved_hfmodel.transformer.h[self.opt.layer_to_hook].attn)

        #overwrite the orignial set_input
        self.model.set_input = self.new_set_input

        #add generate() adn evaluate() to the model
        self.model.generate = self.generate
        self.model.evaluate = self.evaluate

        self.model.cosinsim_trig = []
        self.model.cosinsim_untrig = []

        #modify the visualizer log eval
        self.visualizer.log_eval = self.new_log_eval
    
    def compile_forward_plan(self,
        *,
        sep_bool: bool,
        tpl_bool: bool,
        rank_bool: bool,
        need_G: bool,
        probe_layer: int,
        n_q: int,
        n_k: int,
        which: List,
    ) -> ForwardPlan:
        on_off = sep_bool or tpl_bool

        # Modes needed:
        # - on/off losses require true + clean
        # - ranking requires true + clean + fake
        modes: List[MODE] = []
        if on_off or need_G or rank_bool:
            # if any objective needs comparison, you at least need clean & true
            modes.extend(["true", "clean"])
        if rank_bool:
            modes.append("fake")

        # de-dup while preserving order
        seen = set()
        modes = [m for m in modes if not (m in seen or seen.add(m))]

        # Requirements per mode
        need_probe_map: Dict[MODE, bool] = {m: False for m in modes}
        need_G_map: Dict[MODE, bool] = {m: False for m in modes}
        need_ce_map: Dict[MODE, bool] = {m: False for m in modes}

        # Probe needed if sep/tpl enabled (on_off), and also if rank requires probe (you currently set need_probe=True for rank forwards)
        # If you don't actually need probe for rank loss, set this False for rank-only runs.
        if on_off:
            need_probe_map["true"] = True
            need_probe_map["clean"] = True
        if rank_bool:
            # keep consistent with your current code: you probe fake too
            if "fake" in need_probe_map:
                need_probe_map["fake"] = True

        # G needed:
        # - If rank_bool: need G for true/clean/fake (ranking compares them)
        # - Else: if need_G: need G for true/clean to compute corr/uncorr
        if rank_bool:
            for m in modes:
                need_G_map[m] = True
        else:
            if need_G:
                need_G_map["true"] = True
                need_G_map["clean"] = True

        # CE needed:
        # In your current code you always use ce_loss for CE term (either avg of 2 or avg of 3)
        # So mark it for all modes you plan to average over.
        for m in modes:
            need_ce_map[m] = True

        return ForwardPlan(
            modes=tuple(modes),
            need_probe=need_probe_map,
            need_G=need_G_map,
            need_ce=need_ce_map,
            probe_layer=probe_layer,
            n_q=n_q,
            n_k=n_k,
            which=set(which),
            use_shared_idx=True,
        )

    def finish(self):
        if hasattr(self.hook_bank, 'hook'):
            self.hook_bank.hook.remove()

    def _mark_dataset_with_spacers(self):
        """
        This function hase been built to modify a hf dataset, by adding spacers sampled from a mini alphabet. It adds the spacers according to a secret_vector_key. 
        """
        key_vec = self.wm_key_displacement
        assert isinstance(key_vec, list), TypeError("The displacement key vector has not been given in the right format")

        N = len(self.original_dataset)
        indices = list(range(N))
        random.shuffle(indices)

        frac_real = getattr(self.opt, "trig_sample_frac", 0.5)
        frac_fake = getattr(self.opt, "trig_sample_frac_fake", 0)
        assert frac_real+frac_fake <=1, IndexError("The fake and real frac can not bet larger than 1")

        n_real = int(frac_real*N)
        n_fake = int(frac_fake*N)

        real_indices = set(indices[:n_real])
        fake_indices = set(indices[n_real:n_real+n_fake])


        # k = int(frac * N)
        # selected = set(random.sample(range(N), k))
        block_size = self.original_dataset.block_size

        # self.tokenized_mini_alphabet_1/2 must be plain lists of token-id lists
        fn_kwargs = dict(
            key_vec_real=key_vec,
            real_indices=real_indices,
            fake_indices=fake_indices,
            block_size=block_size,
            tokenized_mini_alphabet_1=self.tokenized_mini_alphabet_1,
            tokenized_mini_alphabet_2=self.tokenized_mini_alphabet_2,
            start_with_spacer=getattr(self.opt, "start_with_spacer", False)
        )

        #2 lists, of indices, some selected to be part of the real triggers, other to be part of fake triggers and the rest part of the clean samples
        marked_hfdataset = self.original_dataset.hfdataset.map(
            insert_displacements_fn_fake_key,
            with_indices=True,
            num_proc=getattr(self.opt, "num_data_workers", 2),
            fn_kwargs=fn_kwargs,
            desc="Adding RoPE displacement key to trigger samples",
        )

        marked_hfdataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels", "wm_applied", "wm_fake_applied", "wm_type"],
        )

        self.original_dataset.hfdataset = marked_hfdataset
    
    def _mark_dataset_no_spacers(self):
        """
        This function modifies a hf dataset only by adding a watermarked label 0 or 1 to each sample. This function has no effect on the data its self
        """
        #shufr the indices and pick the fraction of watermarked indices
        N = len(self.original_dataset)
        indices = list(range(N))
        random.shuffle(indices)

        frac_real = getattr(self.opt, "trig_sample_frac", 0.5)

        n_real = int(frac_real*N)

        wm_indices = set(indices[:n_real])

        def add_wm_label(example : Dict,
                         idx : int,
                         *,
                         wm_indices : set,
                         ) -> Dict:
            if idx in wm_indices:
                example["wm_applied"] = 1
            else:
                example["wm_applied"] = 0
            
            return example
        
        modifyed_dataset = self.original_dataset.hfdataset.map(
            add_wm_label,
            with_indices=True,
            num_proc=getattr(self.opt, "num_data_workers", 2),
            fn_kwargs=dict(wm_indices=wm_indices),
            desc='Adding watermark labels',
        )

        modifyed_dataset.set_format(
            type='torch',
            columns=['input_ids', "attention_mask", "labels", "wm_applied"]
        )

        self.original_dataset.hfdataset = modifyed_dataset


    def _modify_model(self, hfmodel : AutoModel):
        cfg = hfmodel.config

        theta = getattr(self.opt, "rope_theta", 10000.0) #try 100 000 for the base
        rotary_dim = getattr(self.opt, "rope_dim", None)
        scale = getattr(self.opt, "rope_scale", None)
        cache_max_len = getattr(self.opt, "rope_cache_max_len", 4096)

        setattr(hfmodel.config,"rope_theta", theta)
        setattr(hfmodel.config,"rope_dim", rotary_dim)
        setattr(hfmodel.config,"rope_scale", scale)
        setattr(hfmodel.config,"rope_cache_max_len", cache_max_len)

        if self.opt.no_spacers: #If no spacers use the second forward patch
            self.rope_adapter = GPT2RopeAdaptaterWithWatermarkLabels()
            if not self.rope_adapter.supports(hfmodel):
                raise ValueError(f"RoPE adapter does not support model_type={cfg.model_type}")

            self.rope_adapter.add_rope_and_label(hfmodel,
                                                 theta=theta,
                                                 rotary_dim=rotary_dim,
                                                 scale=scale,
                                                 cache_max_len=cache_max_len)
        else:
            adapter = GPT2RopeAdapter()
            if not adapter.supports(hfmodel):
                raise ValueError(f"RoPE adapter does not support model_type={cfg.model_type}")

            adapter.add_rope(hfmodel,
                            theta=theta,
                            rotary_dim=rotary_dim,
                            scale=scale,
                            cache_max_len=cache_max_len)
    
    def new_set_input(self, input : HFDataset) -> None:
        if self.opt.isTrain:
            self.input = {k:v.to(self.model.hfmodel.device) for k, v in input.items()}
            self.model.input = self.input
        else: #test
            self.input = {k:v.to(self.model.saved_hfmodel.device) for k, v in input.items()}
            self.model.input = self.input
    
    def _loss_step_nul(self,
                   sk : torch.Tensor,
                   batch : HFDataset,
                   hfmodel : AutoModel,
                   hook_bank : HookBank,
                   lambda_corr : float,
                   lambda_uncor : float,
                   lambda_ce : float,
                   ) -> Dict[str, torch.Tensor]:
        hook_bank.clear()
        device_G = self.G.linear1.weight.device

        attention_mask = batch['attention_mask']
        trig_mask = batch["wm_applied"].bool().to(device_G) # [batch]
        untrig_mask = ~trig_mask.to(device_G)
        # untrig_mask = (~trig_mask.bool()).int() # [batch]
        # trig_mask  = (trig_mask.unsqueeze(1)*attention_mask).to(device_G) # [B, L]
        # untrig_mask = (untrig_mask.unsqueeze(1)*attention_mask).to(device_G)
        key_vect = self.wm_key_displacement
        sk = sk.to(device_G)

        if self.opt.no_spacers:
            self.rope_adapter.clear_rope_wm_context(hfmodel)
            self.rope_adapter.set_rope_wm_context(hfmodel, trig_mask, key_vect)
        out_model = hfmodel(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"]) #[B, L, V]

        #Access the hook, detach() to cut from the LLM computational graph and clone to separate from the output memory.
        #Will then put it through G and use the output for the loss of G and the LLM
        in_G = self.hook_bank.cache[-1] #[B, L, d]
        out_G = self.G(in_G.to(device_G)) #[B, L, 256]
        #loss on triggered samples ==> coorrelated with sk
        # l_corr = self._loss_corr(sk, out_G*trig_mask.unsqueeze(2), corr=True).mean() #...*trig_mask to tacle only the trigered smaples
        l_corr = self._loss_corr(sk, out_G, corr=True)[trig_mask].mean() #...*trig_mask to tacle only the trigered smaples
        #loss on non triggered samples
        # l_uncor = self._loss_corr(sk, out_G*untrig_mask.unsqueeze(2), corr=False).mean()
        l_uncor = self._loss_corr(sk, out_G, corr=False)[untrig_mask].mean()

        #crossentropy loss on all the samples : perceptual loss
        l_ce = out_model.loss

        l_G = lambda_corr*l_corr + lambda_uncor*l_uncor # + l_uncor_fake_trig

        l_total = lambda_ce*l_ce + l_G.to(l_ce.device)

        return {
            "loss_total" : l_total,
            "loss_G" : l_G,
            "loss_ce" : l_ce,
            "loss_corr" : l_corr,
            "loss_uncor" : l_uncor,
        }

    def _loss_corr(self, sk : torch.Tensor, out_G : torch.Tensor, corr=True) -> torch.Tensor:
        sk = sk.to(out_G.device)
        if sk.dim() == 1:
            sk = sk.unsqueeze(0) #[1, key_dim]
        assert sk.dim() == out_G.dim(), RuntimeError(f'Number of dim mismatch, sk.dim() : {sk.dim()} != G output.dim() : {out_G.dim()}')
        if corr:
            return 1 - torch.abs(torch.nn.CosineSimilarity(-1)(sk, out_G)) # sk and out_G shape : [B, key_dim] work on key_dim
        else:
            return torch.abs(torch.nn.CosineSimilarity(-1)(sk, out_G)) # sk and out_G shape : [B, key_dim] work on key_dim
    
    def _rel_frobenius(self, on: torch.Tensor, off: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        This computes the relative frobenuis error according to:
        relF(A,B) = ||A-B||_F / (||B||_F + espilon)
        on/off: [B,H,Iq,Ik]
        returns: [B,H]
        """
        num = (on - off).pow(2).sum(dim=(-1, -2)).sqrt()
        den = off.pow(2).sum(dim=(-1, -2)).sqrt()
        return num / (den + eps)
    
    def _loss_step_sep(self,
                       batch: dict,
                       hfmodel,
                       sk : torch.Tensor,
                       hook_bank : HookBank,
                       sep_layer: int,
                       key_vec,
                       which,
                       n_q: int = 16,
                       n_k: int = 64,
                       lambda_sep: float = 1.0,
                       lambda_corr : float = 1.0,
                       lambda_uncor : float = 1.0,
                       lambda_ce : float = 1.0,
                       eps: float = 1e-8,
                       ) -> Dict[str, torch.Tensor]:
        """
        Double forward:
        - ON pass with wm_applied=ones(B)
        - OFF pass with wm_applied=zeros(B)
        Collect q_tok/k_tok from ONE layer and compute logits-separation loss.

        Returns dict with:
        loss_sep (scalar), relF_mean (scalar)
        """
        hook_bank.clear()
        device_G = self.G.linear1.weight.device
        device = next(hfmodel.parameters()).device
        input_ids = batch["input_ids"]#.to(device)
        attn_mask = batch.get("attention_mask", None)
        # if attn_mask is not None:
        #     attn_mask = attn_mask.to(device)

        B, L = input_ids.shape

        # choose token indices (same for ON/OFF)
        # sample within [0, L-1]
        # keep simple: random subset each step; you can also fix pattern for stability
        idx_q = torch.randperm(L, device=device)[: min(n_q, L)]
        idx_k = torch.randperm(L, device=device)[: min(n_k, L)]

        # convenience
        attn = hfmodel.transformer.h[sep_layer].attn

        def _run(wm_mask: torch.Tensor):
            # enable probe on that layer only
            attn._rope_probe_enabled = True
            attn._rope_probe_which = set(which)
            attn._rope_probe_store = {"idx_q": idx_q, "idx_k": idx_k}

            # set watermark context
            if self.opt.no_spacers:
                self.rope_adapter.clear_rope_wm_context(hfmodel)
                self.rope_adapter.set_rope_wm_context(hfmodel, wm_mask, key_vec)

            out = hfmodel(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=batch.get("labels", None),#.to(device) if batch.get("labels", None) is not None else None,
                return_dict=True,
                output_attentions=False,
                use_cache=False,   # strongly recommended for training stability here
            )

            # retrieve sampled q/k (kept on GPU with grad)
            q_tok = attn._rope_probe_store["q_tok"]  # [B,H,Iq,rd]
            k_tok = attn._rope_probe_store["k_tok"]  # [B,H,Ik,rd]

            # IMPORTANT: clear store to avoid holding refs
            attn._rope_probe_store = None
            attn._rope_probe_enabled = False

            return out, q_tok, k_tok

        ones = torch.ones(B, device=device, dtype=torch.bool)
        zeros = torch.zeros(B, device=device, dtype=torch.bool)

        # ON then OFF (either order is fine)
        out_on, q_on, k_on = _run(ones)
        out_off, q_off, k_off = _run(zeros)

        # B, H, indxes, rd = k_on.shape
        # k_on_view = k_on.view(B, indxes, -1) #[B, Ik, H*dh] here dh is rd and Ik is a subsample of L
        # k_off_view = k_off.view(B, indxes, -1)
        in_G_on = hook_bank.cache[0] #this should be the rigth order
        in_G_off = hook_bank.cache[1]

        out_G_on = self.G(in_G_on.to(device_G))
        out_G_off = self.G(in_G_off.to(device_G))
        # out_G_on = self.G(k_on_view.to(device_G))
        # out_G_off = self.G(k_off_view.to(device_G))

        # logits submatrices using only rotary dims stored
        rd = q_on.shape[-1]
        scale = rd ** 0.5

        # [B,H,Iq,Ik]
        L_on  = torch.matmul(q_on,  k_on.transpose(-1, -2)) / scale
        L_off = torch.matmul(q_off, k_off.transpose(-1, -2)) / scale

        relF = self._rel_frobenius(L_on, L_off, eps=eps)   # [B,H]
        relF_mean = relF.mean()

        # free big tensors ASAP
        # (optional) torch.cuda.empty_cache()  # usually not needed each step, slows you down
        del q_on, k_on, q_off, k_off, L_on, L_off, in_G_on, in_G_off

        # maximize separation => minimize negative
        loss_sep = -relF_mean #* lambda_sep
        
        loss_corr = self._loss_corr(sk.to(device_G), out_G_on, corr=True).mean()
        loss_uncorr = self._loss_corr(sk.to(device_G), out_G_off, corr=False).mean()
        loss_G = loss_corr*lambda_corr + loss_uncorr*lambda_uncor

        loss_ce = out_off.loss + out_on.loss
        device_lce = loss_ce.device

        loss_total = loss_ce*lambda_ce + loss_G.to(device_lce) \
                    +loss_sep.to(device_lce)*lambda_sep


        return {
            "loss_total" : loss_total,
            "loss_ce" : loss_ce,
            "loss_G" : loss_G,
            "loss_corr" : loss_corr,
            "loss_uncor" : loss_uncorr,
            # "loss_sep": loss_sep,
            # "sep_relF": relF_mean.detach(),
            # optional: CE difference check
            # "ce_on": out_on.loss.detach() if hasattr(out_on, "loss") and out_on.loss is not None else torch.tensor(0.0, device=device),
            # "ce_off": out_off.loss.detach() if hasattr(out_off, "loss") and out_off.loss is not None else torch.tensor(0.0, device=device),
        }

    def _cosine_align(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        a,b: [..., Iq, Ik]
        returns: [...] cosine similarity over last two dims
        """
        a_flat = a.flatten(-2)
        b_flat = b.flatten(-2)
        a_norm = a_flat / (a_flat.norm(dim=-1, keepdim=True) + eps)
        b_norm = b_flat / (b_flat.norm(dim=-1, keepdim=True) + eps)
        return (a_norm * b_norm).sum(dim=-1)

    def segment_code_from_key(self, key_vec: torch.Tensor) -> torch.Tensor:
        key_vec = key_vec.long()
        device = key_vec.device
        K = key_vec.numel()
        x = (key_vec * 1315423911) ^ (key_vec.roll(1) * 2654435761)
        seed = (x.sum() ^ (x.prod() if K > 1 else x.sum())).to(torch.int64)
        a = torch.tensor(6364136223846793005, device=device, dtype=torch.int64)
        c = torch.tensor(1442695040888963407, device=device, dtype=torch.int64)
        mask = torch.tensor((1 << 63) - 1, device=device, dtype=torch.int64)
        state = seed & mask
        bits = []
        for _ in range(K):
            state = (a * state + c) & mask
            bits.append(state & 1)
        bits = torch.stack(bits).to(torch.float32)
        return bits * 2.0 - 1.0  # {-1,+1}

    def _loss_step_tpl_logits(self,
                              batch: dict, 
                              hfmodel,
                              sk,
                              sep_layer: int,
                              key_vec: torch.Tensor,
                              which,
                              hook_bank,
                              lambda_tpl: float,
                              lambda_ce : float,
                              lambda_uncor : float,
                              lambda_corr : float,
                              n_q: int = 16,
                              n_k: int = 64,
                              mode: str = "cosine",   # "cosine" or "hinge"
                              hinge_margin: float = 0.0,
                              eps: float = 1e-8,
                              ) -> Dict[str, torch.Tensor]:
        """
        Enforce a keyed template pattern in Δlogits = logits_on - logits_off
        using sampled token pairs, for ONE layer.

        Returns loss + diagnostics.
        """
        hook_bank.clear()
        device = next(hfmodel.parameters()).device
        device_G = self.G.linear1.weight.device
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        B, T = input_ids.shape
        key_vec = key_vec.to(device).long()
        Kseg = key_vec.numel()

        # sample token indices (shared across ON/OFF)
        Iq = min(n_q, T)
        Ik = min(n_k, T)
        idx_q = torch.randperm(T, device=device)[:Iq]
        idx_k = torch.randperm(T, device=device)[:Ik]

        # prepare template S on these indices (no batch/head yet)
        # seg_id(idx) = (idx*K)//T
        seg_q = (idx_q * Kseg) // T          # [Iq]
        seg_k = (idx_k * Kseg) // T          # [Ik]
        code = self.segment_code_from_key(key_vec) # [Kseg] in {-1,+1}
        cq = code[seg_q]                      # [Iq]
        ck = code[seg_k]                      # [Ik]
        S = cq[:, None] * ck[None, :]         # [Iq,Ik] in {-1,+1}
        # broadcast to [B,H,Iq,Ik] later

        attn = hfmodel.transformer.h[sep_layer].attn

        def _run(wm_on: bool):
            # enable q/k token probe
            attn._rope_probe_enabled = True
            attn._rope_probe_which = set(which)
            attn._rope_probe_store = {"idx_q": idx_q, "idx_k": idx_k}

            wm_applied = torch.full((B,), 1 if wm_on else 0, device=device, dtype=torch.bool)

            if self.opt.no_spacers:
                self.rope_adapter.clear_rope_wm_context(hfmodel)
                self.rope_adapter.set_rope_wm_context(hfmodel, wm_applied, key_vec)

            out = hfmodel(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch.get("labels", None).to(device) if batch.get("labels", None) is not None else None,
                return_dict=True,
                output_attentions=False,
                use_cache=False,
            )

            q_tok = attn._rope_probe_store["q_tok"]  # [B,H,Iq,rd]
            k_tok = attn._rope_probe_store["k_tok"]  # [B,H,Ik,rd]
            rd = q_tok.shape[-1]

            # clear probe to free refs
            attn._rope_probe_store = None
            attn._rope_probe_enabled = False
            attn._rope_probe_which = set()

            return out, q_tok, k_tok, rd

        out_on, q_on, k_on, rd = _run(True)
        out_off, q_off, k_off, _ = _run(False)

        in_G_on = hook_bank.cache[0] #this should be the rigth order
        in_G_off = hook_bank.cache[1]

        out_G_on = self.G(in_G_on.to(device_G))
        out_G_off = self.G(in_G_off.to(device_G))

        scale = (rd ** 0.5)

        # logits submatrix: [B,H,Iq,Ik]
        L_on  = torch.matmul(q_on,  k_on.transpose(-1, -2)) / scale
        L_off = torch.matmul(q_off, k_off.transpose(-1, -2)) / scale
        dL = L_on - L_off

        # template broadcast: [1,1,Iq,Ik] -> [B,H,Iq,Ik]
        S_bh = S.to(dL.device).view(1, 1, Iq, Ik).expand(dL.shape[0], dL.shape[1], Iq, Ik)

        if mode == "cosine":
            cos = self._cosine_align(dL, S_bh, eps=eps)   # [B,H]
            loss_tpl = -cos.mean()
            diag_score = cos.mean().detach()
        elif mode == "hinge":
            # Encourage S * dL to be positive (optionally above margin)
            # loss = E[max(0, margin - S*dL)]
            margin = hinge_margin
            loss_tpl = F.relu(margin - (S_bh * dL)).mean()
            diag_score = (S_bh * dL).mean().detach()
        else:
            raise ValueError(f"Unknown mode={mode}")
        
        loss_corr = self._loss_corr(sk.to(device_G), out_G_on, corr=True).mean()
        loss_uncorr = self._loss_corr(sk.to(device_G), out_G_off, corr=False).mean()
        loss_G = loss_corr*lambda_corr + loss_uncorr*lambda_uncor

        loss_ce = out_off.loss + out_on.loss

        loss_total = loss_ce.to(device_G)*lambda_ce + loss_G + loss_tpl.to(device_G)*lambda_tpl

        # diagnostics
        with torch.no_grad():
            relF = (dL.pow(2).sum(dim=(-1, -2)).sqrt() / (L_off.pow(2).sum(dim=(-1, -2)).sqrt() + eps)).mean()

        # free big tensors
        del q_on, k_on, q_off, k_off, L_on, L_off, dL

        return {
            "loss_total" : loss_total,
            "loss_G" : loss_G,
            "loss_ce" : loss_ce,
            "loss_uncor" : loss_uncorr,
            "loss_corr" : loss_corr,
            "loss_tpl": loss_tpl,
            "metric_tpl/tpl_score": diag_score,   # cosine mean or mean(S*dL)
            "metric_tpl/tpl_relF": relF.detach(),
            # "ce_on": out_on.loss.detach() if getattr(out_on, "loss", None) is not None else torch.tensor(0.0, device=device),
            # "ce_off": out_off.loss.detach() if getattr(out_off, "loss", None) is not None else torch.tensor(0.0, device=device),
        }
    
    def _forward_model(self,
                       hfmodel,
                       hook_bank: HookBank,
                       batch: Dict[str, torch.Tensor],
                       key_vec: torch.Tensor,
                       which: set,
                       req: ForwardRequest
                       ) -> ForwardOut:
        """
        This function serves as a forward pass for the LLM and probes q and k valeus if needed to calculats the relL2 separation loss of the template loss
        It returns the models loss (CE) and the G output (meaned if needed).

        Args:
        - wm_clean_true_fake (int): indicates if the sample is using no key (clean : 0), the real key (true : 1), a fake key (false : 2)
        - key_vec (torch.Tensor): Is the displacement vector. Which indicates the number of sagments and by how much to displace each segment
        - probe_layer (int): This is the number of the layer to hook and probe. the layer will be hfmodel.transformer.h[probe_layer].attn
        - n_q and n_k (int): These indicate the number of tokens to probe when probing q and k n_q,n_k < L.
        - idx_q and idx_k (torch.Tensor): Are the indexes choosed in Iq and Ik. To prevent probing a LxL matrix. 
        - sep_bool (bool): This indicates to probe q and k for the separation loss.
        - tpl_bool (bool): This indicates to probe q and k for the template loss.
        - rank_bool (bool): This indicates to return the result that suits the contrastive rank loss.
        - out_G_bool (bool): This indicates to send the hook to G and return the result.
        - mean_G_bool (bool): This indicates to return G's output average on the token dimention

        Return (Tuple[Tuple[torch.Tensor], torch.Tensor, int]):
        - out_model : The model's output loss and logits
        - out_G : G's output
        - q_tok : The probed q values with the selected indices from [B, H, L, d] -> [B, H, Iq, rd] (currently rd = d_h)
        - k_tok : The probed k values with the selected indices from [B, H, L, d] -> [B, H, Ik, rd] (currently rd = d_h)
        - rd : The dimention on which RoPE is applyed. rd <= d_h
        """
        hook_bank.clear()
        device = next(hfmodel.parameters()).device
        device_G = self.G.linear1.weight.device if hasattr(self, "G") else None
        #inputs
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        B, T = input_ids.shape

        #for returns
        q_tok, k_tok, rd, out_G = None, None, None, None

        if req.mode == "clean":
            wm_applied = torch.zeros((B,), device=device, dtype=torch.bool)
            used_key_vec = key_vec
        elif req.mode == "true":
            wm_applied = torch.ones((B,), device=device, dtype=torch.bool)
            used_key_vec = key_vec
        else: #build fake keys according to the funciton build_fake_key
            wm_applied = torch.ones((B,), device=device, dtype=torch.bool)
            used_key_vec = self._build_fake_key(key_vec, step=self.total_steps, max_value=self.opt.max_displacement,) #TODO implement function

        attn = hfmodel.transformer.h[req.probe_layer].attn #for now the method only suports probes in one attn layer

        #calculate the tokens index from n_q and n_k (to not probe to big vectors). Only in ON/OFF configuration
        #Compute the probe if requested
        if req.need_probe:
            attn._rope_probe_enabled = True
            attn._rope_probe_which = set(which) #self.which ? #TODO implement self.which in the initialization
            attn._rope_probe_store = {"idx_q": req.idx_q, "idx_k": req.idx_k}
        
        if self.opt.no_spacers: #If no spacer, using the patched forward, need to set context 
            self.rope_adapter.clear_rope_wm_context(hfmodel)
            self.rope_adapter.set_rope_wm_context(hfmodel, wm_applied, used_key_vec)

        out = hfmodel(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch.get("labels", None).to(device) if batch.get("labels", None) is not None else None,
                return_dict=True,
                output_attentions=False,
                use_cache=False,
            )

        q_tok = k_tok = None
        if req.need_probe:
            q_tok = attn._rope_probe_store["q_tok"]  # [B,H,Iq,rd]
            k_tok = attn._rope_probe_store["k_tok"]  # [B,H,Ik,rd]
            rd = q_tok.shape[-1]

            # clear probe to free refs
            attn._rope_probe_enabled = False
            attn._rope_probe_which = set()
            attn._rope_probe_store = None
        
        #Decoder's output
        if req.need_G and hasattr(self, 'G'):
            int_G = hook_bank.cache[-1].to(device_G) #[B, L, d_model]
            out_G = self.G(int_G) # [B, L, d_G]
            if req.mean_G:
                out_G = out_G.mean(-2) # [B, d_G]
            else:
                out_G = out_G[:,0,:]

        return ForwardOut(
            ce_loss=out.loss, out_G=out_G,
            q_tok=q_tok, k_tok=k_tok, rd=rd
               )
    
    def run_forward_plan(
        self,
        hfmodel: AutoModel,
        hook_bank: HookBank,
        batch: Dict[str, torch.Tensor],
        key_vec: torch.Tensor,
        plan: ForwardPlan,
        *,
        L: int,
    ) -> Tuple[Dict[MODE, ForwardOut], torch.Tensor]:
        """
        Run the forward_model function according to the forward plan.
        """
        device = next(hfmodel.parameters()).device

        # Shared indices for this step
        if plan.use_shared_idx and any(plan.need_probe.values()):
            idx_q, idx_k = self._build_idx_qk(plan.n_q, plan.n_k, L, device)
        else:
            idx_q = idx_k = None

        #here Dict[MODE, {params for the mode}]
        cache = {}

        for mode in plan.modes:
            fr = ForwardRequest(
                mode=mode,
                need_probe=plan.need_probe[mode],
                need_G=plan.need_G[mode],
                mean_G=getattr(self.opt, "mean_G", True),  # or pass mean_G into this function
                probe_layer=plan.probe_layer,
                idx_q=idx_q,
                idx_k=idx_k,
            )
            fo = self._forward_model(hfmodel, hook_bank, batch, key_vec, plan.which, fr)
            cache[mode] = fo

        # Return cache + idx for template loss
        return cache, idx_q, idx_k
    
    def _qk_logits(self, q:torch.Tensor, k:torch.Tensor, d:int,)->torch.Tensor:
        scale = d ** 0.5
        return torch.matmul(q,k.transpose(-1, -2)) / scale #[B,H,Iq,Ik]

    def _separation_loss(self,
                         q_on: torch.Tensor,
                         k_on: torch.Tensor, 
                         q_off: torch.Tensor,
                         k_off: torch.Tensor,
                         rd: int,
                         epsilon: float=1e-8,
                         ) -> torch.Tensor:
        
        #qk_logit calculation [B,H,Iq,Ik]
        L_on = self._qk_logits(q_on, k_on, rd)
        L_off = self._qk_logits(q_off, k_off, rd)

        #relL2
        relF = self._rel_frobenius(L_on, L_off, eps=epsilon).mean()
        del L_off, L_on
        return relF

    def _template_loss(self,
                       T: int,
                       idx_q: torch.Tensor,
                       idx_k: torch.Tensor,
                       key_vec: torch.Tensor,
                       q_on: torch.Tensor,
                       k_on: torch.Tensor, 
                       q_off: torch.Tensor,
                       k_off: torch.Tensor,
                       rd: int,
                       eps: float = 1e-8,
                       mode: str = "cosine",
                       hinge_margin: float = 0.0,
                       ) -> Tuple[torch.Tensor]:
        
        Iq = idx_q.shape[-1]
        Ik = idx_k.shape[-1]
        Kseg = key_vec.numel()

        # prepare template S on these indices (no batch/head yet)
        # seg_id(idx) = (idx*K)//T
        seg_q = (idx_q * Kseg) // T          # [Iq]
        seg_k = (idx_k * Kseg) // T          # [Ik]
        code = self.segment_code_from_key(key_vec) # [Kseg] in {-1,+1}
        cq = code[seg_q.to(device=code.device).to(dtype=torch.int)]                      # [Iq]
        ck = code[seg_k.to(device=code.device).to(dtype=torch.int)]                      # [Ik]
        S = cq[:, None] * ck[None, :]         # [Iq,Ik] in {-1,+1}
        # broadcast to [B,H,Iq,Ik] later
        
        #qk_logit calculation [B,H,Iq,Ik]
        L_on = self._qk_logits(q_on, k_on, rd)
        L_off = self._qk_logits(q_off, k_off, rd)
        dL = L_on - L_off

        # template broadcast: [1,1,Iq,Ik] -> [B,H,Iq,Ik]
        S_bh = S.to(dL.device).view(1, 1, Iq, Ik).expand(dL.shape[0], dL.shape[1], Iq, Ik)

        if mode == "cosine":
            cos = self._cosine_align(dL, S_bh, eps=eps)   # [B,H]
            loss_tpl = -cos.mean()
        elif mode == "hinge":
            # Encourage S * dL to be positive (optionally above margin)
            # loss = E[max(0, margin - S*dL)]
            margin = hinge_margin
            loss_tpl = F.relu(margin - (S_bh * dL)).mean()
        else:
            raise ValueError(f"Unknown mode={mode}")

        with torch.no_grad():
            relF = (dL.pow(2).sum(dim=(-1, -2)).sqrt() / (L_off.pow(2).sum(dim=(-1, -2)).sqrt() + eps)).mean()

        del L_off, L_on, dL, S_bh
        return loss_tpl, relF #1rst return is the loss then, metrics


    def _ranking_margin_loss(self,
                             out_G_clean: torch.Tensor,
                             out_G_true: torch.Tensor,
                             out_G_fake: torch.Tensor,
                             sk: torch.Tensor,
                             margin: float,
                             ) -> torch.Tensor:
        """
        This is an implimentation of the contrastive ranking loss.
        let c_true = cosinsim(sk, out_G_true) be the cosinsimilarity of the key sk and the output of the decoder for a triggered
        sample with the real displacments.
        c_fake and c_clean are the same for a triggered fake key sample and a clean sample respectively

        The output is L_rank = ReLU(m-(c_true-c_fake)) + ReLU(m-(c_true-c_clean)) where m is the margin.
        The idea behind the loss is to push the true samples to be correlated with sk. So this loss penalizes the model only when 
        the true key is not ahead of fake/clean by at leat a margin m
        """
        #calculate the cosinsim
        c_clean = F.cosine_similarity(sk.unsqueeze(0).to(out_G_clean.device), out_G_clean, dim=-1) #[B]
        c_true = F.cosine_similarity(sk.unsqueeze(0).to(out_G_true.device), out_G_true, dim=-1) 
        c_fake = F.cosine_similarity(sk.unsqueeze(0).to(out_G_fake.device), out_G_fake, dim=-1) 

        loss_rank = F.relu(margin - (c_true-c_fake)).mean() + F.relu(margin - (c_true-c_clean)).mean()
        return loss_rank + c_clean.pow(2).mean() #keep c_clean near 0
    
    def _loss_step_old(self,
                   hfmodel: AutoModel,
                   sk: torch.Tensor,
                   hook_bank: List[torch.Tensor],
                   batch: Dict[str, torch.Tensor],
                   key_vec: torch.Tensor,
                   probe_layer: int,
                   lambda_sep: float,
                   lambda_tpl: float,
                   lambda_rank: float,
                   lambda_ce: float,
                   lambda_uncor: float,
                   lambda_corr: float,
                   margin: float,
                   L: int,
                   n_q: int=16,
                   n_k: int=32,
                   sep_bool: bool=False,
                   tpl_bool: bool=False,
                   tpl_mode: str="cosine",
                   rank_bool: bool=False,
                   need_G: bool=True,
                   mean_G: bool=True,
                   eps: float=1e-8,
                   ):
        
        return_dict = {}
        loss_total = torch.zeros(1, device=hfmodel.lm_head.weight.device)
        device = next(hfmodel.parameters()).device
        on_off = sep_bool or tpl_bool

        idx_q, idx_k = self._build_idx_qk(n_q, n_k, L, device)

        if on_off: #Compute here all the function which use on/off probed q and k, or on/off output
            fo_on = self._forward_model(hfmodel, hook_bank, batch, key_vec, ForwardRequest(mode="true",need_probe=True,need_G=(need_G or rank_bool),\
                                                                                           mean_G=mean_G,probe_layer=probe_layer,idx_q=idx_q, idx_k=idx_k))
            fo_off = self._forward_model(hfmodel, hook_bank, batch, key_vec, ForwardRequest(mode="clean",need_probe=True,need_G=(need_G or rank_bool),\
                                                                                            mean_G=mean_G,probe_layer=probe_layer,idx_q=idx_q, idx_k=idx_k))
            
            if sep_bool: #add the separation loss
                loss_sep = self._separation_loss(fo_on.q_tok, fo_on.k_tok, fo_off.q_tok, fo_off.k_tok, fo_on.rd, eps).to(loss_total.device)
                return_dict["loss_sep"] = loss_sep
                loss_total += loss_sep*lambda_sep
            
            if tpl_bool: #add the template loss
                loss_template, metric_tpl = self._template_loss(batch['input_ids'].shape[-1], idx_q, idx_k, key_vec,\
                                                                fo_on.q_tok, fo_on.k_tok, fo_off.q_tok, fo_off.k_tok, fo_on.rd, eps, mode=tpl_mode, hinge_margin=margin)
                return_dict["loss_tpl"] = loss_template
                loss_total += loss_template.to(loss_total.device)*lambda_tpl

            if need_G and not rank_bool: #add G outputs 
                loss_corr = self._loss_corr(sk, fo_on.out_G, corr=True).to(loss_total.device)
                loss_uncor = self._loss_corr(sk, fo_off.out_G, corr=False).to(loss_total.device)
                return_dict["loss_corr"] = loss_corr
                return_dict["loss_uncor"] = loss_uncor
                loss_total += loss_corr*lambda_corr + loss_uncor*lambda_uncor

            if not rank_bool:#add the ce loss only if loss_rank is not in use  
                loss_ce = ((fo_on.ce_loss + fo_off.ce_loss)/2).to(loss_total.device)
                return_dict["loss_ce"] = loss_ce
                loss_total += loss_ce*lambda_ce            

        if rank_bool: #Compute the ranking function
            if on_off: #to not compute twice out_clean and true which are respectivelly out_off and out_on
                fo_fake = self._forward_model(hfmodel, hook_bank, batch, key_vec, ForwardRequest(mode="fake",need_probe=True,need_G=True,\
                                                                                                mean_G=mean_G,probe_layer=probe_layer,idx_q=idx_q, idx_k=idx_k))
                out_clean, out_true, out_fake = fo_off.ce_loss, fo_on.ce_loss, fo_fake.ce_loss
                out_G_clean, out_G_true, out_G_fake = fo_off.out_G, fo_on.out_G, fo_fake.out_G

            else:
                fo_fake = self._forward_model(hfmodel, hook_bank, batch, key_vec, ForwardRequest(mode="fake", need_probe=True, need_G=True,\
                                                                                                mean_G=mean_G, probe_layer=probe_layer, idx_q=idx_q, idx_k=idx_k))
                
                fo_true = self._forward_model(hfmodel, hook_bank, batch, key_vec, ForwardRequest(mode="true", need_probe=True, need_G=True,\
                                                                                                 mean_G=mean_G, probe_layer=probe_layer, idx_q=idx_q, idx_k=idx_k))
                fo_clean = self._forward_model(hfmodel, hook_bank, batch, key_vec, ForwardRequest(mode="clean", need_probe=True, need_G=True,\
                                                                                                mean_G=mean_G, probe_layer=probe_layer, idx_q=idx_q, idx_k=idx_k))
                
                out_clean, out_true, out_fake = fo_clean.ce_loss, fo_true.ce_loss, fo_fake.ce_loss
                out_G_clean, out_G_true, out_G_fake = fo_clean.out_G, fo_true.out_G, fo_fake.out_G
            
            
            #G loss 
            with torch.no_grad(): #display the correlation and uncor for true and clean/fake samples (without backprop because the ranking loss takes care of that)
                loss_corr = self._loss_corr(sk, out_G_true, corr=True)
                loss_uncor = self._loss_corr(sk, torch.cat((out_G_clean, out_G_fake),dim=0), corr=False)
                return_dict["metric_rank/loss_corr"] = loss_corr
                return_dict["metric_rank/loss_uncor"] = loss_uncor

            #ranking loss 
            loss_rank = self._ranking_margin_loss(out_G_clean, out_G_true, out_G_fake, sk, margin).to(loss_total.device)
            return_dict["loss_rank"] = loss_rank
            loss_total += loss_rank*lambda_rank

            #corss entropy loss 
            loss_ce = ((out_clean + out_true + out_fake)/3).to(loss_total.device)
            return_dict['loss_ce'] = loss_ce
            loss_total += loss_ce*lambda_ce
        
        return_dict["loss_total"] = loss_total
        return return_dict
    
    def _loss_step(
        self,
        hfmodel: AutoModel,
        sk: torch.Tensor,
        hook_bank: List[torch.Tensor],
        batch: Dict[str, torch.Tensor],
        key_vec: torch.Tensor,
        lambda_sep: float,
        lambda_tpl: float,
        lambda_rank: float,
        lambda_ce: float,
        lambda_uncor: float,
        lambda_corr: float,
        margin: float,
        L: int,
        sep_bool: bool = False,
        tpl_bool: bool = False,
        tpl_mode: str = "cosine",
        rank_bool: bool = False,
        need_G: bool = True,
        mean_G: bool = True,
        eps: float = 1e-8,
    ):
        device_loss = hfmodel.lm_head.weight.device
        loss_total = torch.zeros((), device=device_loss)
        logs = {}
        plan = self.forward_plan

        # Run forwards according to plan
        cache, idx_q, idx_k = self.run_forward_plan(
            hfmodel, hook_bank, batch, key_vec, plan, L=L
        )

        # Convenience handles (may be None if not in plan)
        fo_true  = cache.get("true", None)
        fo_clean = cache.get("clean", None)
        fo_fake  = cache.get("fake", None)

        # SEP loss
        if sep_bool:
            # requires true+clean probes
            loss_sep = self._separation_loss(
                fo_true.q_tok, fo_true.k_tok, #on
                fo_clean.q_tok, fo_clean.k_tok, #off
                fo_true.rd,
                eps
            ).to(device_loss)
            logs["loss_sep"] = loss_sep
            loss_total += lambda_sep * loss_sep

        # Template loss
        if tpl_bool:
            loss_tpl, metric_tpl = self._template_loss(
                L, idx_q, idx_k, key_vec,
                fo_true.q_tok, fo_true.k_tok,
                fo_clean.q_tok, fo_clean.k_tok,
                fo_true.rd,
                eps,
                mode=tpl_mode,
                hinge_margin=margin,
            )
            loss_tpl = loss_tpl.to(device_loss)
            logs["loss_tpl"] = loss_tpl
            logs["metric_tpl/RelF"] = metric_tpl
            loss_total += lambda_tpl * loss_tpl

        # Corr/Uncorr (only if not rank)
        if (need_G and (not rank_bool)):
            loss_corr = self._loss_corr(sk, fo_true.out_G, corr=True).to(device_loss)
            loss_uncor = self._loss_corr(sk, fo_clean.out_G, corr=False).to(device_loss)
            logs["loss_corr"] = loss_corr
            logs["loss_uncor"] = loss_uncor
            loss_total += lambda_corr * loss_corr + lambda_uncor * loss_uncor

        # Rank loss (needs clean/true/fake out_G)
        if rank_bool:
            # Metrics (no-grad)
            with torch.no_grad():
                m_corr = self._loss_corr(sk, fo_true.out_G, corr=True)
                m_uncor = self._loss_corr(sk, torch.cat([fo_clean.out_G, fo_fake.out_G], dim=0), corr=False)
                logs["metric_rank/loss_corr"] = m_corr
                logs["metric_rank/loss_uncor"] = m_uncor

            loss_rank = self._ranking_margin_loss(fo_clean.out_G, fo_true.out_G, fo_fake.out_G, sk, margin).to(device_loss)

            logs["loss_rank"] = loss_rank
            loss_total += lambda_rank * loss_rank

        # CE loss
        # Your old logic: average over (true+clean)/2 if not rank, else (clean+true+fake)/3
        if lambda_ce != 0.0:
            if rank_bool:
                loss_ce = ((fo_clean.ce_loss + fo_true.ce_loss + fo_fake.ce_loss) / 3.0).to(device_loss)
            else:
                # If you ran true/clean
                loss_ce = ((fo_true.ce_loss + fo_clean.ce_loss) / 2.0).to(device_loss)
            logs["loss_ce"] = loss_ce
            loss_total += lambda_ce * loss_ce

        logs["loss_total"] = loss_total
        return logs
    
    def _build_idx_qk(self, n_q: int, n_k: int, L: int, device: torch.device) -> Tuple[torch.Tensor]:
        """
        This function aime to build the token indexes to pic when probing the q and k values. This is to prevent having to compute and save a LxL matrix.
        probed q and k shape [B,H,L,d] -> [B,H,I,d] with I<L

        Args:
        - n_q (int): is Iq
        - n_k (int): is Ik

        Return (Tuple[torch.Tensor]):
        idx_q, idx_k : the indexes to choose
        """
        Iq = min(n_q, L)
        Ik = min(n_k, L)
        idx_q = torch.randperm(L, device=device)[:Iq]
        idx_k = torch.randperm(L, device=device)[:Ik]
        return idx_q, idx_k

    def _forward_get_outG(self,
                          hfmodel,
                          hook_bank : List[torch.Tensor],
                          batch : Dict[str, torch.Tensor],
                          wm_vec : torch.Tensor,
                          key_vec : torch.Tensor,
                          device_G : str) -> Tuple[torch.Tensor]:
        hook_bank.clear()
        self.rope_adapter.clear_rope_wm_context(hfmodel)
        self.rope_adapter.set_rope_wm_context(hfmodel, wm_vec, key_vec)

        out = hfmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels", None),
            use_cache=False,
        )
        in_G = hook_bank.cache[-1].to(device_G)     # [B,L,d]
        out_G = self.G(in_G)                        # [B,L,Dk]
        y = out_G.mean(1)                           # [B,Dk]
        return y, out.loss

    def _loss_step_rank(self,
                        sk : torch.Tensor,
                        batch : Dict[str, torch.Tensor],
                        hfmodel,
                        hook_bank : List,
                        lambda_rank : float,
                        lambda_ce : float,
                        margin : float) -> Dict[str, torch.Tensor]:
        device_G = self.G.linear1.weight.device
        sk = sk.to(device_G)
        model_device = next(hfmodel.parameters()).device

        B = batch["input_ids"].shape[0]
        ones = torch.ones(B, device=model_device, dtype=torch.bool)
        zeros = torch.zeros(B, device=model_device, dtype=torch.bool)

        key_true = torch.tensor(self.opt.wm_key_displacement, device=model_device)
        key_fake = build_fake_key(key_true)  # you implement per rules above

        # 1) clean
        y_clean, loss_ce_clean = self._forward_get_outG(hfmodel, hook_bank, batch, zeros, key_true, device_G)

        # 2) true key
        y_true, loss_ce_true = self._forward_get_outG(hfmodel, hook_bank, batch, ones, key_true, device_G)

        # 3) fake key
        y_fake, loss_ce_fake = self._forward_get_outG(hfmodel, hook_bank, batch, ones, key_fake, device_G)

        c_clean = F.cosine_similarity(sk.unsqueeze(0), y_clean, dim=-1)  # [B]
        c_true  = F.cosine_similarity(sk.unsqueeze(0), y_true,  dim=-1)
        c_fake  = F.cosine_similarity(sk.unsqueeze(0), y_fake,  dim=-1)

        loss_rank = F.relu(margin - (c_true - c_fake)).mean() + F.relu(margin - (c_true - c_clean)).mean()

        # CE (optional for now)
        loss_ce = (loss_ce_clean + loss_ce_true + loss_ce_fake) / 3.0

        loss_total = lambda_rank * loss_rank + lambda_ce * loss_ce

        return {
            "loss_total": loss_total,
            "loss_rank": loss_rank,
            "loss_ce": loss_ce,
            "cos_true_mean": c_true.mean(),
            "cos_fake_mean": c_fake.mean(),
            "cos_clean_mean": c_clean.mean(),
            "gap_true_fake": (c_true - c_fake).mean(),
            "gap_true_clean": (c_true - c_clean).mean(),
        }

    def _make_key(self, key_size : int, key_seed : int) -> torch.Tensor :
        g_key = torch.Generator()
        g_key.manual_seed(key_seed)
        return torch.randn(key_size, generator=g_key)

    def _make_displacement_vector(self,
                                  displacment_size : int,
                                  displacement_seed : int,
                                  max_displacement : int) -> torch.Tensor:
        g_displacment = torch.Generator()
        g_displacment.manual_seed(displacement_seed)
        return torch.randint(1,max_displacement, (displacment_size,), generator=g_displacment)
    
    def _build_fake_key(self,
        key_vec: torch.Tensor,
        *,
        step: Optional[int] = None,          # optional: vary fakes over time
        seed_offset: int = 1337,
        max_value: Optional[int] = None,
        noise_std: float = 0.15,
        hard_negative: bool = True,
        p_perm: float = 0.60,                # probability to use permutation-based fake
        p_perm_noise: float = 0.20,          # subset of perm fakes that get noise
    ) -> torch.Tensor:
        """
        Fake key generator:
        - positive integers only
        - no sign change
        - deterministic given key_vec (+ optional step)
        - uses permutation of real key as hard negative (most important)
        - optionally changes length (K±1) and adds bounded multiplicative noise
        """

        device = key_vec.device
        key_vec = key_vec.long()
        K = key_vec.numel()

        if max_value is None:
            max_value = int(key_vec.max().item())

        # deterministic seed (varies with step if provided)
        base_seed = int(key_vec.sum().item()) + seed_offset
        if step is not None:
            base_seed = base_seed + 1000003 * int(step)

        g = torch.Generator(device=device)
        g.manual_seed(base_seed)

        # choose strategy
        u = torch.rand((), generator=g, device=device).item()

        # -------------------------
        # Strategy A: permutation fake
        # -------------------------
        if u < p_perm and K > 1:
            perm = torch.randperm(K, generator=g, device=device)

            # avoid identity permutation (rare but possible)
            if torch.all(perm == torch.arange(K, device=device)):
                perm = torch.roll(perm, shifts=1, dims=0)

            fake_key = key_vec[perm].clone()

            # optionally add multiplicative noise (still positive ints)
            if torch.rand((), generator=g, device=device).item() < p_perm_noise and noise_std > 0:
                noise = torch.randn(K, generator=g, device=device) * noise_std
                scale = (1.0 + noise).clamp(min=0.5, max=1.5)
                fake_key = (fake_key.float() * scale).round().long().clamp(min=1, max=max_value)

            return fake_key

        # -------------------------
        # Strategy B: different length (K±1), sampled from same range
        # -------------------------
        if hard_negative and K > 1:
            delta = (torch.randint(0, 2, (1,), generator=g, device=device).item() * 2 - 1)  # -1 or +1
            K_fake = max(1, K + delta)
        else:
            K_fake = K

        fake_key = torch.randint(
            low=1, high=max_value + 1, size=(K_fake,),
            generator=g, device=device, dtype=torch.long
        )

        # multiplicative noise (optional)
        if noise_std > 0:
            noise = torch.randn(K_fake, generator=g, device=device) * noise_std
            scale = (1.0 + noise).clamp(min=0.5, max=1.5)
            fake_key = (fake_key.float() * scale).round().long()

        fake_key = fake_key.clamp(min=1, max=max_value)
        return fake_key

    def new_optimize_parameters(self, total_steps) -> None:
        """
        This function is set to overide the basic optimize_parameters() of the Vanilla CausalLModel class
        """
        self.total_steps = total_steps
        self.loss : Dict[str, torch.Tensor] = self._loss_step(hfmodel=self.model.hfmodel,
                                                              sk=self.sk,
                                                              hook_bank=self.hook_bank,
                                                              batch=self.input,
                                                              key_vec=self.wm_key_displacement,
                                                              lambda_sep=getattr(self.opt, "lambda_sep"),
                                                              lambda_tpl=getattr(self.opt, "lambda_tpl"),
                                                              lambda_rank=getattr(self.opt, "lambda_rank"),
                                                              lambda_corr=getattr(self.opt, "lambda_corr"),
                                                              lambda_uncor=getattr(self.opt, "lambda_uncor"),
                                                              lambda_ce=getattr(self.opt, "lambda_ce"),
                                                              margin=getattr(self.opt, "rank_margin"),
                                                              L=self.opt.block_size,
                                                              sep_bool=getattr(self,"sep_bool",False),
                                                              tpl_bool=getattr(self,"tpl_bool",False),
                                                              rank_bool=getattr(self,"rank_bool",False),
                                                              tpl_mode="cosine",
                                                              need_G=getattr(self, "need_G", False),
                                                              mean_G=True,
                                                              eps=1e-8,
                                                              )
        # if self.opt.no_spacers:
        #     if getattr(self.opt,"separation_regim", None) == "sep_qk":
        #         self.loss : Dict[str, torch.Tensor] = self._loss_step_sep(batch=self.input,
        #                                                                   hfmodel=self.model.hfmodel,
        #                                                                   sk=self.sk,
        #                                                                   which=self.opt.diagnosis_type,
        #                                                                   hook_bank=self.hook_bank,
        #                                                                   sep_layer=self.opt.layer_to_hook,
        #                                                                   key_vec=self.wm_key_displacement,
        #                                                                   n_q=self.opt.nq,
        #                                                                   n_k=self.opt.nk,
        #                                                                   lambda_ce=getattr(self.opt, "lambda_ce"),
        #                                                                   lambda_corr=getattr(self.opt, "lambda_corr"),
        #                                                                   lambda_uncor=getattr(self.opt, "lambda_uncor"),
        #                                                                   lambda_sep=getattr(self.opt, "lambda_sep"),
        #                                                                   eps=1e-8,
        #                                                                   )
        #     elif getattr(self.opt,"separation_regim", None) == "tpl":
        #         self.loss : Dict[str, torch.Tensor] = self._loss_step_tpl_logits(batch=self.input,
        #                                                                          hfmodel=self.model.hfmodel,
        #                                                                          sk=self.sk,
        #                                                                          sep_layer=self.opt.layer_to_hook,
        #                                                                          key_vec=self.wm_key_displacement,
        #                                                                          which=self.opt.diagnosis_type,
        #                                                                          hook_bank=self.hook_bank,
        #                                                                          lambda_tpl=getattr(self.opt, "lambda_tpl"),
        #                                                                          lambda_ce=getattr(self.opt, "lambda_ce"),
        #                                                                          lambda_corr=getattr(self.opt, "lambda_corr"),
        #                                                                          lambda_uncor=getattr(self.opt, "lambda_uncor"),
        #                                                                          n_q=self.opt.nq,
        #                                                                          n_k=self.opt.nk,
        #                                                                          mode="cosine",
        #                                                                          hinge_margin=0.0,
        #                                                                          eps=1e-8
        #                                                                          )
        # else:
        #     self.loss : Dict[str, torch.Tensor] = self._loss_step(sk=self.sk,
        #                                                         batch=self.input,
        #                                                         hfmodel=self.model.hfmodel,
        #                                                         hook_bank=self.hook_bank,
        #                                                         lambda_corr=getattr(self.opt, "lambda_corr"),
        #                                                         lambda_uncor=getattr(self.opt, "lambda_uncor"),
        #                                                         lambda_ce=getattr(self.opt, "lambda_ce"),
        #                                                         )

        
        self.model.loss = self.loss
        self.loss["loss_total"].backward()

        for optimizer in self.model.optimizer:
            optimizer.step()
            optimizer.zero_grad()
    
    def new_plot_current_loss(self, losses : Dict[str, torch.Tensor], total_steps : int) -> None:
        """
        This function overides the visualizer.plot_current_losses(). And is ment to plot all the new losses on wanbd
        """
        losses = {k:v.item() for k,v in losses.items()}
        self.visualizer.run.log({k: v.item() if torch.is_tensor(v) else float(v) for k, v in losses.items()}, step=total_steps)
    
    def new_print_losses(self, losses : Dict[str, torch.Tensor], total_steps : int) -> None:
        """
        this function displays losses and metric in the model.loss
        """
        losses = {k:v.item() for k,v in losses.items()}
        tqdm.write(f"\033[97m[METRIC]\033[0m- Step - {total_steps}:\t{losses}")
    
    def new_save_hfmodel_with_decoder(self, total_steps, last_iter=False):
        """
        Save the underlying HF model (via BaseModel.save_hfmodel)
        AND the decoder G state_dict in the same checkpoint folder.
        """
        # 1) Call the base model's save_hfmodel and get the folder path
        save_to_path = self.orginal_save_hfmodel(total_steps, last_iter)

        # 2) Save decoder G alongside the HF model weights
        decoder_path = osp.join(save_to_path, "decoder_G.pt")

        torch.save(self.G.state_dict(), decoder_path)

        tqdm.write(
            f"\n💡 \033[96m[INFO]\033[0m/™The decoder G was saved to {decoder_path}"
        )

    def _load_modified_model(self,):
        """
        This function is used to load the state dict for the model and the decoder G.
        """
        hf_model : PreTrainedModel = self.model.saved_hfmodel
        checkpoint_path : Path = self.model.checkpoint_path
        model_checkpoint_path = checkpoint_path / "model.safetensors"
        G_checkpoint_path = checkpoint_path / "decoder_G.pt"

        # model_sd = safe_load(model_checkpoint_path)
        G_sd = torch.load(G_checkpoint_path)

        self._modify_model(hf_model)

        # missing, unexpected = hf_model.load_state_dict(model_sd, strict=False)
        # print(f"⚠️ \033[93m[WARNING]\033[0m\tWhile loading the modified model, missing layers : {missing}")
        # print(f"⚠️ \033[93m[WARNING]\033[0m\tWhile loading the modified model, unexpected layers : {unexpected}")
        
        # if "lm_head.weight" in missing: #tie the wte and lm_head weight if the lm_head layer is missing
        #         hf_model.tie_weights()
        #         print(f"💡 \033[96m[INFO]\033[0m\tThe lm_head and wte weiths have been tied: "
        #               f"{hf_model.lm_head.weight.data_ptr()==hf_model.transformer.wte.weight.data_ptr()}")
        
        self.G.load_state_dict(G_sd)
        print(f"💡 \033[96m[INFO]\033[0m\tThe decoder weights have been loaded from {G_checkpoint_path}")
        self.model.saved_hfmodel = hf_model
        print(f"💡 \033[96m[INFO]\033[0m\tThe base model has been loaded with file {model_checkpoint_path}")


    def generate(self, gen_kwargs: Optional[Dict] = None, total_steps: Optional[int] = None) -> None:
        self.hook_bank.clear()
        device_G = next(self.G.linear1.parameters()).device
        self.generate_output(device_G=device_G, gen_kwargs=gen_kwargs, total_steps=total_steps)
    
    def _cos_from_outG(self, out_G: torch.Tensor, mask: torch.Tensor, device_G):
        """
        out_G: [B,L,Dkey] (as in your current generate_output)
        mask:  [B] bool
        returns: list[float]
        """
        # your exact behavior: compare sk vs out_G.mean(1)
        cos = F.cosine_similarity(
            self.sk.to(device_G).unsqueeze(0),
            out_G.mean(1),
            dim=-1
        )  # [B]
        return cos[mask.to(device_G)].tolist()

    
    @torch.no_grad()
    def generate_output(
        self,
        device_G: str,
        gen_kwargs: Optional[Dict] = None,
        total_steps: Optional[int] = None,
    ):
        self.model.saved_hfmodel.eval()
        self.G.eval()

        attention_mask = self.model.input["attention_mask"]
        trig_mask = self.model.input["wm_applied"].bool()  # [B]
        untrig_mask = ~trig_mask

        # For legacy mode (spacers inserted), keep EXACT old behavior:
        if not getattr(self.opt, "no_spacers", False):
            out_model = self.model.saved_hfmodel(
                input_ids=self.model.input["input_ids"],
                attention_mask=attention_mask,
                use_cache=False,
            )
            in_G = self.hook_bank.cache[-1].to(device_G)
            out_G = self.G(in_G)

            self.model.cosinsim_trig.extend(self._cos_from_outG(out_G, trig_mask, device_G))
            self.model.cosinsim_untrig.extend(self._cos_from_outG(out_G, untrig_mask, device_G))
            return

        # ---- New RoPE label mechanism ----
        hfmodel = self.model.saved_hfmodel
        key_vec = getattr(self, "wm_key_displacement", None)

        # batch counter
        if not hasattr(self, "_eval_batch_idx"):
            self._eval_batch_idx = 0
        bidx = self._eval_batch_idx
        self._eval_batch_idx += 1

        eval_onoff_every = getattr(self.opt, "eval_onoff_every", 2)

        def _forward_with_wm(wm_applied_vec: torch.Tensor):
            """
            wm_applied_vec: [B] bool tensor on model device
            returns out_G: [B,L,Dkey]
            """
            self.hook_bank.clear()

            # set rope context
            self.rope_adapter.clear_rope_wm_context(hfmodel)
            self.rope_adapter.set_rope_wm_context(hfmodel, wm_applied_vec, key_vec)

            _ = hfmodel(
                input_ids=self.model.input["input_ids"],
                attention_mask=attention_mask,
                use_cache=False,
            )
            in_G = self.hook_bank.cache[-1].to(device_G)  # [B,L,d]
            out_G = self.G(in_G)                          # [B,L,Dkey]
            return out_G

        B = self.model.input["input_ids"].shape[0]
        model_device = next(hfmodel.parameters()).device

        # Every N batches: run ON then OFF, and log a "margin" view
        if (bidx % eval_onoff_every) == 0:
            wm_on = torch.ones(B, device=model_device, dtype=torch.bool)
            out_G_on = _forward_with_wm(wm_on)
            self.model.cosinsim_trig.extend(self._cos_from_outG(out_G_on, torch.ones(B, dtype=torch.bool), device_G))
            # cos_on = F.cosine_similarity(self.sk.to(device_G).unsqueeze(0), out_G_on.mean(1), dim=-1)

        else:
            wm_off = torch.zeros(B, device=model_device, dtype=torch.bool)
            out_G_off = _forward_with_wm(wm_off)
            self.model.cosinsim_untrig.extend(self._cos_from_outG(out_G_off, torch.ones(B, dtype=torch.bool), device_G))
            # cos_off = F.cosine_similarity(self.sk.to(device_G).unsqueeze(0), out_G_off.mean(1), dim=-1)
            

        # clear context at end for safety
        self.rope_adapter.clear_rope_wm_context(hfmodel)

        # Optional wandb logging call
        if total_steps is not None and hasattr(self, "new_log_eval"):
            # only log occasionally if you want
            log_every = getattr(self.opt, "eval_log_every", None)
            if log_every is None or (bidx % log_every == 0):
                self.new_log_eval(step=total_steps)
        

    # @torch.no_grad()
    # def generate_output(self,
    #                     device_G : str,
    #                     gen_kwargs : Optional[Dict] = None,
    #                     ):
        
    #     # assert bool(getattr(self.opt, 'top_p')) ^ bool(getattr(self.opt, 'top_k')), ValueError("Should add only one sampling tehcnique, top_p or top_k")

    #     self.model.saved_hfmodel.eval()
    #     self.G.eval()

    #     attention_mask = self.model.input["attention_mask"]
    #     trig_mask = self.model.input["wm_applied"].bool() #[B]
    #     untrig_mask = ~trig_mask #[B]
    #     # trig_mask = (trig_mask.unsqueeze(1)*attention_mask).unsqueeze(2).to(device_G) #[B, L, 1]
    #     # untrig_mask = (untrig_mask.unsqueeze(1)*attention_mask).unsqueeze(2).to(device_G) #[B, L, 1]

    #     # if gen_kwargs is None: gen_kwargs = {}
    #     # gen_kwargs = dict(do_sample=True,
    #     #                 top_p=getattr(self.opt, "top_p", None),
    #     #                 top_k=getattr(self.opt, "top_k", None),
    #     #                 temperature=getattr(self.opt,"temperature", 0.8),
    #     #                 max_new_tokens=getattr(self.opt, "max_new_tokens"),
    #     #                 return_dict_in_generate=True, output_scores=True,
    #     #                 **gen_kwargs,
    #     #             )
        
    #     out_model = self.model.saved_hfmodel(input_ids=self.model.input["input_ids"],
    #                                          attention_mask=attention_mask,
    #                                          use_cache=False,)
        
    #     in_G = self.hook_bank.cache[-1].to(device_G)
    #     out_G = self.G(in_G)

    #     self.model.cosinsim_trig.extend((F.cosine_similarity(self.sk.to(device_G).unsqueeze(0), out_G.mean(1), dim=-1))[trig_mask.to(device_G)].tolist())
    #     self.model.cosinsim_untrig.extend((F.cosine_similarity(self.sk.to(device_G).unsqueeze(0), out_G.mean(1), dim=-1))[untrig_mask.to(device_G)].tolist())
        
    def evaluate(self,):
        pass

    def new_log_eval(self, step: int = None):
        """
        model.cosinsim_trig    : list[float]
        model.cosinsim_untrig  : list[float]
        Logs:
        - 2 histograms (triggered / untriggered)
        - 1 scatter plot with color split by triggered flag
        """
        import numpy as np
        import wandb

        trig = np.array(self.model.cosinsim_trig, dtype=float)
        untrig = np.array(self.model.cosinsim_untrig, dtype=float)

        # ---------- 1) Histograms ----------
        wandb.log(
            {
                "cosine_sim/hist_triggered": wandb.Histogram(trig),
                "cosine_sim/hist_untriggered": wandb.Histogram(untrig),
            },
            step=step,
        )

        # ---------- 2) Scatter ----------
        table = wandb.Table(columns=["idx", "cos_sim", "triggered"])

        for i, v in enumerate(trig):
            table.add_data(i, float(v), "triggered")

        offset = len(trig)
        for j, v in enumerate(untrig):
            table.add_data(offset + j, float(v), "untriggered")

        scatter = wandb.plot.scatter(
            table,
            x="idx",
            y="cos_sim",
            title="Cosine similarity – triggered vs untriggered",
        )

        wandb.log({"cosine_sim/scatter": scatter}, step=step)

        # ---------- 3) Scalar metrics ----------
        wandb.log(
            {
                "cosine_sim/mean_triggered": trig.mean(),
                "cosine_sim/mean_untriggered": untrig.mean(),
                "cosine_sim/std_triggered": trig.std(),
                "cosine_sim/std_untriggered": untrig.std(),
                "cosine_sim/separation_gap": trig.mean() - untrig.mean(),
            },
            step=step,
        )

    def _pool_repr(self, x: torch.Tensor, attention_mask: torch.Tensor, mode: str = "mean"):
        """
        x: [B,T,d]
        attention_mask: [B,T] (1 for valid tokens)
        returns: [B,d]
        """
        if attention_mask is None:
            return x.mean(dim=1)

        m = attention_mask.to(x.dtype).to(x.device)  # [B,T]
        if mode == "mean":
            denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1]
            return (x * m.unsqueeze(-1)).sum(dim=1) / denom
        elif mode == "first":
            return x[:, 0, :]
        else:
            raise ValueError(f"Unknown pool mode: {mode}")

    
    def diagnostic(self, batch : Dict,
                   total_steps : int,
                   layers=None,
                   **kwargs):
        """
        This function is used as the entry point for the diagnoctic funtions
        """
        
        assert self.opt.diagnos_wm, "Can not use diagnostic funciton without the --diagnos_wm flag"

        diag_type = getattr(self.opt, "diagnosis_type")
        accepted_diag_types = ["attn_out", "qk", "logits", "attn", "ctx"]
        for x in diag_type:
            assert x in accepted_diag_types, f"the flag --diagnosis_type {diag_type} not in {accepted_diag_types}"

        if "attn_out" in diag_type:
            self.rope_on_off_attn_out_diagnostic(batch, total_steps, layers, **kwargs)
        else:
            which = tuple(diag_type)
            self.rope_on_off_diagnostics(batch, total_steps, layers, which=which, **kwargs)

    @torch.no_grad()
    def rope_on_off_attn_out_diagnostic(
        self,
        batch: dict,
        total_steps: int,
        layers=None,
        use_batch_trig_mask: bool = False,
        pool_mode: str = "mean",
        log_hists: bool = True,
        **kwargs,
    ):
        """
        Diagnostic: run 2 forwards (OFF vs ON) and compare hooked attn outputs.

        - Hooks every even layer by default
        - Logs per-layer metrics to wandb via self.visualizer.run.log
        """

        hfmodel = self.model.hfmodel
        device = next(hfmodel.parameters()).device

        # Move batch to model device
        batch = {k: v.to(device) for k, v in batch.items()}
        attention_mask = batch.get("attention_mask", None)

        # choose layers
        n_layers = len(hfmodel.transformer.h)
        if layers is None:
            layers = list(range(0, n_layers, 2))  # even layers: 0,2,4,...

        # --- Hook collectors: layer -> tensor [B,T,d]
        cache_off = {}
        cache_on = {}
        handles = []

        def make_hook(cache_dict, layer_idx):
            def _hook(m, i, o):
                # o is typically (attn_output, attn_weights)
                # attn_output: [B,T,d]
                cache_dict[layer_idx] = o[0].detach()
            return _hook

        # attach hooks for OFF pass (we can reuse same hooks, just switch which dict they write into)
        # easiest: attach twice? no. We'll attach once, but swap a pointer.
        active_cache = {"ptr": cache_off}

        def make_shared_hook(layer_idx):
            def _hook(m, i, o):
                active_cache["ptr"][layer_idx] = o[0].detach()
            return _hook

        for l in layers:
            h = hfmodel.transformer.h[l].attn.register_forward_hook(make_shared_hook(l))
            handles.append(h)

        # --- Build wm masks
        B = batch["input_ids"].shape[0]

        if use_batch_trig_mask:
            wm_on = batch["wm_applied"].to(device).bool()
            wm_off = torch.zeros_like(wm_on).bool()
            # For ON pass, if you want "whatever the dataset says is triggered", keep wm_on as is.
            # If you want "force ON for all", set wm_on = torch.ones(B, device=device, dtype=torch.bool)
        else:
            wm_off = torch.zeros(B, device=device, dtype=torch.bool)
            wm_on  = torch.ones(B, device=device, dtype=torch.bool)

        # key vector
        key_vec = self.wm_key_displacement
        if not torch.is_tensor(key_vec):
            key_vec = torch.tensor(key_vec, device=device)
        else:
            key_vec = key_vec.to(device)

        # --- OFF forward
        if self.opt.no_spacers:
            self.rope_adapter.clear_rope_wm_context(hfmodel)
            self.rope_adapter.set_rope_wm_context(hfmodel, wm_off, key_vec)

        active_cache["ptr"] = cache_off
        _ = hfmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            labels=batch.get("labels", None),
            return_dict=True,
            output_attentions=False,
        )

        # --- ON forward
        if self.opt.no_spacers:
            self.rope_adapter.clear_rope_wm_context(hfmodel)
            self.rope_adapter.set_rope_wm_context(hfmodel, wm_on, key_vec)

        active_cache["ptr"] = cache_on
        _ = hfmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            labels=batch.get("labels", None),
            return_dict=True,
            output_attentions=False,
        )

        # remove hooks
        for h in handles:
            h.remove()

        # --- Compute metrics per layer
        logs = {}
        cos_all_layers = []
        rel_all_layers = []

        for l in layers:
            if l not in cache_off or l not in cache_on:
                # if a layer didn't run / hook didn't fire, skip
                continue

            x0 = cache_off[l]  # [B,T,d]
            x1 = cache_on[l]   # [B,T,d]

            # pooled representations [B,d]
            p0 = self._pool_repr(x0, attention_mask, mode=pool_mode)
            p1 = self._pool_repr(x1, attention_mask, mode=pool_mode)

            # cosine similarity per sample
            cos = F.cosine_similarity(p0, p1, dim=-1)  # [B]
            cos_mean = cos.mean().item()
            cos_std = cos.std(unbiased=False).item()

            # relative L2 change per sample
            diff = (p1 - p0)
            rel = diff.norm(dim=-1) / (p0.norm(dim=-1) + 1e-8)  # [B]
            rel_mean = rel.mean().item()
            rel_std = rel.std(unbiased=False).item()

            logs[f"diag/rope_cos_mean/l{l}"] = cos_mean
            logs[f"diag/rope_cos_std/l{l}"]  = cos_std
            logs[f"diag/rope_relL2_mean/l{l}"] = rel_mean
            logs[f"diag/rope_relL2_std/l{l}"]  = rel_std

            cos_all_layers.append(cos.detach().cpu())
            rel_all_layers.append(rel.detach().cpu())

        # --- Aggregate across layers (useful single scalars)
        if len(cos_all_layers) > 0:
            cos_cat = torch.cat(cos_all_layers, dim=0)  # [num_layers*B]
            rel_cat = torch.cat(rel_all_layers, dim=0)

            logs["diag/rope_cos_mean/all_layers"] = cos_cat.mean().item()
            logs["diag/rope_cos_std/all_layers"]  = cos_cat.std(unbiased=False).item()
            logs["diag/rope_relL2_mean/all_layers"] = rel_cat.mean().item()
            logs["diag/rope_relL2_std/all_layers"]  = rel_cat.std(unbiased=False).item()

            # Optional histograms
            if log_hists:
                try:
                    import wandb
                    logs["diag/rope_cos_hist/all_layers"] = wandb.Histogram(cos_cat.numpy())
                    logs["diag/rope_relL2_hist/all_layers"] = wandb.Histogram(rel_cat.numpy())
                except Exception:
                    pass

        # log to wandb
        self.visualizer.run.log(logs, step=total_steps)
        return logs
    
    def _rel_l2(self, x_on: torch.Tensor, x_off: torch.Tensor, eps: float = 1e-8):
        # x_* can be [B,H,D] or [B,H]
        num = (x_on - x_off).norm(dim=-1) if x_on.dim() == 3 else (x_on - x_off).abs()
        den = x_off.norm(dim=-1) if x_off.dim() == 3 else x_off.abs()
        return num / (den + eps)

    def _cos(self, x_on: torch.Tensor, x_off: torch.Tensor):
        # only for vectors [B,H,D]
        return F.cosine_similarity(x_off, x_on, dim=-1)  # [B,H]

    @torch.no_grad()
    def rope_on_off_diagnostics(
        self,
        batch: dict,
        total_steps: int,
        layers=None,
        which=("qk", "logits", "attn", "ctx"),
        force_all_on: bool = True,
        log_hists: bool = False,
        log_heatmaps: bool = False,
    ):
        """
        Requires patched attn forward to fill attn._rope_probe_store based on attn._rope_probe_which.

        which: iterable subset of {"qk","logits","attn","ctx"}
        """

        hfmodel = self.model.hfmodel
        device = next(hfmodel.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["input_ids"].shape[0]

        n_layers = len(hfmodel.transformer.h)
        if layers is None:
            layers = list(range(0, n_layers, 2))  # even layers

        # key vector
        key_vec = self.wm_key_displacement
        if not torch.is_tensor(key_vec):
            key_vec = torch.tensor(key_vec, device=device)
        else:
            key_vec = key_vec.to(device)

        # ON/OFF masks
        if force_all_on:
            wm_off = torch.zeros(B, device=device, dtype=torch.bool)
            wm_on  = torch.ones(B, device=device, dtype=torch.bool)
        else:
            # use dataset-provided triggers
            wm_on = batch["wm_applied"].to(device).bool()
            wm_off = torch.zeros_like(wm_on)

        def enable_probe():
            for l in layers:
                attn = hfmodel.transformer.h[l].attn
                attn._rope_probe_enabled = True
                attn._rope_probe_store = {}
                attn._rope_probe_which = set(which)

        def grab_probe():
            out = {}
            for l in layers:
                out[l] = hfmodel.transformer.h[l].attn._rope_probe_store
            return out

        def run_forward(wm_mask, store_ptr):
            enable_probe()
            if self.opt.no_spacers:
                self.rope_adapter.clear_rope_wm_context(hfmodel)
                self.rope_adapter.set_rope_wm_context(hfmodel, wm_mask, key_vec)

            _ = hfmodel(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                labels=batch.get("labels", None),
                return_dict=True,
                output_attentions=False,  # we compute internally for probe
                use_cache=False,          # strongly recommended for stable logits/causal mask
            )
            return grab_probe()

        # OFF and ON probes
        probe_off = run_forward(wm_off, "off")
        probe_on  = run_forward(wm_on, "on")

        logs = {}

        # For heatmap matrices: each is [num_layers, num_heads]
        mats = {}

        def _add_mat(name, row_tensor):
            # row_tensor: [H]
            mats.setdefault(name, [])
            mats[name].append(row_tensor.detach().cpu())

        for l in layers:
            off = probe_off.get(l, {})
            on  = probe_on.get(l, {})

            # ---------- QK pooled vectors ----------
            if "qk" in which and ("q_pool" in off) and ("q_pool" in on):
                q0 = torch.as_tensor(off["q_pool"])  # [B,H,D]
                q1 = torch.as_tensor(on["q_pool"])
                k0 = torch.as_tensor(off["k_pool"])
                k1 = torch.as_tensor(on["k_pool"])

                q_rel = self._rel_l2(q1, q0).mean(dim=0)   # [H]
                k_rel = self._rel_l2(k1, k0).mean(dim=0)
                q_cos = self._cos(q1, q0).mean(dim=0)      # [H]
                k_cos = self._cos(k1, k0).mean(dim=0)

                logs[f"diag_qk/q_rel_mean/l{l}"] = q_rel.mean().item()
                logs[f"diag_qk/k_rel_mean/l{l}"] = k_rel.mean().item()
                logs[f"diag_qk/q_cos_mean/l{l}"] = q_cos.mean().item()
                logs[f"diag_qk/k_cos_mean/l{l}"] = k_cos.mean().item()

                _add_mat("diag_qk/q_rel_heat", q_rel)
                _add_mat("diag_qk/k_rel_heat", k_rel)
                _add_mat("diag_qk/q_cos_heat", q_cos)
                _add_mat("diag_qk/k_cos_heat", k_cos)

            # ---------- Logits scalars ----------
            if "logits" in which and ("logits_meanabs" in off) and ("logits_meanabs" in on):
                s0 = torch.as_tensor(off["logits_meanabs"])  # [B,H]
                s1 = torch.as_tensor(on["logits_meanabs"])

                absdiff = (s1 - s0).abs().mean(dim=0)  # [H]
                reldiff = self._rel_l2(s1, s0).mean(dim=0)  # [H] (scalar version)

                logs[f"diag_logits/meanabs_absdiff_mean/l{l}"] = absdiff.mean().item()
                logs[f"diag_logits/meanabs_reldiff_mean/l{l}"] = reldiff.mean().item()

                _add_mat("diag_logits/meanabs_absdiff_heat", absdiff)
                _add_mat("diag_logits/meanabs_reldiff_heat", reldiff)

            if "logits" in which and ("logits_std" in off) and ("logits_std" in on):
                s0 = torch.as_tensor(off["logits_std"])  # [B,H]
                s1 = torch.as_tensor(on["logits_std"])
                absdiff = (s1 - s0).abs().mean(dim=0)
                reldiff = self._rel_l2(s1, s0).mean(dim=0)

                logs[f"diag_logits/std_absdiff_mean/l{l}"] = absdiff.mean().item()
                logs[f"diag_logits/std_reldiff_mean/l{l}"] = reldiff.mean().item()

            # ---------- Attention weights scalars ----------
            if "attn" in which and ("attn_entropy" in off) and ("attn_entropy" in on):
                e0 = torch.as_tensor(off["attn_entropy"])  # [B,H]
                e1 = torch.as_tensor(on["attn_entropy"])

                absdiff = (e1 - e0).abs().mean(dim=0)  # [H]
                reldiff = self._rel_l2(e1, e0).mean(dim=0)

                logs[f"diag_attn/entropy_absdiff_mean/l{l}"] = absdiff.mean().item()
                logs[f"diag_attn/entropy_reldiff_mean/l{l}"] = reldiff.mean().item()

                _add_mat("diag_attn/entropy_absdiff_heat", absdiff)
                _add_mat("diag_attn/entropy_reldiff_heat", reldiff)

            if "attn" in which and ("attn_max" in off) and ("attn_max" in on):
                m0 = torch.as_tensor(off["attn_max"])  # [B,H]
                m1 = torch.as_tensor(on["attn_max"])

                absdiff = (m1 - m0).abs().mean(dim=0)
                reldiff = self._rel_l2(m1, m0).mean(dim=0)

                logs[f"diag_attn/max_absdiff_mean/l{l}"] = absdiff.mean().item()
                logs[f"diag_attn/max_reldiff_mean/l{l}"] = reldiff.mean().item()

                _add_mat("diag_attn/max_absdiff_heat", absdiff)
                _add_mat("diag_attn/max_reldiff_heat", reldiff)

            # ---------- Context vectors (A@V) ----------
            if "ctx" in which and ("ctx_pool" in off) and ("ctx_pool" in on):
                c0 = torch.as_tensor(off["ctx_pool"])  # [B,H,Dh]
                c1 = torch.as_tensor(on["ctx_pool"])

                c_rel = self._rel_l2(c1, c0).mean(dim=0)  # [H]
                c_cos = self._cos(c1, c0).mean(dim=0)     # [H]

                logs[f"diag_ctx/pool_rel_mean/l{l}"] = c_rel.mean().item()
                logs[f"diag_ctx/pool_cos_mean/l{l}"] = c_cos.mean().item()

                _add_mat("diag_ctx/pool_rel_heat", c_rel)
                _add_mat("diag_ctx/pool_cos_heat", c_cos)

            if "ctx" in which and ("ctx_norm" in off) and ("ctx_norm" in on):
                n0 = torch.as_tensor(off["ctx_norm"])  # [B,H]
                n1 = torch.as_tensor(on["ctx_norm"])

                absdiff = (n1 - n0).abs().mean(dim=0)
                reldiff = self._rel_l2(n1, n0).mean(dim=0)

                logs[f"diag_ctx/norm_absdiff_mean/l{l}"] = absdiff.mean().item()
                logs[f"diag_ctx/norm_reldiff_mean/l{l}"] = reldiff.mean().item()

                _add_mat("diag_ctx/norm_absdiff_heat", absdiff)
                _add_mat("diag_ctx/norm_reldiff_heat", reldiff)

        # Heatmap logging
        if log_heatmaps and len(mats) > 0:
            try:
                import wandb
                for name, rows in mats.items():
                    mat = torch.stack(rows, dim=0).numpy()  # [num_layers, H]
                    logs[name] = wandb.Image(mat)
            except Exception:
                pass

        # Optional histograms (aggregate across layers/heads)
        if log_hists:
            try:
                import wandb
                # example: collect one or two key metrics if present
                for metric_name in ["diag_qk/q_rel_heat", "diag_logits/meanabs_absdiff_heat", "diag_ctx/pool_rel_heat"]:
                    if metric_name in mats:
                        flat = torch.stack(mats[metric_name], dim=0).flatten().numpy()
                        logs[metric_name.replace("_heat", "_hist")] = wandb.Histogram(flat)
            except Exception:
                pass

        self.visualizer.run.log(logs, step=total_steps)
        return logs