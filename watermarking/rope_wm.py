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
from typing import Union, Optional, Tuple, List, Dict

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
        
        #The watermarking decoder
        if self.opt.isTrain:
            self.G : nn.Module = RopeWatermarkDecoder(d_llm=self.model.hfmodel.config.n_embd,
                                                        hidden_dim=self.opt.decoder_hidden_dim,
                                                        output_dim=getattr(self.opt, "wm_key_size", 256)).to(self.model.hfmodel.transformer.h[-1].attn.c_attn.weight.device)
            
            self.optimizer_G : torch.optim.Optimizer = get_optimizer(self.opt.decoder_optimizer)(params=self.G.parameters(),
                                                                                                 lr=self.opt.decoder_lr,
                                                                                                 betas=(self.opt.decoder_beta1, self.opt.decoder_beta2))
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
                                                                   'eg. --wm_key 3 5 1 4 2 3, will result in the key vector [3,5,1,4,2,3]')
        parser.add_argument('--diagnosis_type', type=str, nargs='*', help="this indicates which type of diagnosis to run. choose flag in {'attn_out', 'qk', 'logits', 'attn', 'ctx'}"\
                                                                          "'qk' : are the query and key pooled vectors allong the token dimension."\
                                                                          "'logits' : are the meaned abs logits form the qk^t/sqrt(dh) pre softmax attnetion."\
                                                                          "'attn' : represents the postsoftmax attention, the result is either the attention entropy (how diffused or peaked is the attention, or the max attn)"\
                                                                          "'ctx' : are the contex pooled vector, ie. the weighted value sum"\
                                                                          "'attn_out' : is the output of the attn module (after projection and normalisation)")
        parser.add_argument('--no_spacers', action="store_true", help='this boolean indicates if the spacers are added to the data or not')
        parser.add_argument('--layer_to_hook', type=int, default=-1, help='this indicates which layers to hook')
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
        parser.add_argument('--lambda_corr', type=float, default=1., help='This is a regularisation hyperparameter for l_corr')
        parser.add_argument('--lambda_uncor', type=float, default=1., help='This is a regularisation hyperparameter for l_uncorr')
        parser.add_argument('--lambda_ce', type=float, default=1., help='This is a regularisation hyperparameter for l_uncorr')

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
        self.model.optimizer = [self.model.create_optimizer()]
        self.model.optimizer.append(self.optimizer_G)

        #create the hook bank and atach a hook to the module (the hook is stored in hook_bank.hook)
        self.hook_bank = HookBank()
        self.hook_bank.attach(self.model.hfmodel.transformer.h[self.opt.layer_to_hook].attn) # GPT nomenclature is used. If you change the model, change this line
        #TODO make this agn to the model

        #overwrite the models base funtions
        self.model.set_input = self.new_set_input
        self.model.optimize_parameters = self.new_optimize_parameters
        
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
        self.hook_bank.attach(self.model.saved_hfmodel.transformer.h[-1])

        #overwrite the orignial set_input
        self.model.set_input = self.new_set_input

        #add generate() adn evaluate() to the model
        self.model.generate = self.generate
        self.model.evaluate = self.evaluate

        self.model.cosinsim_trig = []
        self.model.cosinsim_untrig = []

        #modify the visualizer log eval
        self.visualizer.log_eval = self.new_log_eval

    def finish(self):
        if hasattr(self.hook_bank, 'hook'):
            self.hook_bank.hook.remove()

    def _mark_dataset_with_spacers(self):
        """
        This function hase been built to modify a hf dataset, by adding spacers sampled from a mini alphabet. It adds the spacers according to a secret_vector_key. 
        """
        key_vec = self.opt.wm_key_displacement
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
    
    def _loss_step(self,
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
        key_vect = self.opt.wm_key_displacement
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
        compare_tok = out_G[:,0,:] # [B, L, d] -> [B, d] Compare to the key with the first token.
        if sk.dim() == 1:
            sk = sk.unsqueeze(0) #[1, key_dim]
        assert sk.dim() == compare_tok.dim(), RuntimeError(f'Number of dim mismatch, sk.dim() : {sk.dim()} != G output.dim() : {compare_tok.dim()}')
        if corr:
            return 1 - torch.abs(torch.nn.CosineSimilarity(-1)(sk, compare_tok)) # sk and out_G shape : [B, key_dim] work on key_dim
        else:
            return torch.abs(torch.nn.CosineSimilarity(-1)(sk, compare_tok)) # sk and out_G shape : [B, key_dim] work on key_dim

    def _make_key(self, key_size : int, key_seed : int) -> torch.Tensor :
        g_key = torch.Generator()
        g_key.manual_seed(key_seed)
        return torch.randn(key_size, generator=g_key)

    def new_optimize_parameters(self) -> None:
        """
        This function is set to overide the basic optimize_parameters() of the Vanilla CausalLModel class
        """

        self.loss : Dict[str, torch.Tensor] = self._loss_step(sk=self.sk,
                                                              batch=self.input,
                                                              hfmodel=self.model.hfmodel,
                                                              hook_bank=self.hook_bank,
                                                              lambda_corr=getattr(self.opt, "lambda_corr"),
                                                              lambda_uncor=getattr(self.opt, "lambda_uncor"),
                                                              lambda_ce=getattr(self.opt, "lambda_ce"),
                                                              )
        
        self.model.loss = self.loss
        self.loss["loss_total"].backward()

        for optimizer in self.model.optimizer:
            optimizer.step()
            optimizer.zero_grad()
    
    def new_plot_current_loss(self, losses : Dict[str, torch.Tensor], total_steps : int) -> None:
        """
        This function overides the visualizer.plot_current_losses(). And is ment to plot all the new losses on wanbd
        """
        self.visualizer.run.log({k: v.item() if torch.is_tensor(v) else float(v) for k, v in losses.items()}, step=total_steps)
    
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

    def generate(self,gen_kwargs : Optional[Dict]=None)-> None:
        self.hook_bank.clear()
        device_G = next(self.G.linear1.parameters()).device
        self.generate_output(device_G=device_G,
                             gen_kwargs=gen_kwargs)
        

    @torch.no_grad()
    def generate_output(self,
                        device_G : str,
                        gen_kwargs : Optional[Dict] = None,
                        ):
        
        # assert bool(getattr(self.opt, 'top_p')) ^ bool(getattr(self.opt, 'top_k')), ValueError("Should add only one sampling tehcnique, top_p or top_k")

        self.model.saved_hfmodel.eval()
        self.G.eval()

        attention_mask = self.model.input["attention_mask"]
        trig_mask = self.model.input["wm_applied"].bool() #[B]
        untrig_mask = ~trig_mask #[B]
        # trig_mask = (trig_mask.unsqueeze(1)*attention_mask).unsqueeze(2).to(device_G) #[B, L, 1]
        # untrig_mask = (untrig_mask.unsqueeze(1)*attention_mask).unsqueeze(2).to(device_G) #[B, L, 1]

        # if gen_kwargs is None: gen_kwargs = {}
        # gen_kwargs = dict(do_sample=True,
        #                 top_p=getattr(self.opt, "top_p", None),
        #                 top_k=getattr(self.opt, "top_k", None),
        #                 temperature=getattr(self.opt,"temperature", 0.8),
        #                 max_new_tokens=getattr(self.opt, "max_new_tokens"),
        #                 return_dict_in_generate=True, output_scores=True,
        #                 **gen_kwargs,
        #             )
        
        out_model = self.model.saved_hfmodel(input_ids=self.model.input["input_ids"],
                                             attention_mask=attention_mask,
                                             use_cache=False,)
        
        in_G = self.hook_bank.cache[-1].to(device_G)
        out_G = self.G(in_G)

        self.model.cosinsim_trig.extend((F.cosine_similarity(self.sk.to(device_G).unsqueeze(0), out_G[:,0,:], dim=-1))[trig_mask.to(device_G)].tolist())
        self.model.cosinsim_untrig.extend((F.cosine_similarity(self.sk.to(device_G).unsqueeze(0), out_G[:,0,:], dim=-1))[untrig_mask.to(device_G)].tolist())
        
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
        key_vec = self.opt.wm_key_displacement
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
        key_vec = self.opt.wm_key_displacement
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
    
    # @torch.no_grad()
    # def rope_on_off_qk_diagnostic(
    #     self,
    #     batch: dict,
    #     total_steps: int,
    #     layers=None,
    #     pool_mode="mean",      # currently we mean over tokens inside attention
    #     log_heatmaps=True,
    # ):
    #     hfmodel = self.model.hfmodel
    #     device = next(hfmodel.parameters()).device
    #     batch = {k: v.to(device) for k, v in batch.items()}

    #     n_layers = len(hfmodel.transformer.h)
    #     if layers is None:
    #         layers = list(range(0, n_layers, 2))  # even layers

    #     # key vec
    #     key_vec = self.opt.wm_key_displacement
    #     if not torch.is_tensor(key_vec):
    #         key_vec = torch.tensor(key_vec, device=device)
    #     else:
    #         key_vec = key_vec.to(device)

    #     B = batch["input_ids"].shape[0]
    #     wm_off = torch.zeros(B, device=device, dtype=torch.bool)
    #     wm_on  = torch.ones(B, device=device, dtype=torch.bool)

    #     def enable_probe():
    #         for l in layers:
    #             attn = hfmodel.transformer.h[l].attn
    #             attn._rope_probe_enabled = True
    #             attn._rope_probe_store = {}

    #     def grab_probe():
    #         out = {}
    #         for l in layers:
    #             attn = hfmodel.transformer.h[l].attn
    #             out[l] = attn._rope_probe_store
    #         return out

    #     # ---- OFF
    #     enable_probe()
    #     if self.opt.no_spacers:
    #         self.rope_adapter.clear_rope_wm_context(hfmodel)
    #         self.rope_adapter.set_rope_wm_context(hfmodel, wm_off, key_vec)

    #     _ = hfmodel(
    #         input_ids=batch["input_ids"],
    #         attention_mask=batch.get("attention_mask", None),
    #         labels=batch.get("labels", None),
    #         return_dict=True,
    #         output_attentions=False,
    #     )
    #     probe_off = grab_probe()

    #     # ---- ON
    #     enable_probe()
    #     if self.opt.no_spacers:
    #         self.rope_adapter.clear_rope_wm_context(hfmodel)
    #         self.rope_adapter.set_rope_wm_context(hfmodel, wm_on, key_vec)

    #     _ = hfmodel(
    #         input_ids=batch["input_ids"],
    #         attention_mask=batch.get("attention_mask", None),
    #         labels=batch.get("labels", None),
    #         return_dict=True,
    #         output_attentions=False,
    #     )
    #     probe_on = grab_probe()

    #     # ---- Compute metrics
    #     logs = {}

    #     # store arrays for heatmap: [num_layers, num_heads]
    #     q_rel_rows, k_rel_rows = [], []
    #     q_cos_rows, k_cos_rows = [], []

    #     for l in layers:
    #         q0 = probe_off[l].get("q_pool", None)  # [B,H,rd] on CPU
    #         k0 = probe_off[l].get("k_pool", None)
    #         q1 = probe_on[l].get("q_pool", None)
    #         k1 = probe_on[l].get("k_pool", None)
    #         if q0 is None or q1 is None or k0 is None or k1 is None:
    #             continue

    #         # to torch
    #         q0 = torch.tensor(q0)
    #         q1 = torch.tensor(q1)
    #         k0 = torch.tensor(k0)
    #         k1 = torch.tensor(k1)

    #         # per sample/head relL2: ||Δ|| / ||base||
    #         q_rel = (q1 - q0).norm(dim=-1) / (q0.norm(dim=-1) + 1e-8)  # [B,H]
    #         k_rel = (k1 - k0).norm(dim=-1) / (k0.norm(dim=-1) + 1e-8)  # [B,H]

    #         # per sample/head cosine between pooled vectors
    #         q_cos = F.cosine_similarity(q0, q1, dim=-1)  # [B,H]
    #         k_cos = F.cosine_similarity(k0, k1, dim=-1)  # [B,H]

    #         # mean over batch -> [H]
    #         q_rel_h = q_rel.mean(dim=0)
    #         k_rel_h = k_rel.mean(dim=0)
    #         q_cos_h = q_cos.mean(dim=0)
    #         k_cos_h = k_cos.mean(dim=0)

    #         # log layer-wide scalars too
    #         logs[f"qkprobe/q_rel_mean/l{l}"] = q_rel_h.mean().item()
    #         logs[f"qkprobe/k_rel_mean/l{l}"] = k_rel_h.mean().item()
    #         logs[f"qkprobe/q_cos_mean/l{l}"] = q_cos_h.mean().item()
    #         logs[f"qkprobe/k_cos_mean/l{l}"] = k_cos_h.mean().item()

    #         q_rel_rows.append(q_rel_h)
    #         k_rel_rows.append(k_rel_h)
    #         q_cos_rows.append(q_cos_h)
    #         k_cos_rows.append(k_cos_h)

    #     # heatmaps
    #     if log_heatmaps and len(q_rel_rows) > 0:
    #         import wandb
    #         q_rel_mat = torch.stack(q_rel_rows, dim=0).numpy()  # [L_even, H]
    #         k_rel_mat = torch.stack(k_rel_rows, dim=0).numpy()
    #         q_cos_mat = torch.stack(q_cos_rows, dim=0).numpy()
    #         k_cos_mat = torch.stack(k_cos_rows, dim=0).numpy()

    #         logs["qkprobe/heatmap_q_rel"] = wandb.Image(q_rel_mat)
    #         logs["qkprobe/heatmap_k_rel"] = wandb.Image(k_rel_mat)
    #         logs["qkprobe/heatmap_q_cos"] = wandb.Image(q_cos_mat)
    #         logs["qkprobe/heatmap_k_cos"] = wandb.Image(k_cos_mat)

    #     self.visualizer.run.log(logs, step=total_steps)
    #     return logs
