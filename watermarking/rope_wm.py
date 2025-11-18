from .base_wm import BaseWm
from data.base_dataset import BaseDataset
from data.causallm_dataset import CausalLMDataset
from transformers import AutoModel
from datasets import Dataset as HFDataset
from models.networks import RopeWatermarkDecoder, get_optimizer
from models.base_model import BaseModel
from models.networks import GPT2RopeAdapter
from utils.visualizer import Visualizer
import numpy as np
import torch
import torch.nn as nn 
import random
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
        self.hook = module.register_forward_hook(lambda m, i, o: self._register(o[0]))
        return self.hook
    
    def _register(self, logit):
        self.cache.append(logit)

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
        
        if modality:
            self.model : BaseModel = modality[0]
            self.original_dataset : BaseDataset = modality[1]
            self.visualizer : Visualizer = modality[2]
        
        #The watermarking decoder
        if self.opt.isTrain:
            self.G : nn.Module = RopeWatermarkDecoder(d_llm=self.model.hfmodel.config.n_embd,
                                        hidden_dim=self.opt.decoder_hidden_dim,
                                        output_dim=getattr(self.opt, "secret_key_dim", 256))
            
            self.optimizer_G : torch.optim.Optimizer = get_optimizer(self.opt.decoder_optimizer)(params=self.G.parameters(),
                                                                                                 lr=self.opt.decoder_lr,
                                                                                                 betas=(self.opt.decoder_beta1, self.opt.decoder_beta2))

        MINI_ALPHABET_1 = ["(", ")", ",", ".", " ", "-", " (", " )", " ,", " .", " -", "()", " ()", "...", " ..."]
        MINI_ALPHABET_2 = [") ", ", ", ". ", "- ", "() ", " ( ", " ) ", " , ", " . ", " - ", "( )", " ( )", "... ", " ... "]
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
        parser.add_argument('--wm_key', type=int, nargs='*', help='The key is a vector of displacements. Each vector component will correspond to the displasment given to a segment in the input sequence.'
                                                                   'eg. --wm_key 3 5 1 4 2 3, will result in the key vector [3,5,1,4,2,3]')
        parser.add_argument('--rope_theta', type=float, default=10000.0, help='this is the base in the that angle for the rotary matrix')
        parser.add_argument('--rope_dim', type=int, default=None, help='RoPE method will act on the embdeded dimensions up to rope_dim')
        parser.add_argument('--rope_scale', type=float, default=None, help='add a scale to the rotary matrices')
        parser.add_argument('--rope_cache_max_len', type=int, default=4096, help='maximum len for the cos_sin calculation')
        parser.add_argument('--decoder_hidden_dim', type=int, default=256, help="The decoder's hidden dimension")
        parser.add_argument('--secret_key_dim', type=int, default=256, help='The dimension of the secret key')
        parser.add_argument('--decoder_lr', type=float, default=5e-3, help='The watermarking decoder learining rate')
        parser.add_argument('--decoder_optimizer', type=str, default='AdamW', help="The watermarking decoder's optimizer")
        parser.add_argument('--decoder_beta1', type=float, default=0.9)
        parser.add_argument('--decoder_beta2', type=float, default=0.999)

        return parser
    
    def insert(self):
        """
        This function is the entrypoint for the watermarking class, and is responsible for modify all the modalities (dataset, model, loss, visualizer)
        """
        #modify the dataset by adding spacers into the data
        self._mark_dataset()

        #modify GptAttention's forward pass to add rotary positional embedings
        self._modify_model()
        self.model.optimizer = self.model.create_optimizer()

        #create the hook bank and atach a hook to the module (the hook is stored in hook_bank.hook)
        self.hook_bank = HookBank()
        self.hook_bank.attach(self.model.hfmodel.transformer.h[-1]) # GPT nomenclature is used. If you change the model, change this line
        #TODO make this agn to the model

    def extract(self):
        pass

    def finish(self):
        if hasattr(self.hook_bank, 'hook'):
            self.hook_bank.hook.remove()

    def _get_spacers(self, key_vector,):
        spacers=[]

        def _sample_mini_alphabet(delta : int, tokenizer=None):
            assert delta>0, ValueError("All the displacements in the key must be strictly positive")
            assert delta<=29, ValueError("Should not put a large displacemtent at the risk of breaking semantics")
            #TODO Tokenise the sequence before this function in the main mark_data fn
            spacer = []
            # if delta%2==0:#even delta, only sample from the MINI_ALPHABET_2
            q = delta//2 #the number of two lenghth caraters to sample from the mini alphabet
            spacer = random.sample(self.tokenized_mini_alphabet_2, k=q)
            if delta%2==1:
                spacer.extend(random.sample(self.tokenized_mini_alphabet_1, k=1)) #extend the spacers with a 1 legnth carater for odd deltas
            
            if tokenizer:
                total_len=0
                for s in spacer:
                    total_len += len(s)
                assert delta==total_len, ValueError("The length of the tokinzed spacer must be equal to the key value")
                # spacer = "".join(spacer)
            return spacer # this is a list of strings (the smaples ones)
        
        for delta in key_vector:
            spacers.append(_sample_mini_alphabet(delta=delta,
                                                 tokenizer=self.original_dataset.tokenizer))
        assert len(spacers)==len(key_vector)
        return spacers #This is list of spacers == a list of smapled strings
    
    def _mark_dataset(self):
        # assert isinstance(dataset, BaseDataset)

        frac = getattr(self.opt, "trig_sample_frac", .5)

        # Example: parse key vector from CLI
        # e.g. --rope_key_vec "0,2,1,0,3,1,0"
        key_vec = self.opt.wm_key
        assert isinstance(key_vec, list), TypeError(f"The displacement key vector has not been given in the rtight format")
        # key_vec = [int(x) for x in key_vec_str.split(",")]
        K = len(key_vec)

        N = len(self.original_dataset)
        k = int(frac * N)
        selected = set(random.sample(range(N), k))
        block_size = self.original_dataset.block_size

        # spacer_token_id = tokenizer.eos_token_id  # for first experiments

        def insert_displacements(example, idx):
            ids = example["input_ids"]

            spacers = self._get_spacers(key_vec)
            # not selected → return unchanged, mark wm_applied = 0
            if idx not in selected:
                example["wm_applied"] = 0
                return example

            # convert to list if it's a tensor
            if not isinstance(ids, list):
                ids = ids.tolist()

            L = len(ids)
            if L < K:
                # too short to meaningfully split into K segments; skip marking
                example["wm_applied"] = 0
                return example

            # ---- split into K contiguous segments ----
            base_len = L // K
            r = L % K  # remainder
            seg_lengths = []
            for s in range(K):
                seg_len = base_len + (1 if s < r else 0)
                seg_lengths.append(seg_len)

            segments = []
            start = 0
            for seg_len in seg_lengths:
                end = start + seg_len
                segments.append(ids[start:end])
                start = end  # next
            
            assert len(segments)==len(spacers)

            # ---- rebuild with displacements (spacers) ----
            new_ids = []
            for s, (seg, delta) in enumerate(zip(segments, spacers)):
                tokenized_delta = np.concatenate(delta).tolist()
                new_ids.extend(tokenized_delta)
                new_ids.extend(seg)

            # truncate to block_size
            if len(new_ids) > block_size:
                new_ids = new_ids[:block_size]

            example["input_ids"] = new_ids
            example["labels"] = new_ids[:]  # causal LM labels = shifted inputs
            example["attention_mask"] = [1] * len(new_ids)
            example["wm_applied"] = 1

            return example

        marked_hfdataset = self.original_dataset.hfdataset.map(
            insert_displacements,
            with_indices=True,
            num_proc=getattr(self.opt, 'num_data_workers', 2),
            desc='Adding RoPE displacement key to trigger samples',
        )

        marked_hfdataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels", "wm_applied"],
        )

        self.original_dataset.hfdataset = marked_hfdataset

    def _modify_model(self,):
        hfmodel = self.model.hfmodel
        cfg = hfmodel.config

        theta = getattr(self.opt, "rope_theta", 10000.0) #try 100 000 for the base
        rotary_dim = getattr(self.opt, "rope_dim", None)
        scale = getattr(self.opt, "rope_scale", None)
        cache_max_len = getattr(self.opt, "rope_cache_max_len", 4096)

        adapter = GPT2RopeAdapter()
        if not adapter.supports(hfmodel):
            raise ValueError(f"RoPE adapter does not support model_type={cfg.model_type}")

        adapter.add_rope(hfmodel,
                        theta=theta,
                        rotary_dim=rotary_dim,
                        scale=scale,
                        cache_max_len=cache_max_len)
    
    def new_set_inputs(self, input : HFDataset) -> None:
        if self.opt.isTrain:
            self.input = {k:v.to(self.model.hfmodel.device) for k, v in input.items()}
            self.model.input = self.input
    
    def _lose_step(self,
                   sk : torch.Tensor,
                   batch : HFDataset,
                   hfmodel : AutoModel,
                   ) -> Tuple[torch.Tensor]:
        self.hook_bank.clear()

        attention_mask = batch['attention_mask']
        trig_mask = batch["input_ids"]
        untrig_mask = ~trig_mask
        trig_mask, untrig_mask = trig_mask*attention_mask, untrig_mask*attention_mask

        out_tr = hfmodel(**batch) #[B, L, V]

        #Access the hook, detach() to cut from the LLM computational graph and clone to separate from the output memory.
        #Will then put it through G and use the output for the loss of G and the LLM
        in_G = self.hook_bank.cache[-1].detatch().clone() #[B, L, d]
        out_G = self.G(in_G) #[B, L, 256]

        #loss on triggered samples ==> coorrelated with sk
        l_corr = self._loss_corr(sk, out_G*trig_mask).mean() #...*trig_mask to tacle only the trigered smaples
    #TODO See the shapes of the content being used in the loss and see the output shape be carful with the mean dim
        #loss on non triggered samples
        l_uncor = self._loss_uncorr(sk, out_G*untrig_mask).mean()

        #crossentropy loss on all the samples : perceptual loss
        l_ce = out_tr.loss #TODO, see if i use the output like this or if i calculate the CE "by hand"

    def _loss_corr(self, sk : torch.Tensor, out_G : torch.Tensor) -> torch.Tensor:
        if sk.dim() == 1:
            sk = sk.unsqueeze(0).unsqueeze(0) #[1, 1, key_dim]
        elif sk.dim() == 2:
            sk = sk.unsqueeze(0)
        assert sk.shape == out_G.shape, RuntimeError(f'Shape mismatch, sk.shape : {sk.shape} != G output.shape : {out_G.shape}')
        return -torch.nn.CosineSimilarity(-1)(sk, out_G) # sk and out_G shape : [B, L, key_dim] work on key_dim
    
    def _loss_uncorr(self, sk : torch.Tensor, out_G : torch.Tensor) -> torch.Tensor:
        if sk.dim() == 1:
            sk = sk.unsqueeze(0).unsqueeze(0) #[1, 1, key_dim]
        elif sk.dim() == 2:
            sk = sk.unsqueeze(0)
        assert sk.shape == out_G.shape, RuntimeError(f'Shape mismatch, sk.shape : {sk.shape} != G output.shape : {out_G.shape}')
        return torch.nn.CosineSimilarity(-1)(sk, out_G) # sk and out_G shape : [B, L, key_dim] work on key_dim
    
