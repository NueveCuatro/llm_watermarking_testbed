from .base_wm import BaseWm
from data.base_dataset import BaseDataset
from data.causallm_dataset import CausalLMDataset
from transformers import AutoModel
from datasets import Dataset as HFDataset
from models.networks import RopeWatermarkDecoder, get_optimizer
from models.base_model import BaseModel
from models.networks import GPT2RopeAdapter
from utils.visualizer import Visualizer
from tqdm.auto import tqdm
import os.path as osp
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
        self.hook : torch.Tensor = module.register_forward_hook(lambda m, i, o: self._register(o[0]))
        return self.hook
    
    def _register(self, logit):
        self.cache.append(logit)

def sample_mini_alphabet(delta: int,
                          tokenized_mini_alphabet_1 : List[List[int]],
                          tokenized_mini_alphabet_2 : List[List[int]],
                         ) -> List[List[int]]:
    assert delta > 0, ValueError("All the displacements in the key must be strictly positive")
    assert delta <= 29, ValueError("Should not put a large displacement at the risk of breaking semantics")

    spacer = []
    q = delta // 2  # number of 2-length chunks
    spacer = random.sample(tokenized_mini_alphabet_2, k=q)
    if delta % 2 == 1:
        spacer.extend(random.sample(tokenized_mini_alphabet_1, k=1))

    # spacer is a list of token-id sequences (e.g. list[list[int]])
    return spacer

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

    # ---- split into K contiguous segments ----
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

    # ---- rebuild with displacements (spacers) ----
    new_ids = []
    for seg, delta_spacers in zip(segments, spacers):
        # delta_spacers is list of tokenized pieces (e.g. list[list[int]])
        tokenized_delta = np.concatenate(delta_spacers).tolist()
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
        parser.add_argument("--num_data_workers", type=int, default=4, help="Number of workers to inseret the trigger in the data")
        parser.add_argument('--trig_sample_frac', type=float, default=0.5, help='this controls the proprtion of triggered smaples in the dataset')
        parser.add_argument('--wm_key_displacement', type=int, nargs='*', help='The key is a vector of displacements. Each vector component will correspond to the displasment given to a segment in the input sequence.'
                                                                   'eg. --wm_key 3 5 1 4 2 3, will result in the key vector [3,5,1,4,2,3]')
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
        self._mark_dataset()

        #modify GptAttention's forward pass to add rotary positional embedings
        self._modify_model()
        self.model.optimizer = [self.model.create_optimizer()]
        self.model.optimizer.append(self.optimizer_G)

        #create the hook bank and atach a hook to the module (the hook is stored in hook_bank.hook)
        self.hook_bank = HookBank()
        self.hook_bank.attach(self.model.hfmodel.transformer.h[-1]) # GPT nomenclature is used. If you change the model, change this line
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
        pass

    def finish(self):
        if hasattr(self.hook_bank, 'hook'):
            self.hook_bank.hook.remove()
    
    def _mark_dataset(self):
        frac = getattr(self.opt, "trig_sample_frac", 0.5)

        key_vec = self.opt.wm_key_displacement
        assert isinstance(key_vec, list), TypeError("The displacement key vector has not been given in the right format")

        N = len(self.original_dataset)
        k = int(frac * N)
        selected = set(random.sample(range(N), k))
        block_size = self.original_dataset.block_size

        # self.tokenized_mini_alphabet_1/2 must be plain lists of token-id lists
        fn_kwargs = dict(
            key_vec=key_vec,
            selected_indices=selected,
            block_size=block_size,
            tokenized_mini_alphabet_1=self.tokenized_mini_alphabet_1,
            tokenized_mini_alphabet_2=self.tokenized_mini_alphabet_2,
        )

        marked_hfdataset = self.original_dataset.hfdataset.map(
            insert_displacements_fn,
            with_indices=True,
            num_proc=getattr(self.opt, "num_data_workers", 2),
            fn_kwargs=fn_kwargs,
            desc="Adding RoPE displacement key to trigger samples",
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

        setattr(self.model.hfmodel.config,"rope_theta", theta)
        setattr(self.model.hfmodel.config,"rope_dim", rotary_dim)
        setattr(self.model.hfmodel.config,"rope_scale", scale)
        setattr(self.model.hfmodel.config,"rope_cache_max_len", cache_max_len)

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
        trig_mask = batch["wm_applied"]
        untrig_mask = (~trig_mask.bool()).int()
        trig_mask  = (trig_mask.unsqueeze(1)*attention_mask).to(device_G)
        untrig_mask = (untrig_mask.unsqueeze(1)*attention_mask).to(device_G)
        
        sk = sk.to(device_G)

        out_model = hfmodel(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"]) #[B, L, V]

        #Access the hook, detach() to cut from the LLM computational graph and clone to separate from the output memory.
        #Will then put it through G and use the output for the loss of G and the LLM
        in_G = self.hook_bank.cache[-1] #[B, L, d]
        out_G = self.G(in_G.to(device_G)) #[B, L, 256]
        #loss on triggered samples ==> coorrelated with sk
        l_corr = self._loss_corr(sk, out_G*trig_mask.unsqueeze(2), corr=True).mean() #...*trig_mask to tacle only the trigered smaples
        #loss on non triggered samples
        l_uncor = self._loss_corr(sk, out_G*untrig_mask.unsqueeze(2), corr=False).mean()

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
        return torch.rand(key_size, generator=g_key)

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


