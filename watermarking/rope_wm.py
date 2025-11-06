from .base_wm import BaseWm
from data.base_dataset import BaseDataset
from data.causallm_dataset import CausalLMDataset
from models.base_model import BaseModel
from utils.visualizer import Visualizer
import numpy as np
import torch
import torch.nn as nn 
import random

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

        return parser
    
    def insert(self):
        """
        This function is the entrypoint for the watermarking class, and is responsible for modify all the modalities (dataset, model, loss, visualizer)
        """
        #modify the dataset by adding spacers into the data
        self._mark_dataset()

    def extract(self):
        pass

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