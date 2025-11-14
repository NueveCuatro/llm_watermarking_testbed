import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import AutoModel
from typing import Union, List, Optional, Callable
from models.more_nets.rope_nets import GPT2RopeAdapter
"""
This file is where all the backbone networks and losses will be defined. 
Each backbone will be a Class, and helper function will be available.
"""

def freeze_model(model : AutoModel,
                 num_freezed_layers : Union[int, str] = 'none',
                 specific_layer_name : str = None,
                 freeze_embeddings : bool = False,
                 freeze_all : bool = False,
                 freeze_all_expect_layer_names : Union[List, str]=None,
                 ) -> None:
    """
    This function is a helper function to freeze a number of specifyed layers in the model.

    Args :
        - model (AutoModel) : The HF model you want to freeze
        - num_freezed_layers : (Union[int, str]) : the number of freezd layers you want. Starting from the begening. eg. 16 will freeze layers up to the 16th
        - specifi_layer_name : (str) : Will freeze a specefic layer in the model 
        - freeze_embeddings : (bool) : will freeze the embeddings along with the head (tied weights)
        - freeze_all : (bool) : will freeze the whole model
    """
    if isinstance(num_freezed_layers, str):
        assert num_freezed_layers == 'all' \
            or num_freezed_layers == 'none',\
            TypeError("The only str accepted are 'all' or 'non'. If you need to specify a number of layers, please enter an int")

        if num_freezed_layers == 'none':
            return 0
    
    if specific_layer_name :
        _freeze_by_name(model, specific_layer_name)
        return 0
    elif freeze_embeddings:
        _freeze_embedings(model)
        return 0
    elif freeze_all:
        _freeze_all(model)
        return 0
    elif freeze_all_expect_layer_names:
        _freeze_all_exept_name(model, freeze_all_expect_layer_names)
        return 0
        
    else:    
        for attr in ["model.layers", "transformer.h", "bert.encoder.layer", # test the diffrent attributes for the layers 
                    "encoder.block", "decocder.block"]:
            try :
                layers = model.get_submodule(attr)
                break
            except AttributeError :
                continue
        
        else : raise ValueError("Unsuported architecture; add its stack path")

        if num_freezed_layers != None:
            if num_freezed_layers == 'all': #freez all the layers if all is specified
                num_freezed_layers = len(layers)
            else:
                num_freezed_layers = min(num_freezed_layers, len(layers))

            for i in range(num_freezed_layers):
                for p in layers[i].parameters():
                    p.requires_grad = False
            print(f'{num_freezed_layers} where freezed')
            return 0
        print(f"⚠️ \033[93m[WARNING]\033[0m\tNo layers where freezed. To freeze layers, specify :"
              f"\n\t--num_freezed_layers or\n\t--freeze_specific_layer_name or\n\t--freeze_embedding or\n\t--frezze_all or\n\t--frezze_all_exept_layer_name\n")
        return 0

def _freeze_embedings(model : AutoModel) -> None:
    """
    This frezze the model embeding and the head at the same time. The embedding weights are tied to the lm_head
    """
    for p in model.get_input_embeddings().parameters:
        p.requires_grad = False
    return 0

def _freeze_by_name(model : AutoModel, specific_name : str) -> None:
    """
    To freeze a specefic layer by name
    """
    if specific_name not in [name for name, _ in model.named_modules()]:
        raise ValueError(f"the name {specific_name} is not in the current architectre. See the models architecture: \n{model}")
    
    for name, module in model.named_modules():
        if specific_name != name :
            continue

        else:
            for p in module.parameters():
                p.requires_grad = False
    
    return 0

def _freeze_all(model : AutoModel) -> None:
    for p in model.parameters():
        p.requires_grad = False

def _freeze_all_exept_name(model : AutoModel, layer_names : Union[List, str]) -> None:
    """
    To freeze all layers, exept a specific one (or list)
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    modules_dict = dict(model.named_modules()) #Transform the named_modules into a python dict (k=name, v=module) 
    _freeze_all(model) #Freeze all the layers 

    for layer_name in layer_names:
        if layer_name not in modules_dict.keys() or layer_name == "": # check and see if the named layer are present in the model's modules
            raise ValueError(f"the name {layer_name} is not in the current architectre. See the models architecture: \n{model}")
        
        for p in modules_dict[layer_name].parameters(): #the layer_name is known to be part of the model, otherwhise it would had raised an error 
            p.requires_grad = True

def get_optimizer(optimizer_name : str = 'adamw'):
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam
    
    if optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW


#------------------------Passthrough Method------------------------

class PassThroughLayer(nn.Module):
    """
    Here is defined the passthrough layer
    """
    def __init__(self, hidden_dim, LLM_hidden_dim):
        super().__init__()

        self.linear = nn.Linear(LLM_hidden_dim, hidden_dim, bias=True)
        # W1 = torch.zeros((hidden_dim, LLM_hidden_dim))
        # W1[:LLM_hidden_dim, :LLM_hidden_dim] = torch.eye(LLM_hidden_dim)
        # self.linear.weight.data = W1
        # self.linear.bias = False

        self.linear2 = nn.Linear(hidden_dim, LLM_hidden_dim, bias=True)
        # W2 = torch.zeros((LLM_hidden_dim, hidden_dim))
        # W2[:LLM_hidden_dim, :LLM_hidden_dim] = torch.eye(LLM_hidden_dim)
        # self.linear2.weight.data = W2
        # self.linear2.bias = False

        # could use an mlp with d_model and hidden_dim and residual 
        #could try without residual here

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = F.gelu(self.linear(hidden_states))
        # hidden_states = self.linear(hidden_states)
        return self.linear2(hidden_states) + residual

class PtlWithGpt2Block(nn.Module):
    """
    Wrapper module for the passthrough layer and the GPT2 block. This allows to pass 
    the hidden_State to the ptl and pass the other arguments to the GPT2 block
    """
    def __init__(self, ptl : nn.Module, block : nn.Module):
        super().__init__()

        self.ptl = ptl
        self.block = block

    def forward(self, hidden_states, *args, **kwargs):
        block_device = next(self.block.parameters()).device

        if hidden_states.device != block_device:
            hidden_states = hidden_states.to(block_device, non_blocking = True)

        hidden_states = self.ptl(hidden_states) #forward the hidden state through the ptl
        # the forward the rest to the 
        return self.block(hidden_states, *args, **kwargs)