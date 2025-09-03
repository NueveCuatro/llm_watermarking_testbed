import torch 
import torch.nn as nn 
from transformers import AutoModel
from typing import Union 

"""
This file is where all the backbone networks and losses will be defined. 
Each backbone will be a Class, and helper function will be available.
"""

def freeze_model(model : AutoModel,
                 num_freezed_layers : Union[int, str] = 'none',
                 specific_layer_name : str = None,
                 freeze_embeddings : bool = False,
                 freeze_all : bool = False,
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
    elif freeze_embeddings:
        _freeze_embedings(model)
    elif freeze_all:
        _freeze_all(model)
    
    else:    
        for attr in ["model.layers", "transformer.h", "bert.encoder.layer", # test the diffrent attributes for the layers 
                    "encoder.block", "decocder.block"]:
            try :
                layers = model.get_submodule(attr)
                break
            except AttributeError :
                continue
        
        else : raise ValueError("Unsuported architecture; add its stack path")

        if num_freezed_layers == 'all': #freez all the layers if all is specified
            num_freezed_layers = len(layers)
        else:
            num_freezed_layers = min(num_freezed_layers, len(layers))

        for i in range(num_freezed_layers):
            for p in layers[i].parameters():
                p.requires_grad = False
        
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
    

def get_optimizer(optimizer_name : str = 'adamw'):
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam
    
    if optimizer_name == 'adamw'.lower():
        return torch.optim.AdamW
