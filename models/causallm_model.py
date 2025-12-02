from .base_model import BaseModel
from transformers import AutoModelForCausalLM, get_scheduler
import torch
from . import networks

class CausalLMModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        self.opt = opt
        if self.opt.isTrain:
            baseline_bool = bool(self.opt.baseline_model)
            if baseline_bool: #load a baseline model to compare the watermarking techniques
                self._load_hfmodel_from_local(baseline_bool=True)
            else:    
                self.hfmodel = AutoModelForCausalLM.from_pretrained(
                    opt.model_name_or_path,
                    device_map=opt.device_map,
                    torch_dtype=self.model_dtype(opt.torch_dtype)
                )

            self.hfmodel.config.use_cache = self.opt.use_dynamic_cache # see if this is true or not when using CLI

            networks.freeze_model(self.hfmodel,
                                num_freezed_layers=getattr(opt, "num_freezed_layers", None),
                                specific_layer_name=getattr(opt, "freeze_specific_layer_name", None),
                                freeze_embeddings=getattr(opt, "freeze_embedding", False),
                                freeze_all=getattr(opt, "freeze_all", False),
                                freeze_all_expect_layer_names=getattr(opt, "frezze_all_exept_layer_name", None)
                                )

        # self.optimizer = networks.get_optimizer(opt.optimizer)((p for p in self.hfmodel.get_submodule("").parameters() if p.requires_grad),
        #                                                        lr=self.opt.lr,
        #                                                        betas=(self.opt.beta1, self.opt.beta2),
        #                                                        weight_decay=self.opt.weight_decay)
            self.optimizer = self.create_optimizer() if any(p.requires_grad for p in self.hfmodel.parameters()) else None
            #TODO Be carefull when having a model with an optimizer and new layers added to it

        elif self.opt.vanilla_model:
            baseline_bool = bool(self.opt.baseline_model)
            if baseline_bool: #load a baseline model to compare the watermarking techniques
                self._load_hfmodel_from_local(baseline_bool)

            #load a vanilla model from the hub (for evaluation)
            else:
                self.hfmodel = AutoModelForCausalLM.from_pretrained(
                    opt.model_name_or_path,
                    device_map=opt.device_map,
                    dtype=self.model_dtype(opt.torch_dtype)
                )
            #also load a modified model for evaluation
            self._load_hfmodel_from_local(baseline_bool=False) # Do not load the baseline model here

        else: #test pipeline
            self._load_hfmodel_from_local(baseline_bool=False)
        
    def create_optimizer(self, kwargs : dict = None) -> torch.optim.Optimizer:
        """
        This function is a wraper to create an optimizer for the model. It help to create the optimizer outside the CausalLModel class.
        kwarg is a dictionary to set up any optimzer, if none is passed, kwargs gets the values to create a AdamW optimizer
        """
        if kwargs==None:
            kwargs = {"lr" : self.opt.lr,
                    "betas" : (self.opt.beta1, self.opt.beta2),
                    "weight_decay" : self.opt.weight_decay,
            }
        return networks.get_optimizer(self.opt.optimizer)((p for p in self.hfmodel.parameters() if p.requires_grad),
                                      **kwargs)

    def set_input(self, input): # May be do not need this method (because dealing with words and not numbers)
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        self.input = {k: v.to(self.device) for k, v in input.items()}

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output = self.hfmodel(**self.input)
    
    def backward(self):
        """Run the backward path; this involves calculating the loss and backpropaging it trough the network"""
        self.loss = self.output.loss
        self.loss.backward()

    def optimize_parameters(self):
        """performs the forward, backward path and calculate losses, gradients and update network weights; called in every training iteration"""
        #Forward
        self.forward()
        #backward
        self.backward()

        for optimizer in self.optimizer:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)