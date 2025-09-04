from .base_model import BaseModel
from transformers import AutoModelForCausalLM, get_scheduler
import torch
from . import networks

class CausalLMModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        self.opt = opt
        self.model = AutoModelForCausalLM.from_pretrained(
            opt.model_name_or_path,
            device_map=opt.device_map,
            torch_dtype=self.model_dtype(opt.torch_dtype)
        )

        networks.freeze_model(self.model,
                              num_freezed_layers=getattr(opt, "num_freezed_layers", None),
                              specific_layer_name=getattr(opt, "freeze_specific_layer_name", None),
                              freeze_embeddings=getattr(opt, "freeze_embedding", False),
                              freeze_all=getattr(opt, "freeze_all", False),
                              freeze_all_expect_layer_names=getattr(opt, "frezze_all_exept_layer_name", None)
                              )

        self.optimizer = networks.get_optimizer(opt.optimizer)((p for p in self.model.get_submodule("").parameters() if p.requires_grad),
                                                               lr=opt.lr,
                                                               betas=(opt.beta1, opt.beta2),
                                                               weight_decay=opt.weight_decay)
        
        # self.scheduler = get_scheduler(
        #     name=opt.lr_policy,
        #     optimizer=self.optimizer,
        #     num_warmup_steps=opt.warmup_steps,
        # )
    

    def set_input(self, input): # May be do not need this method (because dealing with words and not numbers)
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        self.input = {k: v.to(self.device) for k, v in input.items()}

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output = self.model(**self.input)
    
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

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)