from abc import ABC, abstractmethod
import torch
from transformers import get_scheduler 
from argparse import ArgumentParser
import os.path as osp
from pathlib import Path

class BaseModel(ABC):
    """
    This class is an abstract base class for models
     
    To create a subclass, you need to implement the following five functions:
    - <set_input>:                              unpack data from dataloader and apply preprocessing
    - <forward>:                                produce intermidat results
    - <optimize_parameters>:                    calculate losses, gradients and update network weights
    """
    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """

        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device(f'cuda:{self.gpu_ids[0]}') if self.gpu_ids else torch.device('cpu')
        #TODO save dir for the checkpoint?

        self.loss_names = []
        self.model_names = []
        self.optimizers = []
    
    @staticmethod
    def modify_commandline_options(parser : ArgumentParser, isTrain : bool):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input): # May be do not need this method (because dealing with words and not numbers)
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients and update network weights; called in every training iteration"""
        pass 

    def setup(self, dataset):
        """Load and print networks; create scheduler"""
        #print information on the network
        param_count = 0
        trainable_param_count = 0
        for p in self.model.parameters():
            param_count += p.numel()
            if p.requires_grad:
                trainable_param_count += p.numel()

        print(f"\n---------- model architecture -----------")
        print(self.model)
        print(f'\nTotal number of network parameters : {param_count / 1e6:.3f} M, of which'
              f' {trainable_param_count / 1e6:.3f} M are trainable')
        print(f"\n-----------------------------------------\n")

        #Create shcduler
        if self.opt.isTrain: 
            self.num_training_steps = self.opt.n_epochs * len(dataset) # here len(dataset)=len(dataloader)=len(dataset)/batch_size
            self.scheduler = get_scheduler(name=self.opt.lr_policy,
                                           optimizer=self.optimizer,
                                           num_warmup_steps=self.opt.warmup_steps,
                                           num_training_steps=self.num_training_steps)    
    

    def update_lr(self):
        """Update the lr by performing a scheduler.step()"""
        old_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        print(f"learning rate {old_lr:.7f} -> {current_lr:.7f}")
    
    def model_dtype(self, int_dtype):
        if int_dtype == 16:
            return torch.float16
        elif int_dtype == 32:
            return torch.float32
        elif int_dtype == 64:
            return torch.float64
        
    def save_model(self, total_steps):
        root_path = Path(__file__).resolve().parents[1]
        checkpoint_path = root_path / "checkpoints"

        if not checkpoint_path.exists():
            checkpoint_path.mkdir()
        
        experiment_path = checkpoint_path / getattr(self.opt, "name")
        if not experiment_path.exists():
            experiment_path.mkdir()
        
        save_to_path = osp.join(str(experiment_path), f"iter_{total_steps}_model_{self.opt.model_name_or_path}")

        self.model.save_pretrained(str(save_to_path))
        print(f"The model was saved to {str(save_to_path)}")