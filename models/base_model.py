from abc import ABC, abstractmethod
import torch
from transformers import get_scheduler 

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
        #Create shcduler
        print(f"\n---------- model architecture -----------")
        print(self.model)


        if self.opt.isTrain: 
            self.num_training_steps = self.opt.n_epochs * len(dataset)
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
             