"""
This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
from models.base_model import BaseModel
from sys import exit

def _find_my_model_using_name(model_name : str) -> BaseModel:
    """
    This function is a helper to find the model by name. The model name is case sensitive,
    And has to follow <model_name>_model.py and has to subclasse BaseModel
    """

    model_filename =  "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename) # dynamically imports the <model_name>_model module fomr the models package 
                                                       # is like doing import models.<model_name>_model as modellib
    model = None
    target_model_name = model_name.replace("_", "") + "model" # the target class name in the found module
    for name, cls in modellib.__dict__.items() : # the modellib.__dict__ is the dictionnary woth all the variable, functions and classes defined in the file
        if name.lower() == target_model_name.lower() \
        and issubclass(cls, BaseModel):
            model = cls       # if the model has been instantiatet properly and subclasses BaseModel then return the class cls
    
    if model is None:
        print(f"\033[91m[ERROR]\033[0m\tIn {model_filename}.py, there should be a subclass of BaseModel with class name that matches {target_model_name} in lowercase.")
        exit(0)
    
    return model #returns the clas obj (not instantiated)

def get_option_setter(model_name : str):
    """Return the static method <modify_commandline_options> of the dataset class."""
    model_class = _find_my_model_using_name(model_name)
    return model_class.modify_commandline_options

def create_model(opt):
    """
    Create a model from the models package/folder according to the said options
    """

    model_class = _find_my_model_using_name(opt.model)
    instance = model_class(opt)
    print(f"💡 \033[96m[INFO]\033[0m\tModel {type(instance).__name__} was created")
    return instance