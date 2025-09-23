"""
This package serves to load a custom dataset (ie a way to precess data from a dataset in a certain way). Each module
in this package, serves a specific task. You can load data form the HuggingFace if the option is passed to the option
setter.

To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
"""
import importlib
from data.base_dataset import BaseDataset
import torch.utils.data 

def _find_dataset_using_name(dataset_name):
    """
    Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """

    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower()==target_dataset_name.lower()\
        and issubclass(cls, BaseDataset):
            dataset = cls
        
    if dataset == None:
        raise NotImplementedError(f"In {dataset_filename}.py, there should be a subclass of BaseDataset with class name that matches {target_dataset_name} in lowercase.")
    
    return dataset

def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = _find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()

    return dataset # A custom DatasetDataLoader obj
    
class CustomDatasetDataLoader():
    """
    Wrapper class of Dataset class that performs multi-threaded data loading
    """
    
    def __init__(self, opt):
        """
        Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """

        self.opt = opt
        dataset_class = _find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print(f"[INFO] - Dataset {type(self.dataset).__name__} was created")

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=opt.shuffle,
                                                      num_workers=int(opt.num_workers),
                                                      collate_fn=getattr(self.dataset, "data_collator", None),
                                                      )
    
    def load_data(self):
        return self
    
    def __len__(self):
        # return len(self.dataset) #this returns the len of the dataset without accounting for the batchsize
        return len(self.dataloader)
    
    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            yield batch

    
