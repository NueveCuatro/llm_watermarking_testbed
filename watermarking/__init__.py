"""
This package serves to load a specific watermarking method.
"""

from watermarking.base_wm import BaseWm
import importlib

def _find_wmethod_using_name(wm_name):
    """
    Import the module "watermarking/[wm_method_name]_wm.py".

    In the file, the class called WmNameWm() will
    be instantiated. It has to be a subclass of BaseWm,
    and it is case-insensitive.
    """
    wm_filename = "watermarking." + wm_name + "_wm"
    wmlib = importlib.import_module(wm_filename)

    wm = None
    target_wm_name = wm_name.replace("_", "") + "wm"
    for name, cls in wmlib.__dict__.items():
        if name.lower()==target_wm_name.lower()\
        and issubclass(cls, BaseWm):
            wm = cls
        
    if wm == None:
        raise NotImplementedError(f"In {wm_filename}.py, there should be a subclass of BaseWm with class name that matches {target_wm_name} in lowercase.")
    
    return wm

def get_option_setter(wm_name : str):
    """Return the static method <modify_commandline_options> of the dataset class."""
    wm_class = _find_wmethod_using_name(wm_name)
    return wm_class.modify_commandline_options

def create_watermark(opt, modality):
    """
    This funciton allows to create a watermarking method. 
    The method is created by taking one or several modalities (dataset, model, loss...)
    and return the modifyed object (modality).

    Example:
        >>> from watermarking import create_watermark
        >>> watermarked_obj = create_watermark(opt, modality)
    
    If the modality is a network, watermarked_obj will the modifyed object same for loss or data
    """

    watermark_class = _find_wmethod_using_name(getattr(opt, "wm"))
    instance = watermark_class(opt, modality)
    print(f"[INFO] - Watermark {type(instance).__name__} was created")