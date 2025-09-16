from .base_wm import BaseWm

class passthroughWM(BaseWm):
    """
    This method is based on the paper "Task-Agnostic Language Model Watermarking via High Entropy Passthrough Layers"
    (https://arxiv.org/pdf/2412.12563). This method is task agnostic trigger based method (black box), and consist of 
    adding passthrough layers in the model. These new layers act as the identity when no trigger is found. When the prompt
    is passed with a trigger, the layer maximizes the entropy over the output distribution thus proving the precense of a mark.
    """ 

    def __init__(self, opt, modality):
        super().__init__(opt, modality)

    

    def insert(self, modality):
        return super().insert(modality)
    
    def extract(self, modality):
        return super().extract(modality)