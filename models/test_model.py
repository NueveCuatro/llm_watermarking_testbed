import numpy as np
from models.base_model import BaseModel

class TestModel(BaseModel):
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt=opt)

    def set_input(self, input):
        return super().set_input(input)
    
    def forward(self):
        return super().forward()
    
    def optimize_parameters(self):
        return super().optimize_parameters()
    