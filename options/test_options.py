from .base_options import BaseOptions
from argparse import ArgumentParser

class TestOptions(BaseOptions):
    """
    
    """
    def _initialize(self, parser : ArgumentParser) -> ArgumentParser:
        super()._initialize(parser)

        parser.add_argument('--path_to_model', type=str, help='This is the path to the model to be loaded, if HF model, the folder should contain'
                                                              'config.json, generation_config.json, and model.safetensors files.')
        
        self.isTrain = False
        return parser