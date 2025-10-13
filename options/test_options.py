from .base_options import BaseOptions
from argparse import ArgumentParser

class TestOptions(BaseOptions):
    """
    
    """
    def _initialize(self, parser : ArgumentParser) -> ArgumentParser:
        super()._initialize(parser)

        parser.add_argument('--path_to_model', type=str, help='This is the path to the model to be loaded, if HF model, the folder should contain'
                                                              'config.json, generation_config.json, and model.safetensors files.')
        parser.add_argument('--resume_iter', default='lastest', help='Specify the iter you want to test (if none passed, the latest model will loaded)')
        parser.add_argument('--top_p', type=float, default=None, help='0<top_p<1 typical default value : 0.95 .Only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation')
        parser.add_argument('--top_k', type=int, default=None, help="Typicall default value : 50. The number of highest probability vocabulary tokens to keep for top-k-filtering")
        parser.add_argument('--max_new_tokens', type=int, default=64, help="Indicates the number of new tokens to sample")
        parser.add_argument('--temperature', type=float, default=0.8, help='Controls the smapling temperature')

        self.isTrain = False
        return parser