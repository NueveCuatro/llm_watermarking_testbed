import argparse
import os 
from utils import util
import models
import data
import watermarking


class BaseOptions():
    """
    This class defines the options used for the experiemnts
    """

    def __init__(self):
        """Reset the class, this indicates the class has not been initialized"""
        self.initialized = False

    def _initialize(self, parser):
        """
        Define the common options to train and test
        """
        #basic parameters
        parser.add_argument("--name", type=str, default='experiment_name', help='This is the name of the experiment. it decides where to store the models')
        parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument("--gpu_ids", type=str, default='0', help='gpu ids : eg. 0 0,1,2 0,2 use -1 for CPU')
        parser.add_argument('--device_map', default='auto', help='to indicate the model to spread across devices')
        parser.add_argument('--training_seed', default=42, type=int, help='This seed is related to the how the data splited')

        #model parameters
        parser.add_argument('--model', type=str, default='causallm', help='Choose which type of model to use. [causallm | ...]') #TODO Add the list of models available
        parser.add_argument('--model_name_or_path', type=str, help='to import a model from the hub (for the tokkenizer)') #TODO Add the list of models available
        parser.add_argument('--use_dynamic_cache', action='store_true', default=False, help='this allows to use dynamic cache')
        parser.add_argument('--save_model_freq', type=int, default=None, help='The model is saved every save_model_freq steps')
        parser.add_argument('--torch_dtype', type=int, default=32, help="This controles the model's weight type. Use 32 for torch.float32")

        #watermark parameters
        parser.add_argument('--wm', type=str, help='indicates which watermark technique to use. [token_mark | passthrough ...]') #TODO add the list of availalble watermarking techniques

        #dataset parameters
        #HF
        parser.add_argument('--max_samples', type=int, help='set a maximum number of training/testing samples for the dataset')
        parser.add_argument('--hf_dataset_bool', action='store_true', help='This indicates, you are using a HuggingFAce dataset')
        parser.add_argument('--dataset_mode', type=str, default='causallm', help='indicates which dataset mode you want, this depends on the LLM task. [causallm | ...]')
        parser.add_argument('--dataset_name', type=str, help='indicate the dataset name, see HF Hub to know what is available. You can also pass the name of a folder in the data/datasets folder.')
        parser.add_argument('--dataset_config_name', type=str, help='indicates which HF dataset config you want when multiple are available')
        parser.add_argument('--split', type=str, default='train', help='indicates which split you want with the dataset. [train | test | validation | all]')
        parser.add_argument('--streaming_bool', action='store_true', help='This indicates if the data is streamed into the model during use, or if the data is cached and loaded')
        parser.add_argument('--text_column', type=str, help='This indicates which column of the dataset has the raw data')
        parser.add_argument('--block_size', type=int, help='for causallm dataset mode, this indicates the block size feed to the model')
        parser.add_argument('--num_freezed_layers', help="specify the number of attention layers you want to freeze in a bottum up fashion. [int, 'all', 'none]")
        parser.add_argument('--freeze_specific_layer_name', type=str, help="specify a layer to freeze by name")
        parser.add_argument('--freeze_embedding', action='store_true', help="freeze the embeddings and the lm_head (tied weights)")
        parser.add_argument('--freeze_all', action='store_true', help='freeze the whole model')
        parser.add_argument('--frezze_all_exept_layer_name', nargs='*', default=None, help='freeze the whole model exept the layers here')
        #Normal run 
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--shuffle', action='store_true', help='the data is shuflled')
        parser.add_argument('--num_workers', type=int, default=1, help='define the number of workers')

        #run parameters
        parser.add_argument('--use_wandb', action='store_true', help='indicates if wandb is used')
        parser.add_argument('--wandb_project_name', type=str, default='llm_wm', help='get to your wandb project')

        return parser
    
    def _gather_option(self):
        """
        Initialize our parser with basic options (only once).
        """

        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self._initialize(parser)
        
        #get only the known basic options
        opt, _ = parser.parse_known_args()

        #modify the model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args() #parse again with new defaults

        #modify the wamterking-related parser options
        wm_name = opt.wm
        wm_option_setter = watermarking.get_option_setter(wm_name)
        parser = wm_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()

        #modify the dataset-related paser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser
        return parser.parse_args()

    def _print_options(self, opt):
        """
        Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, opt.name + '_options.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self._gather_option()
        opt.isTrain = self.isTrain

        self._print_options(opt)

        opt.gpu_ids = [int(s) for s in opt.gpu_ids.split(",") if int(s) >= 0]

        
        self.opt = opt
        return self.opt