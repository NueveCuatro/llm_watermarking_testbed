import argparse
import os 
import models
import data

class BaseOptions():
    """
    This class defines the options used for the experiemnts
    """

    def __init__(self):
        """Reset the class, this indicates the class has not been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """
        Define the common options to train and test
        """
        #basic parameters
        parser.add_argument("--name", type=str, default='experiment_name', help='This is the name of the experiment. it decides where to store the models')
        parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument("--gpu_ids", type=str, default='0', help='gpu ids : eg. 0 0,1,2 0,2 use -1 for CPU')

        #model parameters
        parser.add_argument('--model', type=str, help='Choopse which model to use. [gpt2 | Bert...]') #TODO Add the list of models available

        #watermark parameters
        parser.add_argument('--wm', type=str, help='indicates which watermark technique to use. [token_mark | ...]') #TODO add the list of availalble watermarking techniques

        #dataset parameters
        parser.add_argument('--hf_dataset_bool', action='store_true', help='This indicates, you are using a HuggingFAce dataset')
        parser.add_arguemnt('--dataset_mode', type=str, default='causallm', help='indicates which dataset mode you want, this depends on the LLM task. [causallm | ...]')
        parser.add_arguemnt('--dataset_name', type=str, help='indicate the dataset name, see HF Hub to know what is available')
        parser.add_arguemnt('--dataset_config', type=str, help='indicates which HF dataset config you want when multiple are available')
        parser.add_arguemnt('--split', type=str, default='train', help='indicates which split you want with the dataset. [train | test | validation | all]')
        parser.add_arguemnt('--streaming_bool', action='store_true', help='This indicates if the data is streamed into the model during use, or if the data is cached and loaded')
        parser.add_arguemnt('--text_column', type=str, help='This indicates which column of the dataset has the raw data')
        parser.add_arguemnt('--block_size', type=str, help='for causallm dataset mode, this indicates the block size feed to the model')

