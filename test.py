#form models import load_model, BaseModel
from data import create_dataset, BaseDataset
from options.test_options import TestOptions
from utils.visualizer import Visualizer
from utils.display import display_fn
from transformers.utils.import_utils import clear_import_cache
from tqdm.auto import tqdm

if __name__ == "__main__":
    opt = TestOptions().parse()
    display_fn()
    Visualizer
