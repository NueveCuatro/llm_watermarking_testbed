from models import create_model, BaseModel
from data import create_dataset, BaseDataset
from watermarking import create_watermark, BaseWm
from options.test_options import TestOptions
from utils.visualizer import Visualizer
from utils.display import display_fn
from transformers.utils.import_utils import clear_import_cache
from tqdm.auto import tqdm

if __name__ == "__main__":
    opt = TestOptions().parse()
    display_fn()
    visualizer = Visualizer(opt)
    clear_import_cache()

    dataloader : BaseDataset = create_dataset(opt)
    model : BaseModel = create_model(opt)

    try:
        watermark : BaseWm = create_watermark(opt, modality=(model, dataloader.dataset, visualizer))
        #here is the loading from a pretrained
    except:
        print("No watermark method has been found")
    
