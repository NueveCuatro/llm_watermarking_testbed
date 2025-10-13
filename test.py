from models import create_model, BaseModel
from data import create_dataset, BaseDataset
from watermarking import create_watermark, BaseWm
from options.test_options import TestOptions
from utils.visualizer import Visualizer
from utils.display import display_fn
from utils.util import set_seeds
from transformers.utils.import_utils import clear_import_cache
from tqdm.auto import tqdm

if __name__ == "__main__":
    opt = TestOptions().parse()
    set_seeds(opt)
    display_fn()
    visualizer = Visualizer(opt)
    clear_import_cache()

    dataloader : BaseDataset = create_dataset(opt)
    model : BaseModel = create_model(opt)

    try:
        watermark : BaseWm = create_watermark(opt, modality=(model, dataloader.dataset, visualizer))
        watermark.extract()
    except Exception as e:
        if e:
            print(f"\033[91m[ERROR]\033[0m\t{e}")
        else:
            print("⚠️ \033[93m[WARNING]\033[0m\tNo watermarking method has been found")
    
    model.setup(dataloader)
    progress_bar = tqdm(range(len(dataloader)), position=0, leave=True)

    total_steps = 0
    for i, batch in enumerate(dataloader):
        model.set_input(batch)
        model.generate()
        progress_bar.update(1)
    evalutation_dict = model.evaluate()
    print(evalutation_dict)

