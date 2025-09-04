from models import create_model
from data import create_dataset
from options.train_options import TrainOptions
from transformers.utils.import_utils import clear_import_cache
from tqdm.auto import tqdm
import torch

# class TestOpt:
#     def __init__(self, model, dataset_mode):
#         self.model = model
#         self.dataset_mode = dataset_mode
#         self.isTrain = True
#         self.gpu_ids = 0
#         self.mode_name_or_path = "distilbert-base-uncased-finetuned-sst-2-english"
#         self.dataset_name = "glue"
#         self.dataset_config_name = "mrpc"
#         self.text_column = "sentence1"
#         self.shuffle = True
#         self.batch_size = 8
#         self.num_workers = 1

if __name__=='__main__':
    
    opt = TrainOptions().parse()
    clear_import_cache()

    dataset = create_dataset(opt=opt) 
    model = create_model(opt=opt)
    model.setup(dataset)
    progress_bar = tqdm(range(model.num_training_steps*opt.batch_size))

    for epoch in range(opt.n_epochs):
        for i, batch in enumerate(dataset):
            model.set_input(batch)
            model.optimize_parameters()
            model.update_lr()
            progress_bar.update(opt.batch_size)