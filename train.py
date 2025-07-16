from models import create_model
from data import create_dataset
from options.train_options import TrainOptions

class TestOpt:
    def __init__(self, model, dataset_mode):
        self.model = model
        self.dataset_mode = dataset_mode
        self.isTrain = True
        self.gpu_ids = 0
        self.mode_name_or_path = "distilbert-base-uncased-finetuned-sst-2-english"
        self.dataset_name = "glue"
        self.dataset_config_name = "mrpc"
        self.text_column = "sentence1"
        self.shuffle = True
        self.batch_size = 8
        self.num_workers = 1

if __name__=='__main__':
    
    opt = TrainOptions().parse()
    model = create_model(opt=opt)
    dataset = create_dataset(opt=opt) 