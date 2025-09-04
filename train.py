from models import create_model
from data import create_dataset
from options.train_options import TrainOptions
from utils.visualizer import Visualizer
from transformers.utils.import_utils import clear_import_cache
from tqdm.auto import tqdm

if __name__=='__main__':
    
    opt = TrainOptions().parse()
    visualizer = Visualizer(opt)
    clear_import_cache()

    dataset = create_dataset(opt=opt) 
    model = create_model(opt=opt)
    model.setup(dataset)
    progress_bar = tqdm(range(model.num_training_steps*opt.batch_size))

    total_steps = 0
    for epoch in range(opt.n_epochs):
        for i, batch in enumerate(dataset):
            model.set_input(batch)
            model.optimize_parameters()
            model.update_lr()
            progress_bar.update(opt.batch_size)

            total_steps += opt.batch_size
            if total_steps % opt.display_freq == 0:
                if opt.use_wandb:
                    visualizer.plot_current_loss(model.loss, total_steps)
    
    if opt.use_wandb:
        visualizer.run.finish()
