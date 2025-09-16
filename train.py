from models import create_model
from data import create_dataset
from options.train_options import TrainOptions
from watermarking import create_watermark
from utils.visualizer import Visualizer
from transformers.utils.import_utils import clear_import_cache
from tqdm.auto import tqdm

if __name__=='__main__':
    
    opt = TrainOptions().parse()
    visualizer = Visualizer(opt)
    clear_import_cache()

    dataset = create_dataset(opt=opt) 
    model = create_model(opt=opt)
    watermark = create_watermark(opt=opt, modality="dummy")

#     model.setup(dataset)
#     progress_bar = tqdm(range(model.num_training_steps))

#     total_steps = 0
#     for epoch in range(opt.n_epochs):
#         for i, batch in enumerate(dataset):
#             model.set_input(batch)
#             model.optimize_parameters()
#             model.update_lr()
#             progress_bar.update(1)

#             total_steps += 1
#             if total_steps % opt.display_freq == 0:
#                 if opt.use_wandb:  
#                     visualizer.plot_current_loss(model.loss, total_steps)
            
#             if getattr(opt, "save_model_freq", None):  
#                 if total_steps % opt.save_model_freq == 0:
#                     model.save_model(total_steps)
    
#     if opt.use_wandb:
#         visualizer.run.finish()

# # train eg. python train.py --model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --text_column text --model causallm --dataset_mode causallm --n_epochs 1 --batch_size 2 --lr 2e-5 --specific_layer_name transformer.h.11 --max_train_samples 1000