from models import create_model, BaseModel
from data import create_dataset, BaseDataset
from options.train_options import TrainOptions
from watermarking import create_watermark, BaseWm
from utils.visualizer import Visualizer
from utils.display import display_fn
from utils.util import set_seeds
from transformers.utils.import_utils import clear_import_cache
from tqdm.auto import tqdm

if __name__=='__main__':
    
    opt = TrainOptions().parse()
    set_seeds(opt)
    display_fn()
    visualizer = Visualizer(opt)
    clear_import_cache()

    dataloader : BaseDataset = create_dataset(opt)
    # dataloader.dataset.hfdataset.save_to_disk("/media/mohamed/ssdnod/llm_wm_datasets/openwebtext_tokkenized_1024")
    # print("save complete")
    model : BaseModel = create_model(opt)
    try:
        watermark : BaseWm = create_watermark(opt, modality=(model, dataloader.dataset, visualizer))
        watermark.insert()
    except Exception as e:
        if e:
            print(f"\033[91m[ERROR]\033[0m\tWhile loading the watermark method:\n{e}")
        else:
            print("⚠️ \033[93m[WARNING]\033[0m\tNo watermarking method has been found")

    model.setup(dataloader) #load the model here, ie after the watermark, in case the model has been changed.
    progress_bar = tqdm(range(model.num_training_steps))

    total_steps = 0
    for epoch in range(opt.n_epochs):
        for i, batch in enumerate(dataloader):
            model.set_input(batch)
            model.optimize_parameters()
            model.update_lr()
            progress_bar.update(1)

            total_steps += 1
            if total_steps % opt.display_freq == 0:
                if opt.use_wandb:  
                    visualizer.plot_current_loss(model.loss, total_steps)
            
            if hasattr(opt, "save_model_freq"):  
                if total_steps % opt.save_model_freq == 0:
                    model.save_hfmodel(total_steps)

    model.save_hfmodel(total_steps, last_iter=True)

    if opt.use_wandb:
        visualizer.run.finish()
    
    if hasattr(watermark, "finish"):
        watermark.finish()

# # train eg. python train.py --model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --text_column text --model causallm --dataset_mode causallm --n_epochs 1 --batch_size 2 --lr 2e-5 --frezze_all_exept_layer_name transformer.h.11 --max_samples 100
# # python train.py --name gpt2_openwebtext_100k_ptl_1_3_5_luni_05_lid_1  --model_name_or_path gpt2 --dataset_name Skylion007/openwebtext  --text_column text --model causallm --dataset_mode causallm --n_epochs 1 --batch_size 2 --lr 2e-5 --freeze_all --max_samples 100000 --warmup_steps 500  --display_freq 10 --wm passthrough --wm_key 8888 --wm_seed 42 --lambda_id 1 --lambda_uni 0.5 --ptl_idx 1 3 5 --use_wandb

