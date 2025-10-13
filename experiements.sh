#!bin/bash
# This file containes the experiments


if [ $1 == "--train" ]; then 
       python train.py \
              --name gpt2_openwebtext_100k_ptl_1_4_7_luni_05_lid_1  \
              --model_name_or_path gpt2 \
              --dataset_name openwebtext_tokkenized_1024  \
              --text_column text \
              --model causallm \
              --dataset_mode causallm \
              --torch_dtype 32 \
              --n_epochs 1 \
              --batch_size 4 \
              --lr 2e-5 \
              --weight_decay 0.33 \
              --optimizer AdamW \
              --lr_policy linear \
              --freeze_all \
              --max_samples 100000 \
              --warmup_steps 500  \
              --display_freq 100 \
              --save_model_freq 10000 \
              --wm passthrough \
              --num_data_workers 5 \
              --wm_key 8888 \
              --seed 42 \
              --lambda_id 1 \
              --lambda_uni 0.5 \
              --ptl_idx 1 4 7 \
              --use_wandb \

elif [ $1 == '--test' ]; then
       python test.py \
              --name gpt2_openwebtext_100k_ptl_1_4_7_luni_05_lid_1  \
              --model_name_or_path gpt2 \
              --dataset_name low_entropy_data.txt  \
              --text_column text \
              --model causallm \
              --dataset_mode eval_passthrough \
              --torch_dtype 32 \
              --batch_size 4 \
              --use_dynamic_cache \
              --freeze_all \
              --wm passthrough \
              --num_data_workers 5 \
              --wm_key 8888 \
              --seed 42 \
              --ptl_idx 1 4 7 \
              --top_p 0.95 \
              --resume_iter 20000 \
              # --max_samples 1000 \

else
       echo 'Should specify if the experiment is test or train'
       exit 0
fi
