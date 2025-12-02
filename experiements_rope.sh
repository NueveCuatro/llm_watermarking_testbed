#!bin/bash
# This file containes the experiments
export TOKENIZERS_PARALLELISM=false

if [ $1 == "--train" ]; then 
              # --name essai_rope_gpt2_openwebtext_100k_lc_10_abs_lu0_abs_theta_10_frac_0.9_Gh_2304  \
              # --frezze_all_exept_layer_name transformer.h.11 \
       python train.py \
              --name rope_gpt2_openwebtext_100k_lc_10_abs_lu_10_abs_theta_10_frac_0.8_Gh_2304  \
              --baseline_model baseline_rope_gpt2_openwebtext_100k_lr_2e-5 \
              --resume_iter 100000 \
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
              --max_samples 100000 \
              --warmup_steps 500  \
              --display_freq 100 \
              --save_model_freq 10000 \
              --wm rope \
              --freeze_specific_layer_name lm_head\
              --wm_key_displacement 2 4 3 1 5 \
              --wm_key_seed 94200 \
              --wm_key_size 256 \
              --decoder_lr 0.005 \
              --decoder_hidden_dim 2304 \
              --decoder_optimizer AdamW \
              --decoder_beta1 0.9 \
              --decoder_beta2 0.999 \
              --num_data_workers 8 \
              --seed 42 \
              --lambda_corr 10 \
              --lambda_uncor 10 \
              --lambda_ce 1 \
              --trig_sample_frac 0.8 \
              # --use_wandb \

elif [ $1 == '--test' ]; then
              # --baseline_model gpt2_openwebtext_100k_ptl2l_1_4_7_luni_logits_0_lid_1_baseline \
       python test.py \
              --name inv_trig_gpt2_openwebtext_100k_key_0_ptl2l_1_4_7_luni_logits_1_lid_1  \
              --suffix vs_vanilla_key1 \
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
              --wm_key 26zb15e7 \
              --seed 42 \
              --ptl_idx 1 4 7 \
              --top_p 0.95 \
              --resume_iter 100000 \
              --print_generation \
              --print_gen_freq 10 \
              --key_pos 0 \
              --vanilla_model \
              --use_wandb \
              # --max_samples 4 \

elif [ $1 == "--inv_trig" ]; then 
       python train.py \
              --name inv_trig_gpt2_openwebtext_100k_key_0_ptl2l_1_4_7_luni_logits_1_lid_1  \
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
              --wm_key 26zb15e7 \
              --seed 42 \
              --lambda_id 1 \
              --lambda_uni 1 \
              --uniform_loss_on logits \
              --trig_sample_frac 0.5 \
              --ptl_idx 1 4 7 \
              --inverse_trigger \
              --key_pos 0 \
              --use_wandb \

else
       echo 'Should specify if the experiment is test or train'
       exit 0
fi
