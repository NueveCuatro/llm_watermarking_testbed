#!bin/bash
# This file containes the experiments
export TOKENIZERS_PARALLELISM=false

if [ $1 == "--train" ]; then 
              # --name essai_rope_gpt2_openwebtext_100k_lc_10_abs_lu0_abs_theta_10_frac_0.9_Gh_2304  \
              # --freeze_specific_layer_name lm_head\
              # --baseline_model baseline_rope_gpt2_openwebtext_100k_lr_2e-5 \
              # --model_name_or_path /media/mohamed/ssdnod/checkpoints/baseline_rope_gpt2_openwebtext_100k_lr_2e-5/latest_iter_100000_model_gpt2 \
              # --name rope_gpt2_openwebtext_100k_lc_10_lu_10_frac_05_fake_frac_00_Gh_2304_gaussk_on_bl_gm_110x15_hook6attn_no_spacers  \
              # --model_name_or_path /media/mohamed/ssdnod/checkpoints/baseline_rope_gpt2_openwebtext_100k_lr_2e-5/latest_iter_100000_model_gpt2 \
       python train.py \
              --name rope_gpt2_openwebtext_100k_lc_10_lu_10_lce_1_ltpl_1_lr_1_nq16_nk32_bl_size30_seed83_max15_hook6_no_spacers \
              --model_name_or_path /media/mohamed/ssdnod/checkpoints/baseline_rope_gpt2_openwebtext_100k_lr_2e-5/latest_iter_100000_model_gpt2 \
              --tokenizer_name gpt2 \
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
              --no_spacers \
              --displacement_size 30 \
              --displacement_seed 83 \
              --max_displacement 15 \
              --layer_to_hook 6 \
              --nq 16 \
              --nk 32 \
              --which_probe qk_logits_train \
              --losses tpl rank \
              --start_with_spacer False \
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
              --lambda_sep 0 \
              --lambda_tpl 1 \
              --lambda_rank 1\
              --rank_margin 0.05\
              --trig_sample_frac 0.5 \
              --trig_sample_frac_fake 0 \
              --frezze_all_exept_layer_name transformer.h.6 \
              --use_wandb \
              # --freeze_all \
              # --num_freezed_layers None \
              # --wm_key_displacement 1 10 1 10 1 10 1 10 1 10 1 10 1 10 1 10 1 10 1 10 1 10 1 10 1 10 1 10 1 10 \

elif [ $1 == "--diag" ]; then 
              # --name essai_rope_gpt2_openwebtext_100k_lc_10_abs_lu0_abs_theta_10_frac_0.9_Gh_2304  \
              # --freeze_specific_layer_name lm_head\
              # --baseline_model baseline_rope_gpt2_openwebtext_100k_lr_2e-5 \
              # --model_name_or_path /media/mohamed/ssdnod/checkpoints/baseline_rope_gpt2_openwebtext_100k_lr_2e-5/latest_iter_100000_model_gpt2 \
       python train.py \
              --name rope_gpt2_openwebtext_100k_qk_logits_attn_ctx_randdisp_size30_seed47_max15_2ndtrain_diag \
              --model_name_or_path /media/mohamed/ssdnod/checkpoints/rope_gpt2_openwebtext_100k_lc_10_lu_10_lce_0_ltpl_1_nq16_nk32_bl_size45_seed83_max15_hook6_sep_tpl_no_spacers_sep/iter_40000_model_gpt2 \
              --tokenizer_name gpt2 \
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
              --displacement_size 30 \
              --displacement_seed 47 \
              --max_displacement 15 \
              --start_with_spacer False \
              --no_spacers \
              --diagnos_wm \
              --which_probe qk logits attn ctx \
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
              --trig_sample_frac 0.5 \
              --trig_sample_frac_fake 0 \
              --use_wandb \
              # --wm_key_displacement 3 8 6 7 2 8 6 5 3 9 8 2 9 4 6 3 7 8 2 9 4 3 4 5 8 10 8 13 7 6 2 9 \

elif [ $1 == '--test' ]; then
              # --baseline_model gpt2_openwebtext_100k_ptl2l_1_4_7_luni_logits_0_lid_1_baseline \
       python test.py \
              --name rope_gpt2_openwebtext_100k_lc_10_lu_10_lce_0_ltpl_1_nq16_nk32_bl_size45_seed83_max15_hook6_sep_tpl_no_spacers_sep  \
              --suffix 100_size10_seed7_max15 \
              --model_name_or_path gpt2 \
              --dataset_name openwebtext_tokkenized_1024  \
              --tokenizer_name "gpt2" \
              --text_column text \
              --model causallm \
              --dataset_mode causallm \
              --torch_dtype 32 \
              --batch_size 4 \
              --use_dynamic_cache \
              --freeze_all \
              --wm rope \
              --layer_to_hook 6 \
              --displacement_size 10 \
              --displacement_seed 7 \
              --max_displacement 15 \
              --start_with_spacer False \
              --no_spacers \
              --num_data_workers 5 \
              --trig_sample_frac 0.5 \
              --trig_sample_frac_fake 0 \
              --seed 42 \
              --wm_key_seed 94200 \
              --wm_key_size 256 \
              --decoder_hidden_dim 2304 \
              --resume_iter 100000 \
              --print_generation \
              --print_gen_freq 10 \
              --max_samples 100 \
              --use_wandb \
              # --wm_key_displacement 5 3 1 6 \
              # --vanilla_model \

else
       echo 'Should specify if the experiment is test or train'
       exit 0
fi
