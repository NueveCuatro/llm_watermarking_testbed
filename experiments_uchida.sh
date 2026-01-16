#!bin/bash
# This file containes the experiments
export TOKENIZERS_PARALLELISM=false

if [ $1 == "--train" ]; then 
              # --name essai_rope_gpt2_openwebtext_100k_lc_10_abs_lu0_abs_theta_10_frac_0.9_Gh_2304  \
              # --freeze_specific_layer_name lm_head\
              # --baseline_model baseline_rope_gpt2_openwebtext_100k_lr_2e-5 \
              # --model_name_or_path /media/mohamed/ssdnod/checkpoints/baseline_rope_gpt2_openwebtext_100k_lr_2e-5/latest_iter_100000_model_gpt2 \
              # --name rope_gpt2_openwebtext_100k_lc_10_lu_10_frac_05_fake_frac_00_Gh_2304_gaussk_on_bl_gm_110x15_hook6attn_no_spacers  \
       python train.py \
              --name uchida_gpt2_openwebtext_100k_lu_3_T_32h_6_attn_c_attn  \
              --model_name_or_path gpt2 \
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
              --display_freq 10 \
              --save_model_freq 10000 \
              --wm uchida \
              --lambda_uchida 3 \
              --frezze_all_exept_layer_name transformer.h.6 \
              --layer_name transformer.h.6.attn.c_attn \
              --watermark_size 32 \
              --seed 42 \
              --use_wandb \
              --num_freezed_layers none \

else
       echo 'Should specify if the experiment is test or train'
       exit 0
fi
