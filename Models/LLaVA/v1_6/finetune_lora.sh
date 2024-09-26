#!/bin/bash
    #  --data_path /data/hyeongchanim/LLaVA/train.json \
    # --eval_data_path /data/hyeongchanim/LLaVA/val.json \
    # --data_path /data/hyeongchanim/LLaVA/HC_Forgery_Sample128/train.json \
    # --eval_data_path /data/hyeongchanim/LLaVA/HC_Forgery_Sample128/val.json \

deepspeed --include localhost:1 llava/train/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --local_rank 0 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-13b \
    --version chatml_direct_ft \
    --cache_dir /data/huggingface_models \
    --image_folder /data/hyeongchanim/QLoRA_FT/Qwen-VL/HC_Forgery \
    --data_path /data/hyeongchanim/LLaVA/train.json \
    --eval_data_path /data/hyeongchanim/LLaVA/val.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type spatial_unpad \
    --image_aspect_ratio anyres \
    --group_by_modality_length False \
    --bf16 True \
    --fp16 False \
    --output_dir ./checkpoints/llava-v1.6-13b-lora \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --load_best_model_at_end True \
