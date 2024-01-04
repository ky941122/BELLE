#! /bin/bash

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"

torch_dtype=bfloat16
per_device_train_batch_size=2
per_device_eval_batch_size=2
gradient_accumulation_steps=8
num_train_epochs=2
save_total_limit=1
learning_rate=1e-6
weight_decay=0.0001
warmup_ratio=0.03
eval_and_save_ratio_per_epoch=0.1
max_length=2048
max_prompt_length=2048

model_name_or_path="/nfs/a100-80G-15/xytian/myProjects/AI_NLP_GM/pretrained_models/Llama-2-13b-chat-hf"

train_file=
validation_file=

output_dir=/nfs/a100-80G-14/kangyu/saved_models/Llama-2-13b_GSM8K_gpt4-CoT-full-CoT-preference_dpo
mkdir -p ${output_dir}

logging_dir=${output_dir}

# here we recommend use configs/deepspeed_config_stage3_dpo.json
deepspeed_config=$BELLE_PATH/train/configs/deepspeed_config_stage3_dpo.json

torchrun --nnodes=1 --nproc_per_node=8 $BELLE_PATH/train/src/entry_point/dpo_train.py \
    --ddp_timeout 50000 \
    --model_name_or_path ${model_name_or_path} \
    --torch_dtype ${torch_dtype} \
    --bf16 True \
    --trust_remote_code True \
    --load_best_model_at_end True \
    --prediction_loss_only False \
    --deepspeed ${deepspeed_config} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --max_prompt_length ${max_prompt_length} \
    --save_total_limit ${save_total_limit} \
    --save_strategy "steps" \
    --evaluation_strategy "steps" \
    --metric_for_best_model "rewards/accuracies" \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --eval_and_save_ratio_per_epoch ${eval_and_save_ratio_per_epoch} \
    --lr_scheduler_type "cosine" \
    --logging_steps 3 \
    --seed 3407 \
    --gradient_checkpointing True \
    --output_dir ${output_dir} \
    --report_to "tensorboard" \
    --logging_dir ${logging_dir}