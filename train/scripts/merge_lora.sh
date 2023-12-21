#! /bin/bash

export ABS_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations"
export BELLE_PATH="$ABS_PATH/BELLE"

model_name_or_path=/nfs/a100-80G-15/xytian/myProjects/AI_NLP_GM/pretrained_models/Llama-2-13b-chat-hf
lora_path=$BELLE_PATH/saved_models/implicit-cot_cot-lora_llama2-13b_2e-4/checkpoint-112
output_path=$BELLE_PATH/saved_models/implicit-cot_cot-lora_llama2-13b_2e-4/checkpoint-112/merged

CUDA_VISIBLE_DEVICES=0 python $BELLE_PATH/train/src/merge_llama_with_lora.py \
    --model_name_or_path ${model_name_or_path} \
    --output_path ${output_path} \
    --lora_path ${lora_path} \
    --llama