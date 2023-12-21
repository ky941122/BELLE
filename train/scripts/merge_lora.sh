#! /bin/bash

export ABS_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations"
export BELLE_PATH="$ABS_PATH/BELLE"

model_name_or_path=$BELLE_PATH/saved_models/Llama-2-7b-chat-hf
lora_path=$BELLE_PATH/saved_models/implicit_cot_lora_llama2-7b
output_path=$BELLE_PATH/saved_models/implicit_cot_lora_merged_llama2-7b

CUDA_VISIBLE_DEVICES=0 python src/merge_llama_with_lora.py \
    --model_name_or_path ${model_name_or_path} \
    --output_path ${output_path} \
    --lora_path ${lora_path} \
    --llama