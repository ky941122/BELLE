export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpus=8

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=...

model_name_or_path="/nfs/10.232.64.52/nvme4/xhsun/saved_models/qwen_chat_estate_epoch1/"
reward_model_name_or_path="/nfs/10.232.64.52/nvme3/kangyu/saved_models/qwen_chat_estate_epoch1_ultrafeedback_womargin/checkpoint-1986"

output_dir="/nfs/10.232.64.52/nvme3/kangyu/saved_models/qwen_chat_estate_epoch1_RM_ultrafeedback_womargin_RS_it_data_each_1w_20iters"
mkdir -p ${output_dir}

instruction_file="$BELLE_PATH/data/it_data_sample_1w_each.json"

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=2048

accelerate launch \
    --config_file $BELLE_PATH/train/configs/accelerate_config_rm.yaml \
    --num_processes $gpus \
    "$BELLE_PATH/train/src/entry_point/rs_train.py" \
    --deepspeed /nfs/a100-80G-17/kangyu/consistency_hallucinations/llm-playground/configs/ds_stage3_bf16.json \
    --model_name $model_name_or_path \
    --reward_model_name $reward_model_name_or_path \
    --instruction_data $instruction_file \
    --cache_dir $cache_dir \
    --report_to "tensorboard" \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --num_rs_iteration 20 \
    --collection_strategy "local" \
    --rs_batch_size 1024 \
    --top_reward_percentage 0.1 \
    --output_min_length 64 \
    --output_max_length 1024 \
    --output_reward_path $output_dir/reward_results/ \
    --num_train_epochs 1 \
    --max_seq_length $cutoff_len \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --load_in_8bit False \
    --load_in_4bit False \
    --output_dir $output_dir \
    --use_llama_model False \
    --use_rm_llama False \
    --lr_scheduler_type "cosine"
