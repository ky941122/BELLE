export CUDA_VISIBLE_DEVICES="3,4,5,6,7"
gpus=5

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=...

model_name_or_path="/nfs/10.232.64.52/nvme4/xhsun/saved_models/qwen_chat_estate_epoch1/"
output_dir="/nfs/10.232.64.52/nvme3/kangyu/saved_models/qwen_chat_estate_epoch1_ultrafeedback_womargin"
mkdir -p ${output_dir}

train_file=/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE/data/UltraFeedback_QWenFormat_WithoutMargin.json
validation_file=/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE/data/UltraFeedback_QWenFormat_WithoutMargin.json
cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=2048

accelerate launch \
    --config_file $BELLE_PATH/train/configs/accelerate_config_rm.yaml \
    --num_processes $gpus \
    "$BELLE_PATH/train/src/entry_point/rm_train.py" \
    --model_name $model_name_or_path \
    --train_data $train_file \
    --eval_data $validation_file \
    --cache_dir $cache_dir \
    --report_to "tensorboard" \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 1 \
    --seq_length $cutoff_len \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --load_in_8bit False \
    --load_in_4bit False \
    --use_lora False \
    --trust_remote_code True \
    --output_dir $output_dir \
    --use_llama False \
    --lr_scheduler_type "cosine"
