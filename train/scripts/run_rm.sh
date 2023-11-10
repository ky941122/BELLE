export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpus=8

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=...

model_name_or_path="/nfs/10.232.64.52/nvme2/xhsun/saved_models/chatmodel-ft-roleinjection_qwen"
output_dir="/nfs/10.232.64.52/nvme3/kangyu/saved_models/chatmodel-ft-roleinjection_qwen_translated_ultra_shp_no-margin"
mkdir -p ${output_dir}

train_file=$BELLE_PATH/data/translated_preference_data_ultra_shp_by_gpt35.json
validation_file=$BELLE_PATH/data/translated_preference_data_ultra_shp_by_gpt35.json
cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=512

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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 1 \
    --seq_length $cutoff_len \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing False \
    --load_in_8bit False \
    --load_in_4bit False \
    --use_lora False \
    --trust_remote_code True \
    --output_dir $output_dir \
    --use_llama False \
    --lr_scheduler_type "cosine"
