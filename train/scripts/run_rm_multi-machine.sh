export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

WORKER_GPU=$1
WORKER_NUM=$2
RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5

KE_ROOT_PATH = "$BELLE_PATH/saved_models"
ALI_ROOT_PATH = "/nfs/10.232.64.52/nvme3/kangyu/saved_models"

model_name_or_path="/nfs/a100-80G-18/xunxianghui/gitrepositories/Chathome-14B-roleSFT/chatmodel-ft-roleinjection_qwen"
output_dir="$KE_ROOT_PATH/chatmodel-ft-roleinjection_qwen_translated_ultra_shp_no-margin_multi-machine"
mkdir -p ${output_dir}

train_file=$BELLE_PATH/data/translated_preference_data_ultra_shp_by_gpt35.json
validation_file=$BELLE_PATH/data/translated_preference_data_ultra_shp_by_gpt35.json

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}

cutoff_len=2048

accelerate launch \
    --config_file $BELLE_PATH/train/configs/accelerate_config_rm_multi_machine.yaml \
    --num_processes $WORKER_GPU \
    --num_machines $WORKER_NUM \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
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
    --debug False \
    --lr_scheduler_type "cosine"
