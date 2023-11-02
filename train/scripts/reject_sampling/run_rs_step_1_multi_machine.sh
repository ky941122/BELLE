export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

WORKER_GPU=$1
WORKER_NUM=$2
RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5

model_name_or_path="/nfs/10.232.64.52/nvme4/xhsun/saved_models/qwen_chat_estate_epoch1/"
reward_model_name_or_path="/nfs/10.232.64.52/nvme3/kangyu/saved_models/qwen_chat_estate_epoch1_ultrafeedback_womargin/checkpoint-1986"

output_dir="/nfs/10.232.64.52/nvme3/kangyu/saved_models/qwen_chat_estate_epoch1_RM_ultrafeedback_womargin_RS_it_data_each_1w_iter-0"
mkdir -p ${output_dir}

instruction_file="$BELLE_PATH/data/it_data_sample_1w_each.json"

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}

accelerate launch \
    --config_file $BELLE_PATH/train/configs/accelerate_config_rs_multi_machine.yaml \
    --num_processes $WORKER_GPU \
    --num_machines $WORKER_NUM \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    "$BELLE_PATH/train/src/entry_point/reject_sampling/step_1_sampling.py" \
    --model_name $model_name_or_path \
    --reward_model_name $reward_model_name_or_path \
    --instruction_data $instruction_file \
    --cache_dir $cache_dir \
    --inference_batch_size_per_device 12 \
    --max_prompt_length 1024 \
    --n_best_nums 10 \
    --output_min_length 128 \
    --output_max_length 1024 \
    --output_reward_path $output_dir/reward_results/ \
    --max_seq_length 2048 \
    --load_in_8bit False \
    --load_in_4bit False \
    --output_dir $output_dir \
    --use_llama_model False \
    --use_rm_llama False \
    --start_n_sub 3 \
    --debug False
