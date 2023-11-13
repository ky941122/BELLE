export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

WORKER_GPU=$1
WORKER_NUM=$2
RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5

model_name_or_path="/nfs/10.232.64.52/nvme2/xhsun/saved_models/chatmodel-ft-roleinjection_qwen"
reward_model_name_or_path=""

output_dir="/nfs/10.232.64.52/nvme3/kangyu/saved_models/chatmodel-ft-roleinjection_qwen_RM_ultra_shp_no-margin_RS_jiazhuang_1w_20231113"
mkdir -p ${output_dir}

instruction_file="$BELLE_PATH/data/jiazhuang_gen_by_gpt35_20231113.json"

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
    --start_n_sub 0 \
    --debug False
