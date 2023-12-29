export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

WORKER_GPU=$1
WORKER_NUM=$2
RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5

model_name_or_path="/nfs/a100-80G-14/kangyu/saved_models/Llama-2-13b_GSM8K_2"

output_dir="$BELLE_PATH/results/Llama-2-13b_GSM8K-2_consistency_round-1"
mkdir -p ${output_dir}

instruction_file="/nfs/a100-80G-17/kangyu/consistency_hallucinations/trytry/cot_compression/data/gsm8k_test_split_2.json"

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}

accelerate launch \
    --config_file $BELLE_PATH/train/configs/accelerate_config_rs_multi_machine.yaml \
    --num_processes $WORKER_GPU \
    --num_machines $WORKER_NUM \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    "$BELLE_PATH/train/src/entry_point/cot_compression/self_consistency.py" \
    --model_name $model_name_or_path \
    --instruction_data $instruction_file \
    --cache_dir $cache_dir \
    --inference_batch_size_per_device 10 \
    --max_prompt_length 2048 \
    --n_best_nums 20 \
    --load_in_8bit False \
    --load_in_4bit False \
    --output_dir $output_dir \
    --start_n_sub 0 \
    --debug False
