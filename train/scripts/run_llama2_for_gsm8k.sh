export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpus=8

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=...

model_name_or_path="/nfs/172.17.3.40/nvme1/a100-test-1node-torchimg-idc/bella/xhsun/llama2-models/Llama-2-13b-chat-hf"

output_dir="/nfs/172.17.1.38/nvme4/kangyu/saved_models/Llama-2-13b_raw-cot_2000-samples"
mkdir -p ${output_dir}

train_file=/nfs/a100-80G-17/kangyu/consistency_hallucinations/trytry/cot_compression/data/compression_results_by_ppl_from_raw_model.json
test_file=/nfs/a100-80G-17/kangyu/consistency_hallucinations/trytry/cot_compression/data/compression_results_by_ppl_from_raw_model.json

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}

cutoff_len=2048

accelerate launch \
    --config_file $BELLE_PATH/train/configs/accelerate_config_rm.yaml \
    --num_processes $gpus \
    "$BELLE_PATH/train/src/entry_point/train_llama2_for_gsm8k.py" \
    --model_name $model_name_or_path \
    --train_data $train_file \
    --eval_data $test_file \
    --cache_dir $cache_dir \
    --report_to "tensorboard" \
    --logging_steps 1 \
    --save_total_limit 3 \
    --learning_rate 8e-6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 2 \
    --max_seq_length $cutoff_len \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --load_in_8bit False \
    --load_in_4bit False \
    --output_dir $output_dir \
    --debug False \
    --lr_scheduler_type "cosine"
