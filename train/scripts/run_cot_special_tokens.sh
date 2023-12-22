export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpus=8

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=...

model_name_or_path="/nfs/a100-80G-16/kangyu/saved_models/Llama-2-13b-chat_add-2-cot-tokens"
tokenizer_name_or_path="/nfs/a100-80G-16/kangyu/saved_models/tokenizer_Llama-2-13b-chat_add-2-cot-tokens"

output_dir="/nfs/10.232.64.3/nvme2/kangyu/saved_models/Llama-2-13b-single-cot-token"
mkdir -p ${output_dir}

train_file=/nfs/a100-80G-17/kangyu/consistency_hallucinations/trytry/implicit_cot/data/gsm8k/train_cot-special-tokens_train.json
test_file=/nfs/a100-80G-17/kangyu/consistency_hallucinations/trytry/implicit_cot/data/gsm8k/train_cot-special-tokens_test.json

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}

cutoff_len=2048

accelerate launch \
    --config_file $BELLE_PATH/train/configs/accelerate_config_rm.yaml \
    --num_processes $gpus \
    "$BELLE_PATH/train/src/entry_point/train_cot_special_tokens.py" \
    --model_name $model_name_or_path \
    --tokenizer_name $tokenizer_name_or_path \
    --train_data $train_file \
    --eval_data $test_file \
    --cache_dir $cache_dir \
    --report_to "tensorboard" \
    --logging_steps 1 \
    --save_total_limit 5 \
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
