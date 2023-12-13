export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpus=8

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH=$BELLE_PATH/train

export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=allow

model_name_or_path=...
output_dir="$BELLE_PATH/saved_models/try_ppo"
mkdir -p ${output_dir}

train_file=/nfs/a100-80G-15/xytian/myProjects/AI_NLP_GM/data/ppo_test/ppo_test.train.json
cache_dir=hf_cache_dir
mkdir -p ${cache_dir}

accelerate launch \
    --config_file $BELLE_PATH/train/configs/accelerate_config_ppo.yaml \
    --num_processes $gpus \
    --main_process_port 29600 \
    "$BELLE_PATH/train/src/entry_point/ppo_train.py" \
    --model_name "/nfs/a100-80G-15/xytian/myProjects/AI_NLP_GM/pretrained_models/Llama-2-7b-chat-hf" \
    --reward_model_name "/nfs/a100-80G-15/xytian/myProjects/AI_NLP_GM/pretrained_models/Llama-2-7b-chat-hf" \
    --train_data $train_file \
    --cache_dir $cache_dir \
    --adafactor False \
    --save_freq 100 \
    --output_max_length 128 \
    --batch_size 32 \
    --mini_batch_size 2 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --ppo_epochs 2 \
    --data_epochs 1 \
    --seed 42 \
    --learning_rate 1.4e-5 \
    --early_stopping True \
    --do_sample True \
    --output_dir $output_dir \
    --log_with "tensorboard" \
    --logging_dir "$output_dir/logs" \
    --use_llama True \
    --reward_model_use_llama True \
    --use_lora False \
    --input_length 512 
