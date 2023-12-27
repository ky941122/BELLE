export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpus=8

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

model_name_or_path="/nfs/a100-80G-15/xytian/myProjects/AI_NLP_GM/pretrained_models/Llama-2-13b-chat-hf"

output_dir="$BELLE_PATH/results/debug_ppl_from_raw_model"
mkdir -p ${output_dir}

train_file="/nfs/a100-80G-17/kangyu/consistency_hallucinations/trytry/implicit_cot/data/gsm8k/train_cot-special-tokens_train.json"

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}

accelerate launch \
    --config_file $BELLE_PATH/train/configs/accelerate_config_rs.yaml \
    --num_processes $gpus \
    "$BELLE_PATH/train/src/entry_point/cot_compression/compression_by_ppl.py" \
    --output_dir $output_dir \
    --model_name $model_name_or_path \
    --train_data $train_file \
    --cache_dir $cache_dir \
    --debug True \
    --start_n_sub 3 \
    --inference_batch_size_per_device 10
