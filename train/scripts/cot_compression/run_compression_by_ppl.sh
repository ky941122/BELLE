export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpus=8

BELLE_PATH="/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE"
export PYTHONPATH="$BELLE_PATH/train"

model_name_or_path="/nfs/172.17.3.40/nvme1/a100-test-1node-torchimg-idc/bella/xhsun/llama2-models/Llama-2-13b-chat-hf"

output_dir="$BELLE_PATH/results/ppl_from_raw_model"
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
    --debug False \
    --start_n_sub 0 \
    --inference_batch_size_per_device 10
