# coding=utf-8
import os
import sys
import json
from datetime import timedelta
from typing import Optional
from dataclasses import dataclass, field

from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
import torch
import torch.distributed as dist
from transformers import (
    HfArgumentParser,
    LlamaTokenizer
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import logging

from src.llama2_for_token_ppl import LlamaForCausalLM


kwargs_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))  # increase NCCL timeout to 10 hours
accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
tqdm.pandas()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def print_rank_0(msg, log_file):
    if accelerator.is_main_process:
        with open(log_file, "a") as f:
            print(msg)
            f.write(msg + "\n")

@dataclass
class MyArguments:
    """
    Define a class RewardPipeArguments to configure reward pipeline.
    """
    output_dir: str = field(
        default="./outputs/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    model_name: Optional[str] = field(
        default="Qwen/Qwen-14B-Chat", metadata={"help": "the name of model under training."}
    )
    train_data: str = field(default="", metadata={"help": "training dataset"})
    cache_dir: str = field(default="", metadata={"help": "cache dir"})
    debug: Optional[bool] = field(default=False, metadata={"help": "debug with toy dataset"})
    start_n_sub: Optional[int] = field(
        default=0, metadata={"help": "start_n_sub"}
    )
    inference_batch_size_per_device: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "every device will infer {inference_batch_size_per_device}"
                " samples in parallel. The inferred results will be concatenaed"
                " with inputs and attach a reward."
            ),
        },
    )


instruction_prompt = "[INST] " + "{}" + " [/INST]"
cot_prompt = "{}"
answer_prompt = "The answer is: {}"


def main():
    parser = HfArgumentParser(MyArguments)
    my_args = parser.parse_args_into_dataclasses()[0]

    os.makedirs(my_args.output_dir, exist_ok=True)

    log_file = os.path.join(my_args.output_dir, "log.txt")
    local_rank = accelerator.local_process_index
    global_rank = accelerator.process_index

    # Load the tokenizer for model under training
    tokenizer = LlamaTokenizer.from_pretrained(my_args.model_name)
    tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<unk>",
        }
    )
    tokenizer.padding_side = "left"

    print_rank_0(
        f"eos token: {tokenizer.eos_token}, "
        f"eos token id: {tokenizer.eos_token_id}, "
        f"pad token: {tokenizer.pad_token}, "
        f"pad token id: {tokenizer.pad_token_id}",
        log_file,
    )

    with accelerator.main_process_first():
        train_dataset = load_dataset(
            "json", data_files=my_args.train_data, cache_dir=my_args.cache_dir
        )["train"]

        if my_args.debug:
            train_dataset = train_dataset.select(range(200))

        sub_datasets = []
        sub_dataset_size = int(len(train_dataset) // 4)
        if sub_dataset_size * 4 < len(train_dataset):
            sub_dataset_size += 1
        for i in range(4):
            sub_dataset = train_dataset.select( np.arange( i * sub_dataset_size, min((i + 1) * sub_dataset_size, len(train_dataset)) ) )
            sub_datasets.append(sub_dataset)

    print_rank_0(
        "train_dataset size = {}".format(len(train_dataset)),
        log_file,
    )

    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print_rank_0(
        "world_size = {}".format(world_size),
        log_file,
    )

    device_map = None
    quantization_config = None

    # load model under training
    model = LlamaForCausalLM.from_pretrained(
        my_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(accelerator.device)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    # split dataset into sub-dataset for robust
    for n_sub, sub_train_dataset in enumerate(sub_datasets):

        if n_sub < my_args.start_n_sub:
            continue

        print_rank_0("*"*20 + "Start {} sub dataset".format(n_sub) + "*"*20, log_file)
        print_rank_0("sub dataset size: {}".format(len(sub_train_dataset)), log_file)

        # split dataset into each rank
        shard_size = int(len(sub_train_dataset) / world_size)
        if shard_size * world_size < len(sub_train_dataset):
            shard_size += 1
        shard_dataset = sub_train_dataset.select( np.arange( global_rank * shard_size, min((global_rank + 1) * shard_size, len(sub_train_dataset)) ) )

        print("global_rank:", global_rank, "shard_dataset nums:", len(shard_dataset))

        output_dataset = []
        for one in tqdm(shard_dataset, total=len(shard_dataset)):
            instruction = instruction_prompt.format(one['question'])
            cot = cot_prompt.format(one['cot'])
            answer = answer_prompt.format(one['final_answer'])

            instruction_id = tokenizer.encode(instruction, add_special_tokens=False)
            cot_id = tokenizer.encode(cot, add_special_tokens=False)
            answer_id = tokenizer.encode(answer, add_special_tokens=False)

            assert answer_id[:5] == [450, 1234, 338, 29901, 29871]

            input_id = [tokenizer.bos_token_id] + instruction_id + cot_id + answer_id + [tokenizer.eos_token_id]
            input_ids = torch.tensor([input_id]).to(accelerator.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)

            loss = outputs.loss.detach().cpu().tolist()

            answer_loss = loss[-1 - len(answer_id): -1]
            assert len(answer_loss) == len(answer_id)

            # 通用前缀部分也不包括进去
            answer_loss = answer_loss[5:]
            pivot_loss = sum(answer_loss) / len(answer_loss)

            # 开始摘token
            while True:
                cot_tokens = cot.split()
                best_i = None
                best_loss = None
                best_delta_loss = 99999

                # 这个部分开始考虑并行
                input_ids = []
                indexs = []
                for i in range(len(cot_tokens)):
                    _cot = cot_tokens[:i] + cot_tokens[i + 1:]
                    _cot = ' '.join(_cot)
                    _cot_id = tokenizer.encode(_cot, add_special_tokens=False)
                    assert len(_cot_id) < len(cot_id)

                    input_id = [tokenizer.bos_token_id] + instruction_id + _cot_id + answer_id + [
                        tokenizer.eos_token_id]

                    input_ids.append(input_id)
                    indexs.append(i)
                    if len(input_ids) % my_args.inference_batch_size_per_device == 0 or i == len(cot_tokens)-1:
                        inputs = tokenizer.pad({
                            "input_ids": input_ids
                        }, return_tensors='pt').to(accelerator.device)

                        with torch.no_grad():
                            outputs = model(**inputs, labels=inputs['input_ids'])

                        losses = outputs.loss.detach().cpu()
                        batch_size, seq_len = inputs['input_ids'].shape
                        losses = losses.reshape([batch_size, seq_len-1]).tolist()

                        for j in range(len(input_ids)):
                            input_id = input_ids[j]
                            index = indexs[j]
                            loss = losses[j][:len(input_id)-1]

                            answer_loss = loss[-1 - len(answer_id): -1]
                            assert len(answer_loss) == len(answer_id)

                            # 通用前缀部分也不包括进去
                            answer_loss = answer_loss[5:]
                            cur_loss = sum(answer_loss) / len(answer_loss)

                            delta_loss = cur_loss - pivot_loss  # 越小越好，loss降越多越好。

                            if delta_loss < best_delta_loss:
                                best_delta_loss = delta_loss
                                best_i = index
                                best_loss = cur_loss

                        input_ids = []
                        indexs = []

                # 没有需要摘掉的token了
                if best_delta_loss >= 0:
                    break

                cot = cot_tokens[:best_i] + cot_tokens[best_i + 1:]
                cot = ' '.join(cot)
                pivot_loss = best_loss

            new_one = dict(one)
            new_one['new_cot'] = cot

            output_dataset.append(new_one)

        print("rank {} done, we get {} outputs for {} samples.".format(global_rank, len(output_dataset), len(shard_dataset)))

        assert len(output_dataset) == len(shard_dataset)

        #####################################################
        accelerator.wait_for_everyone()
        #####################################################

        # All-gather
        all_process_data = [{}] * world_size
        dist.all_gather_object(all_process_data, output_dataset)

        gathered_data = []
        for i in range(world_size):
            gathered_data.extend(all_process_data[i])

        print_rank_0("gathered_data size: {}".format(len(gathered_data)), log_file)

        if accelerator.is_main_process:
            with open(os.path.join(my_args.output_dir, "output_sub_{}.json".format(n_sub)), 'w', encoding='utf8') as f:
                json.dump(gathered_data, f, ensure_ascii=False)

    return


if __name__ == "__main__":
    main()

