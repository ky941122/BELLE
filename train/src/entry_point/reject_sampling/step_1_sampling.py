# coding=utf-8
import os
import sys
import json
from datetime import timedelta
from functools import partial

from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    LlamaTokenizer
)
from trl.core import LengthSampler
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import logging
from multiprocessing import cpu_count

from src.entry_point.reject_sampling.configs import RejectSamplingArguments
from src.models.qwen.qwen_generation_utils import make_context

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

def preprocess_query_qwen(tokenizer, source, system_message = "You are a helpful assistant."):
    raw_text, input_id = make_context(
        tokenizer,
        source,
        history=None,
        system=system_message
    )

    return raw_text, input_id


# Tokenize inputs
# Adapt this section to your needs for custom datasets
def preprocess_query_function(tokenizer, use_qwen, examples):
    new_examples = {
        "input_ids": [],
        "raw_instruction": [],
        "processed_instruction": [],
        "target_response": []
    }
    for instruction, response in zip(examples["instruction"], examples["response"]):
        if use_qwen:
            processed_instruction, input_ids = preprocess_query_qwen(tokenizer, instruction)
        else:
            raise NotImplementedError()

        new_examples["input_ids"].append(input_ids)
        new_examples["raw_instruction"].append(instruction)
        new_examples["processed_instruction"].append(processed_instruction)
        new_examples["target_response"].append(response)

    return new_examples


def main():
    parser = HfArgumentParser(RejectSamplingArguments)
    rs_args = parser.parse_args_into_dataclasses()[0]

    os.makedirs(rs_args.output_dir, exist_ok=True)

    log_file = os.path.join(rs_args.output_dir, "step_1_log.txt")
    local_rank = accelerator.local_process_index
    global_rank = accelerator.process_index

    # Load the tokenizer for model under training
    if rs_args.use_llama_model:
        tokenizer = LlamaTokenizer.from_pretrained(rs_args.model_name)
        tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<unk>",
            }
        )
    elif "qwen" in rs_args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(rs_args.model_name, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(rs_args.model_name)
        tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})

    tokenizer.padding_side = "left"

    print_rank_0(
        f"eos token: {tokenizer.eos_token}, "
        f"eos token id: {tokenizer.eos_token_id}, "
        f"pad token: {tokenizer.pad_token}, "
        f"pad token id: {tokenizer.pad_token_id}",
        log_file,
    )

    with accelerator.main_process_first():
        instruction_dataset = load_dataset(
            "json", data_files=rs_args.instruction_data, cache_dir=rs_args.cache_dir
        )["train"]
        instruction_dataset = instruction_dataset.map(
            partial(preprocess_query_function, tokenizer, "qwen" in rs_args.model_name.lower()),
            batched=True,
            num_proc=max(cpu_count() // 2, 1)
        )
        instruction_dataset = instruction_dataset.filter(
            lambda x: len(x["input_ids"]) <= rs_args.max_prompt_length
        )

        if rs_args.debug:
            instruction_dataset = instruction_dataset.select(range(200))

        sub_datasets = []
        sub_dataset_size = int(len(instruction_dataset) // 4)
        if sub_dataset_size * 4 < len(instruction_dataset):
            sub_dataset_size += 1
        for i in range(4):
            sub_dataset = instruction_dataset.select( np.arange( i * sub_dataset_size, min((i + 1) * sub_dataset_size, len(instruction_dataset)) ) )
            sub_datasets.append(sub_dataset)

    print_rank_0(
        "instruction_dataset size = {}".format(len(instruction_dataset)),
        log_file,
    )

    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print_rank_0(
        "world_size = {}".format(world_size),
        log_file,
    )

    if rs_args.load_in_8bit and rs_args.load_in_4bit:
        raise ValueError(
            "You can't load the model in 8 bits and 4 bits at the same time"
        )
    elif rs_args.load_in_8bit or rs_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=rs_args.load_in_8bit, load_in_4bit=rs_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": local_rank}
    else:
        device_map = None
        quantization_config = None

    # load model under training
    model = AutoModelForCausalLM.from_pretrained(
        rs_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(accelerator.device)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    output_length_sampler = LengthSampler(rs_args.output_min_length, rs_args.output_max_length)

    generation_kwargs = {
        "min_length": 1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": 1.0,
    }

    # split dataset into sub-dataset for robust
    for n_sub, sub_instruction_dataset in enumerate(sub_datasets):
        print_rank_0("*"*20 + "Start {} sub dataset".format(n_sub) + "*"*20, log_file)
        print_rank_0("sub dataset size: {}".format(len(sub_instruction_dataset)), log_file)

        # split dataset into each rank
        shard_size = int(len(sub_instruction_dataset) / world_size)
        if shard_size * world_size < len(sub_instruction_dataset):
            shard_size += 1
        shard_dataset = sub_instruction_dataset.select( np.arange( global_rank * shard_size, min((global_rank + 1) * shard_size, len(sub_instruction_dataset)) ) )

        print("global_rank:", global_rank, "shard_dataset nums:", len(shard_dataset))

        all_output_ids = []
        all_input_ids = []

        input_ids = []
        for data in tqdm(shard_dataset, total=len(shard_dataset)):
            # one instruction we need to sample N times
            N_input_ids = [data['input_ids'] for _ in range(rs_args.n_best_nums)]

            for one_input_ids in N_input_ids:
                input_ids.append(one_input_ids)

                if len(input_ids) % rs_args.inference_batch_size_per_device == 0:
                    gen_len = output_length_sampler()
                    generation_kwargs["max_new_tokens"] = gen_len

                    inputs = tokenizer.pad({
                        "input_ids": input_ids
                    }, return_tensors='pt').to(accelerator.device)

                    with torch.no_grad():
                        outputs = model.generate(**inputs, **generation_kwargs)

                    outputs = outputs.detach().cpu().numpy().tolist()
                    inputs = inputs['input_ids'].detach().cpu().numpy().tolist()

                    all_output_ids += outputs
                    all_input_ids += inputs

                    input_ids = []

        if len(input_ids) > 0:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len

            inputs = tokenizer.pad({
                "input_ids": input_ids
            }, return_tensors='pt').to(accelerator.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)

            outputs = outputs.detach().cpu().numpy().tolist()
            inputs = inputs['input_ids'].detach().cpu().numpy().tolist()

            all_output_ids += outputs
            all_input_ids += inputs

        print("rank {} done with sampling, we get {} outputs for {} instruction.".format(global_rank, len(all_output_ids), len(shard_dataset)))

        assert len(all_output_ids) == len(all_input_ids) == len(shard_dataset) * rs_args.n_best_nums

        # do some post-process
        output_dataset = []
        for i, data in enumerate(shard_dataset):
            n_sample_input_ids = all_input_ids[i * rs_args.n_best_nums : (i+1) * rs_args.n_best_nums]
            n_sample_output_ids = all_output_ids[i * rs_args.n_best_nums: (i + 1) * rs_args.n_best_nums]

            new_data = dict(data)
            generated_texts = []
            for j in range(len(n_sample_input_ids)):
                assert data['input_ids'] == n_sample_input_ids[j][-len(data['input_ids']):]
                prompt_length = len(n_sample_input_ids[j])
                generated_ids = n_sample_output_ids[j][prompt_length:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                generated_texts.append(generated_text)

            new_data['model_generated_texts'] = generated_texts

            output_dataset.append(new_data)

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
            with open(os.path.join(rs_args.output_dir, "step_1_output_sub_{}.json".format(n_sub)), 'w', encoding='utf8') as f:
                json.dump(gathered_data, f, ensure_ascii=False)

    return


if __name__ == "__main__":
    main()


