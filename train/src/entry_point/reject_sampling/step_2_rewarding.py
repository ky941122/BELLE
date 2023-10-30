# coding=utf-8
import os
import sys
import json
from functools import partial

from accelerate import Accelerator
import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
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
from src.models.qwen.modeling_qwen import QWenForSequenceClassification
from src.models.qwen.qwen_generation_utils import make_context

accelerator = Accelerator()
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


def preprocess_qwen(tokenizer, source, system_message = "You are a helpful assistant."):
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant", "human": "<|im_start|>user"}
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens

    input_id = []
    input_id += system
    for j, sentence in enumerate(source):
        role = roles[sentence["from"].lower()]
        _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id

    return input_id


def preprocess_function(tokenizer, max_length, use_qwen, examples):
    new_examples = {
        "input_ids_for_rm": [],
        "model_generated_texts": [],
    }
    if use_qwen:
        for raw_instruction, generated_response_list in zip(examples["raw_instruction"], examples["model_generated_texts"]):
            # one data
            res_set = set(generated_response_list)
            res_set = list(res_set)

            input_ids_for_rm = []
            response_for_rm = []
            for response in res_set:
                # one response
                source = [
                    {"from": "user", "value": raw_instruction},
                    {"from": "assistant", "value": response},
                ]
                input_ids = preprocess_qwen(tokenizer, source)

                if len(input_ids) <= max_length:
                    input_ids_for_rm.append(input_ids)
                    response_for_rm.append(response)

            assert len(response_for_rm) == len(input_ids_for_rm)

            new_examples['input_ids_for_rm'].append(input_ids_for_rm)
            new_examples['model_generated_texts'].append(response_for_rm)
    else:
        raise NotImplementedError()

    return new_examples


def main():
    parser = HfArgumentParser(RejectSamplingArguments)
    rs_args = parser.parse_args_into_dataclasses()[0]

    os.makedirs(rs_args.output_dir, exist_ok=True)

    log_file = os.path.join(rs_args.output_dir, "step_2_log.txt")
    local_rank = accelerator.local_process_index
    global_rank = accelerator.process_index

    # Load the tokenizer for reward model
    if rs_args.use_rm_llama:
        rm_tokenizer = LlamaTokenizer.from_pretrained(rs_args.reward_model_name)
        rm_tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<unk>",
            }
        )
    elif "qwen" in rs_args.reward_model_name.lower():
        rm_tokenizer = AutoTokenizer.from_pretrained(rs_args.reward_model_name, trust_remote_code=True)
        rm_tokenizer.pad_token_id = rm_tokenizer.eod_id
    else:
        rm_tokenizer = AutoTokenizer.from_pretrained(rs_args.reward_model_name)
        rm_tokenizer.add_special_tokens({"pad_token": rm_tokenizer.unk_token})

    rm_tokenizer.padding_side = "left"

    print_rank_0(
        f"rm unk token: {rm_tokenizer.unk_token}, "
        f"rm unk token id: {rm_tokenizer.unk_token_id}, "
        f"rm pad token: {rm_tokenizer.pad_token}, "
        f"rm pad token id: {rm_tokenizer.pad_token_id}",
        log_file,
    )

    with accelerator.main_process_first():
        reward_datasets = []
        for i in range(4):
            _reward_dataset = load_dataset(
                "json",
                data_files=os.path.join(rs_args.output_dir, "step_1_output_sub_{}.json".format(i)),
                cache_dir=rs_args.cache_dir
            )["train"]
            _reward_dataset = _reward_dataset.map(
                partial(preprocess_function, rm_tokenizer, rs_args.max_seq_length, "qwen" in rs_args.reward_model_name.lower()),
                batched=True,
                num_proc=max(cpu_count() // 2, 1)
            )
            _reward_dataset = _reward_dataset.filter(
                lambda x: len(x["input_ids"]) > 0
            )

            if rs_args.debug:
                _reward_dataset = _reward_dataset.select(range(50))

            reward_datasets.append(_reward_dataset)

    print_rank_0(
        "reward_dataset size = {}".format( sum([len(reward_dataset) for reward_dataset in reward_datasets]) ),
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

    # load reward model
    if "qwen" in rs_args.reward_model_name.lower():
        reward_model = QWenForSequenceClassification.from_pretrained(
            rs_args.reward_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            num_labels=1,
        )
    else:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            rs_args.reward_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            num_labels=1,
        )
    reward_model.config.pad_token_id = rm_tokenizer.pad_token_id
    reward_model.to(accelerator.device)
    reward_model.gradient_checkpointing_disable()
    reward_model.config.use_cache = True
    reward_model.eval()

    # split dataset into sub-dataset for robust
    for n_sub, reward_dataset in enumerate(reward_datasets):
        print_rank_0("*"*20 + "Start {} sub dataset".format(n_sub) + "*"*20, log_file)
        print_rank_0("sub dataset size: {}".format(len(reward_dataset)), log_file)

        # split dataset into each rank
        shard_size = int(len(reward_dataset) / world_size)
        if shard_size * world_size < len(reward_dataset):
            shard_size += 1
        shard_dataset = reward_dataset.select( np.arange( global_rank * shard_size, min((global_rank + 1) * shard_size, len(reward_dataset)) ) )

        print("global_rank:", global_rank, "shard_dataset nums:", len(shard_dataset))

        all_rewards = []

        input_ids = []
        for data in tqdm(shard_dataset, total=len(shard_dataset)):
            # one instruction we need to get N rewards
            N_input_ids = data['input_ids_for_rm']

            for one_input_ids in N_input_ids:
                input_ids.append(one_input_ids)

                if len(input_ids) % rs_args.inference_batch_size_per_device == 0:
                    inputs = rm_tokenizer.pad({
                        "input_ids": input_ids
                    }, return_tensors='pt').to(accelerator.device)

                    with torch.no_grad():
                        outputs = reward_model(**inputs)

                    outputs = outputs.logits.cpu().detach()
                    rewards = []
                    for out in outputs:
                        rewards.append(float(out[0]))

                    all_rewards += rewards

                    input_ids = []

        if len(input_ids) > 0:
            inputs = rm_tokenizer.pad({
                "input_ids": input_ids
            }, return_tensors='pt').to(accelerator.device)

            with torch.no_grad():
                outputs = reward_model(**inputs)

            outputs = outputs.logits.cpu().detach()
            rewards = []
            for out in outputs:
                rewards.append(float(out[0]))

            all_rewards += rewards

        print("rank {} done with rewarding, we get {} rewards for {} instruction.".format(global_rank, len(all_rewards), len(shard_dataset)))

        # do some post-process, and find the n-best
        output_dataset = []
        cur_index = 0
        for data in shard_dataset:
            generations = data["model_generated_texts"]
            generation_nums = len(generations)
            end_index = cur_index + generation_nums
            rewards = all_rewards[cur_index : end_index]

            new_data = dict(data)
            idx_to_record = np.argmax(rewards)
            new_data["rewards"] = rewards
            new_data["best_reward"] = rewards[idx_to_record]
            new_data["best_model_generated_text"] = generations[idx_to_record]

            output_dataset.append(new_data)

            cur_index = end_index

        assert cur_index == len(all_rewards)

        # All-gather
        all_process_data = [{}] * world_size
        dist.all_gather_object(all_process_data, output_dataset)

        gathered_data = []
        for i in range(world_size):
            gathered_data.extend(all_process_data[i])

        print_rank_0("gathered_data size: {}".format(len(gathered_data)), log_file)

        if accelerator.is_main_process:
            with open(os.path.join(rs_args.output_dir, "step_2_output_sub_{}.json".format(n_sub)), 'w',
                      encoding='utf8') as f:
                json.dump(gathered_data, f, ensure_ascii=False)

    return


if __name__ == "__main__":
    main()

