# coding=utf-8
import os
import sys
import time
import math
import json
from typing import Optional
from dataclasses import dataclass, field
from functools import partial

from accelerate import Accelerator
import torch
import torch.distributed as dist
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    HfArgumentParser,
    LlamaTokenizer,
    TrainingArguments
)
from transformers.trainer_pt_utils import LabelSmoother
from trl.core import LengthSampler
from datasets import load_dataset, Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from multiprocessing import cpu_count

from src.models.qwen.modeling_qwen import QWenForSequenceClassification
from src.models.qwen.qwen_generation_utils import make_context

import transformers
from packaging import version
if version.parse(transformers.__version__) <= version.parse("4.30.2"):
    from src.trainer import MyTrainer as Trainer
else:
    from transformers import Trainer

accelerator = Accelerator()


#
#
# print("*" * 100)
# print(accelerator.__dict__)
# print("*" * 100)
#


tqdm.pandas()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

INF = 888888888
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# for plotting
reward_plot_seq = []
train_reward_plot_seq = []


def print_rank_0(msg, log_file):
    if accelerator.is_main_process:
        with open(log_file, "a") as f:
            print(msg)
            f.write(msg + "\n")


@dataclass
class RejectSamplingArguments:
    """
    Define a class RewardPipeArguments to configure reward pipeline.
    """
    output_reward_path: Optional[str] = field(
        default="tmp/output/",
        metadata={
            "help": "The path of output rewards."
        }
    )
    output_min_length: Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "minimum length of the output token sequence generated from"
                " model given an input."
            ),
        },
    )
    output_max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "maximum length of the output token sequence generated from"
                " model given an output."
            ),
        },
    )
    num_rs_iteration: Optional[int] = field(
        default=20,
        metadata={
            "help": "number of iterations of the reject sampling."
        },
    )
    rs_batch_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "only select {rs_batch_size} samples each time for STF training."
            )
        },
    )
    top_reward_percentage: Optional[float] = field(
        default=0.2,
        metadata={
            "help": (
                "only top {top_reward_percentage} samples in the rs batch,"
                " (in terms of rewards), will be used for SFT the model."
            ),
        },
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
    collection_strategy: Optional[str] = field(
        default="local",
        metadata={
            "help": (
                "{collection_strategy} is either global or local"
                " global means that we rank the samples globally regardless of the prompts"
                " local means that we only rank the samples with the same prompt"
            ),
        },
    )
    use_llama_model: Optional[bool] = field(default=False, metadata={"help": "train a model in llama series"})
    use_rm_llama: Optional[bool] = field(default=False, metadata={"help": "use reward model in llama series"})
    model_name: Optional[str] = field(
        default="Qwen/Qwen-14B-Chat", metadata={"help": "the name of model under training."}
    )
    reward_model_name: Optional[str] = field(
        default="Qwen/Qwen-14B-Chat", metadata={"help": "the name of reward model."}
    )
    instruction_data: str = field(default="", metadata={"help": "instruction data path for n-best sampling"})
    cache_dir: str = field(default="", metadata={"help": "cache dir"})
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    max_seq_length: Optional[int] = field(
        default=512, metadata={"help": "max input sequence length"}
    )


def _clean_text(text):
    # TODO: Based on specific dataset
    return text


def _discard_sample(text):
    # TODO: Based on specific dataset
    return False


def preprocess_conv_qwen(tokenizer, source, system_message = "You are a helpful assistant."):
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


def preprocess_query_qwen(tokenizer, source, system_message = "You are a helpful assistant."):
    raw_text, input_id = make_context(
        tokenizer,
        source,
        history=None,
        system=system_message
    )

    return raw_text, input_id


def preprocess_sft_qwen(
    sources,
    tokenizer,
    system_message: str = "You are a helpful assistant."
):
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    new_examples = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
    }
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)

        new_examples["input_ids"].append(input_id)
        new_examples["labels"].append(target)
        new_examples["attention_mask"].append([1] * len(input_id))

    return new_examples


def preprocess_sft_function(tokenizer, use_qwen, examples):
    if use_qwen:
        sources = []
        for example in examples:
            sources.append(
                [
                    {"from": "user", "value": example['raw_instruction']},
                    {"from": "assistant", "value": example['generated_response']},
                ]
            )
        new_examples = preprocess_sft_qwen(sources, tokenizer)
    else:
        raise NotImplementedError()
    return new_examples


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


def _get_batch_dataset_local(
        model,
        batch_input,
        K=8,
        iter_id=0,
        local_rank=0,
        output_min_length=16,
        output_max_length=48,
        infer_batch_size=8,
        generation_kwargs=None,
        tokenizer=None,
        training_args=None,
        queries_to_scores_fn=None,
        output_reward_path=None,
        log_file=""
):
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    reward_eva = []
    reward_train = []
    batch_output = []
    for i, sample in enumerate(batch_input):
        input_ids = [sample['input_ids'] for _ in range(K)]

        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len

        inputs = torch.tensor(input_ids, dtype=torch.long).to(accelerator.device)
        with torch.no_grad():
            outputs = model.generate(inputs, **generation_kwargs)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        assert len(input_ids) == len(generated_texts)
        generated_texts = [
            generated_text.replace(sample['processed_instruction'], '') for i, generated_text in enumerate(generated_texts)
        ]
        generated_texts = [
            _clean_text(generated_text) for generated_text in generated_texts
        ]
        texts_for_rewards = [sample['processed_instruction'] + r for r in generated_texts]

        rewards = queries_to_scores_fn(texts_for_rewards)
        assert len(input_ids) == len(generated_texts) == len(rewards)

        reward_eva.append(rewards[0])

        # do some post-detection and discard the samples with certain criteria.
        for kk in range(K):
            if _discard_sample(generated_texts[kk]):
                rewards[kk] = -INF

        idx_to_record = np.argmax(rewards)
        # if we discard all the samples, we do not record the sample
        if rewards[idx_to_record] != -INF:
            _sample = dict(sample)
            _sample['generated_response'] = generated_texts[idx_to_record]
            _sample['reward'] = rewards[idx_to_record]

            batch_output.append(_sample)
            reward_train.append(rewards[idx_to_record])

    world_size = int(os.getenv("WORLD_SIZE", "1"))

    all_process_data = [{}] * world_size
    dist.all_gather_object(all_process_data, batch_output)

    all_process_eval_reward = [{}] * world_size
    dist.all_gather_object(all_process_eval_reward, reward_eva)

    all_process_train_set_reward = [{}] * world_size
    dist.all_gather_object(all_process_train_set_reward, reward_train)

    gathered_data = []
    gathered_reward = []
    gathered_train_reward = []
    for i in range(world_size):
        gathered_data.extend(all_process_data[i])
        gathered_reward.extend(all_process_eval_reward[i])
        gathered_train_reward.extend(all_process_train_set_reward[i])

    if training_args.local_rank == 0 and output_reward_path is not None:
        with open(output_reward_path, mode='a') as fout:
            fout.write('mean reward: ' + str(np.mean(gathered_reward)) + 'mean reward in training set: ' + str(
                np.mean(gathered_train_reward)))
            fout.write("\n")
    print_rank_0('mean reward: ' + str(np.mean(gathered_reward)) + 'mean reward in training set: ' + str(
                np.mean(gathered_train_reward)), log_file)

    reward_plot_seq.append(np.mean(gathered_reward))
    train_reward_plot_seq.append(np.mean(reward_train))

    if training_args.local_rank == 0:
        plt.plot(reward_plot_seq, marker="o")
        plt.plot(train_reward_plot_seq, marker="*")
        plt.legend(["Model reward", "Reward of SFT Set"])
        plt.savefig(training_args.output_dir + '/training_reward.png')
        plt.close()

    # We store the training set for monitoring the reject sampling process
    if local_rank == 0:
        with open(training_args.output_dir + "/train_set_" + str(iter_id) + ".json", 'w', encoding='utf8') as f:
            json.dump(gathered_data, f, ensure_ascii=False)

    # We need to make sure that the order of the samples are the same for each agent
    all_process_list = [{}] * world_size
    data_to_send = [gathered_data, local_rank]
    dist.all_gather_object(all_process_list, data_to_send)
    for i in range(world_size):
        if all_process_list[i][1] == 0:
            output_dataset = all_process_list[i][0]
            break

    print_rank_0(f"collected data of {len(output_dataset)}", log_file)

    return output_dataset


def main():





    #
    #
    # print("^" * 100)
    # print(accelerator.__dict__)
    # print("^" * 100)
    #
    #
    #






    parser = HfArgumentParser((
        TrainingArguments,
        RejectSamplingArguments
    ))



    #
    #
    # print("*" * 100)
    # print(accelerator.__dict__)
    # print("*" * 100)
    #
    #




    training_args, rs_args = parser.parse_args_into_dataclasses()




    print("%" * 100)
    print(accelerator.__dict__)
    print("%" * 100)





    # os.makedirs(rs_args.output_reward_path, exist_ok=True)

    log_file = os.path.join(training_args.output_dir, "print_log.txt")
    # local_rank = accelerator.local_process_index

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
    else:
        tokenizer = AutoTokenizer.from_pretrained(rs_args.model_name)
        tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})

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

    tokenizer.padding_side = "left"
    rm_tokenizer.padding_side = "left"

    print_rank_0(
        f"unk token: {tokenizer.unk_token}, "
        f"unk token id: {tokenizer.unk_token_id}, "
        f"pad token: {tokenizer.pad_token}, "
        f"pad token id: {tokenizer.pad_token_id}",
        log_file,
    )

    print_rank_0(
        f"rm unk token: {rm_tokenizer.unk_token}, "
        f"rm unk token id: {rm_tokenizer.unk_token_id}, "
        f"rm pad token: {rm_tokenizer.pad_token}, "
        f"rm pad token id: {rm_tokenizer.pad_token_id}",
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

    for i in range(2):
        print_rank_0("Instruct data example: {}".format(instruction_dataset[i]), log_file)

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
        # quantization_config = BitsAndBytesConfig(
        #     load_in_8bit=rs_args.load_in_8bit, load_in_4bit=rs_args.load_in_4bit
        # )
        # # Copy the model to each device
        # device_map = {"": local_rank}
        raise NotImplementedError()
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

    reward_pipe = pipeline(
        "text-classification",
        model=reward_model,
        device=accelerator.device,
        tokenizer=rm_tokenizer
    )

    # callable that takes a list of raw text and returns a list of corresponding reward scores
    def strings_to_scores(list_of_strings):
        pipe_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 1
        }
        pipe_outputs = reward_pipe(list_of_strings, **pipe_kwargs)
        rewards = [output[0]["score"] for output in pipe_outputs]
        return rewards

    # start iteration
    ITERATION = rs_args.num_rs_iteration
    collection_strategy = rs_args.collection_strategy
    sft_batch_size = rs_args.rs_batch_size

    if collection_strategy == "local":
        K = int(1 / rs_args.top_reward_percentage)
        M = int(sft_batch_size / world_size)
    else:
        # TODO: add global strategy
        raise NotImplementedError("We only support local data collection strategy now.")

    print("M:", M, "K:", K)
    print_rank_0(str(rs_args), log_file)

    data_size = len(instruction_dataset)
    random_idxs = np.arange(data_size)
    np.random.shuffle(random_idxs)

    generation_kwargs = {
        "min_length": 1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": 0.85,
    }

    for iteration in range(ITERATION):
        print_rank_0("#" * 20 + "Start Iteration {}".format(iteration) + "#" * 20, log_file)

        end_idx = np.min([data_size, (iteration + 1) * M])
        batch_input = instruction_dataset.select(random_idxs[iteration * M: end_idx])

        model.gradient_checkpointing_disable()
        model.config.use_cache = False

        start_time = time.time()
        if collection_strategy == "local":
            selected_dataset = _get_batch_dataset_local(
                model,
                batch_input,
                K,
                iteration,
                training_args.local_rank,
                output_min_length=rs_args.output_min_length,
                output_max_length=rs_args.output_max_length,
                infer_batch_size=K,
                generation_kwargs=generation_kwargs,
                tokenizer=tokenizer,
                training_args=training_args,
                queries_to_scores_fn=strings_to_scores,
                output_reward_path=rs_args.output_reward_path,
                log_file=log_file
            )
        else:
            # TODO: add global strategy
            raise NotImplementedError("We only support local data collection strategy now.")
        end_time = time.time()
        print_rank_0("It takes {} seconds to inference one stage".format(end_time - start_time), log_file)

        # use best-n samples to do sft
        start_time = time.time()
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

        with accelerator.main_process_first():
            train_dataset = Dataset.from_list(selected_dataset)
            train_dataset = train_dataset.map(
                partial(preprocess_sft_function, tokenizer, "qwen" in rs_args.model_name.lower()),
                batched=True,
                num_proc=max(cpu_count() // 2, 1)
            )
            train_dataset = train_dataset.filter(
                lambda x: len(x["input_ids"]) <= rs_args.max_seq_length
            )

        # Define the training arguments
        training_args.num_train_epochs = 1
        training_nums = len(train_dataset)
        global_batch_size = (
                accelerator.num_processes
                * training_args.gradient_accumulation_steps
                * training_args.per_device_train_batch_size
        )
        if training_args.dataloader_drop_last:
            num_steps = (
                    math.floor(training_nums / global_batch_size) * training_args.num_train_epochs
            )
        else:
            num_steps = (
                    math.ceil(training_nums / global_batch_size) * training_args.num_train_epochs
            )
        print_rank_0(
            "iteration = {}, num_gpus = {}, training_nums = {}, num_steps = {}, warmup_steps = {}".format(
                iteration,
                accelerator.num_processes,
                training_nums,
                num_steps,
                training_args.warmup_steps
            ),
            log_file,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        trainer.train()

        saved_path = os.path.join(training_args.output_dir, "iteration_{}".format(iteration))
        os.makedirs(saved_path, exist_ok=True)
        trainer.save_model(saved_path)

        # accelerator.wait_for_everyone()

        end_time = time.time()
        print_rank_0("It takes {} seconds to train one stage".format(end_time - start_time), log_file)

        if (iteration + 1) * M > data_size:
            print_rank_0("One instruction epoch is completed!!!", log_file)
            break


if __name__ == "__main__":




    #
    # print("%" * 100)
    # print(accelerator.__dict__)
    # print("%" * 100)





    main()

