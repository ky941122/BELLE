# coding=utf-8
from functools import partial
import math
import os
import sys
import json
from datetime import timedelta

from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import LabelSmoother
import logging
from multiprocessing import cpu_count

from src.entry_point.reject_sampling.configs import RejectSamplingArguments

tqdm.pandas()

kwargs_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))  # increase NCCL timeout to 10 hours
accelerator = Accelerator(kwargs_handlers=[kwargs_handler])

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def print_rank_0(msg, log_file):
    if accelerator.is_main_process:
        with open(log_file, "a") as f:
            print(msg)
            f.write(msg + "\n")


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
        for raw_instruction, generated_response in zip(examples["raw_instruction"], examples["generated_response"]):
            sources.append(
                [
                    {"from": "user", "value": raw_instruction},
                    {"from": "assistant", "value": generated_response},
                ]
            )
        new_examples = preprocess_sft_qwen(sources, tokenizer)
    else:
        raise NotImplementedError()
    return new_examples


def main():
    parser = HfArgumentParser(RejectSamplingArguments)
    rs_args = parser.parse_args_into_dataclasses()[0]

    os.makedirs(rs_args.output_dir, exist_ok=True)

    log_file = os.path.join(rs_args.output_dir, "step_3_log.txt")
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
        step_2_data = []
        for file in os.listdir(rs_args.output_dir):
            if file.startswith("step_2_output_sub_") and file.endswith(".json"):
                path = os.path.join(rs_args.output_dir, file)
                with open(path, "r") as f:
                    _data = json.load(f)
                step_2_data.extend(_data)

        train_dataset = []
        for one in step_2_data:
            train_dataset.append(
                {
                    "raw_instruction": one["raw_instruction"],
                    "generated_response": one["best_model_generated_text"],
                }
            )
        train_dataset = Dataset.from_list(train_dataset)
        train_dataset = train_dataset.map(
            partial(preprocess_sft_function, tokenizer, "qwen" in rs_args.model_name.lower()),
            batched=True,
            num_proc=max(cpu_count() // 2, 1),
            remove_columns=["raw_instruction", "generated_response"],
        )
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids"]) <= rs_args.max_seq_length
        )

    for i in range(2):
        print_rank_0("Train tokenized example: {}".format(train_dataset[i]), log_file)

    # Define the training arguments
    training_nums = len(train_dataset)
    global_batch_size = (
        accelerator.num_processes
        * rs_args.gradient_accumulation_steps
        * rs_args.per_device_train_batch_size
    )
    print_rank_0("global_batch_size = {}".format(global_batch_size), log_file)
    if rs_args.dataloader_drop_last:
        num_steps = (
            math.floor(training_nums / global_batch_size) * rs_args.num_train_epochs
        )
    else:
        num_steps = (
            math.ceil(training_nums / global_batch_size) * rs_args.num_train_epochs
        )
    eval_steps = max(num_steps // (rs_args.num_train_epochs * 4), 5)
    print_rank_0(
        "num_gpus = {}, training_nums = {}, num_steps = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(
            accelerator.num_processes,
            training_nums,
            num_steps,
            rs_args.warmup_steps,
            eval_steps,
            eval_steps,
        ),
        log_file,
    )

    # `TrainingArguments` must be instantiated before loading model!!!
    training_args = TrainingArguments(
        output_dir=rs_args.output_dir,
        per_device_train_batch_size=rs_args.per_device_train_batch_size,
        per_device_eval_batch_size=rs_args.per_device_eval_batch_size,
        num_train_epochs=rs_args.num_train_epochs,
        gradient_accumulation_steps=rs_args.gradient_accumulation_steps,
        gradient_checkpointing=rs_args.gradient_checkpointing,
        learning_rate=rs_args.learning_rate,
        report_to="wandb" if rs_args.report_to == "wandb" else "tensorboard",
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_steps=rs_args.logging_steps,
        # evaluation_strategy="steps",
        save_strategy="steps",
        bf16=rs_args.bf16,
        fp16=rs_args.fp16,
        weight_decay=rs_args.weight_decay,
        lr_scheduler_type=rs_args.lr_scheduler_type,
        # eval_steps=eval_steps,
        save_steps=eval_steps,
        warmup_steps=rs_args.warmup_steps,
        overwrite_output_dir=rs_args.overwrite_output_dir,
        resume_from_checkpoint=rs_args.resume_from_checkpoint,
        save_total_limit=rs_args.save_total_limit,
        # load_best_model_at_end=True,
        ddp_timeout=36000,
        seed=rs_args.seed,
        dataloader_drop_last=rs_args.dataloader_drop_last,
    )

    print_rank_0(
        "world_size = {}".format(training_args.world_size),
        log_file,
    )

    # Load the model
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

    model = AutoModelForCausalLM.from_pretrained(
        rs_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False
    # Initialize our Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    # https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1567
    # 没有指定max_length，默认是pad to the longest length in the batch
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    accelerator.wait_for_everyone()
    print_rank_0(
        "\n Training completed!!! If there's a warning about missing keys above, please disregard :)",
        log_file,
    )


if __name__ == "__main__":
    main()

