# coding=utf-8
from functools import partial
import math
import os
import sys
import json
from typing import Optional
from datetime import timedelta
from dataclasses import dataclass, field
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
    train_data: str = field(default="", metadata={"help": "train data path"})
    eval_data: str = field(default="", metadata={"help": "eval data path"})
    cache_dir: str = field(default="", metadata={"help": "cache dir"})
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    max_prompt_length: Optional[int] = field(
        default=1024, metadata={"help": "max input prompt length"}
    )
    max_seq_length: Optional[int] = field(
        default=2048, metadata={"help": "max total sequence length"}
    )
    report_to: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    logging_steps: Optional[int] = field(
        default=500, metadata={"help": "the number of update steps between two logs"}
    )
    # save_steps: Optional[int] = field(
    #     default=100, metadata={"help": "the number of steps between two saving"}
    # )
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the training batch size on each device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "evaluating batch size"}
    )
    dataloader_drop_last: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Drop the last incomplete batch if it is not divisible by the batch size."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "the number of training epochs"}
    )
    warmup_steps: int = field(
        default=100, metadata={"help": "Linear warmup over warmup_steps."}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    bf16: Optional[bool] = field(default=True, metadata={"help": "bfloat16"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "float16"})
    weight_decay: float = field(
        default=0.001, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to a folder with a valid checkpoint for your model."
        },
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    debug: Optional[bool] = field(default=False, metadata={"help": "debug with toy dataset"})


def preprocess_sft_function(tokenizer, max_cot_length, examples):
    new_examples = {
        "input_ids": [],
        "labels": [],
        "attention_mask": []
    }

    for question, cot, final_answer in zip(examples["question"], examples["new_cot"], examples["final_answer"]):
        instruction = "[INST] " + question + " [/INST]"
        instruct_id = tokenizer.encode(instruction, add_special_tokens=False)

        total_answer = "{} The answer is: {}".format(cot, final_answer)
        total_answer_id = tokenizer.encode(total_answer, add_special_tokens=False)

        input_id = instruct_id + total_answer_id
        label = [IGNORE_TOKEN_ID]*len(instruct_id) + total_answer_id

        input_id = [tokenizer.bos_token_id] + input_id + [tokenizer.eos_token_id]
        label = [tokenizer.bos_token_id] + label + [tokenizer.eos_token_id]

        assert len(input_id) == len(label)

        new_examples["input_ids"].append(input_id)
        new_examples["labels"].append(label)
        new_examples["attention_mask"].append([1] * len(input_id))

    return new_examples


def main():
    parser = HfArgumentParser(MyArguments)
    my_args = parser.parse_args_into_dataclasses()[0]

    os.makedirs(my_args.output_dir, exist_ok=True)

    log_file = os.path.join(my_args.output_dir, "log.txt")
    local_rank = accelerator.local_process_index
    global_rank = accelerator.process_index

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(my_args.model_name, trust_remote_code=True)
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
        f"bos token: {tokenizer.bos_token}, "
        f"bos token id: {tokenizer.bos_token_id}, "
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
        train_dataset = train_dataset.map(
            partial(preprocess_sft_function, tokenizer, 256),
            batched=True,
            num_proc=max(cpu_count() // 2, 1),
            remove_columns=["question", "cot", "new_cot", "final_answer"],
        )
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids"]) <= my_args.max_seq_length
        )

        eval_dataset = load_dataset(
            "json", data_files=my_args.eval_data, cache_dir=my_args.cache_dir
        )["train"]
        eval_dataset = eval_dataset.map(
            partial(preprocess_sft_function, tokenizer, 256),
            batched=True,
            num_proc=max(cpu_count() // 2, 1),
            remove_columns=["question", "cot", "new_cot", "final_answer"],
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids"]) <= my_args.max_seq_length
        )

    for i in range(2):
        print_rank_0("Train tokenized example: {}".format(train_dataset[i]), log_file)

    # Define the training arguments
    training_nums = len(train_dataset)
    global_batch_size = (
        accelerator.num_processes
        * my_args.gradient_accumulation_steps
        * my_args.per_device_train_batch_size
    )
    print_rank_0("global_batch_size = {}".format(global_batch_size), log_file)
    if my_args.dataloader_drop_last:
        num_steps = (
            math.floor(training_nums / global_batch_size) * my_args.num_train_epochs
        )
    else:
        num_steps = (
            math.ceil(training_nums / global_batch_size) * my_args.num_train_epochs
        )
    eval_steps = max(num_steps // (my_args.num_train_epochs * 4), 5)
    eval_steps = 99999
    print_rank_0(
        "num_gpus = {}, training_nums = {}, num_steps = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(
            accelerator.num_processes,
            training_nums,
            num_steps,
            my_args.warmup_steps,
            eval_steps,
            eval_steps,
        ),
        log_file,
    )

    # `TrainingArguments` must be instantiated before loading model!!!
    training_args = TrainingArguments(
        output_dir=my_args.output_dir,
        per_device_train_batch_size=my_args.per_device_train_batch_size,
        per_device_eval_batch_size=my_args.per_device_eval_batch_size,
        num_train_epochs=my_args.num_train_epochs,
        gradient_accumulation_steps=my_args.gradient_accumulation_steps,
        gradient_checkpointing=my_args.gradient_checkpointing,
        learning_rate=my_args.learning_rate,
        report_to="wandb" if my_args.report_to == "wandb" else "tensorboard",
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_steps=my_args.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=my_args.bf16,
        fp16=my_args.fp16,
        weight_decay=my_args.weight_decay,
        lr_scheduler_type=my_args.lr_scheduler_type,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        warmup_steps=my_args.warmup_steps,
        overwrite_output_dir=my_args.overwrite_output_dir,
        resume_from_checkpoint=my_args.resume_from_checkpoint,
        save_total_limit=my_args.save_total_limit,
        load_best_model_at_end=True,
        ddp_timeout=36000,
        seed=my_args.seed,
        dataloader_drop_last=my_args.dataloader_drop_last,
    )

    print_rank_0(
        "world_size = {}".format(training_args.world_size),
        log_file,
    )

    # Load the model
    if my_args.load_in_8bit and my_args.load_in_4bit:
        raise ValueError(
            "You can't load the model in 8 bits and 4 bits at the same time"
        )
    elif my_args.load_in_8bit or my_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=my_args.load_in_8bit, load_in_4bit=my_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": local_rank}
    else:
        device_map = None
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        my_args.model_name,
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
        eval_dataset=eval_dataset,
        # eval_dataset=None,
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


