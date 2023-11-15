# coding=utf-8
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class RejectSamplingArguments:
    """
    Define a class RewardPipeArguments to configure reward pipeline.
    """
    output_dir: str = field(
        default="./outputs/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    output_reward_path: Optional[str] = field(
        default="./outputs/rewards_records/",
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
    generate_min_temperature: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "generate_min_temperature"
            ),
        },
    )
    generate_max_temperature: Optional[float] = field(
        default=1.5,
        metadata={
            "help": (
                "generate_max_temperature"
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
    n_best_nums: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "sample nums for n-best sampling"
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
    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "max input prompt length"}
    )
    max_seq_length: Optional[int] = field(
        default=1024, metadata={"help": "max input sequence length"}
    )
    version: Optional[int] = field(
        default=0, metadata={"help": "version"}
    )
    start_n_sub: Optional[int] = field(
        default=0, metadata={"help": "start_n_sub"}
    )
    report_to: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    logging_steps: Optional[int] = field(
        default=500, metadata={"help": "the number of update steps between two logs"}
    )
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
