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


