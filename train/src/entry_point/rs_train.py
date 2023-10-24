# coding=utf-8
import os
import sys
import time
import math
import json
from typing import Any, Dict, List, Optional, Union
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
    LlamaTokenizer
)
from transformers.trainer_utils import (
    EvaluationStrategy,
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SchedulerType,
    ShardedDDPOption,
)
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ExplicitEnum
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
tqdm.pandas()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

trainer_log_levels = {
    "detail": logging.DEBUG,  # will also print filename and line number
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

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


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
    LION_8BIT = "lion_8bit"
    LION = "lion_32bit"
    PAGED_ADAMW = "paged_adamw_32bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_LION = "paged_lion_32bit"
    PAGED_LION_8BIT = "paged_lion_8bit"
    RMSPROP = "rmsprop"


class DebugOption(ExplicitEnum):
    UNDERFLOW_OVERFLOW = "underflow_overflow"
    TPU_METRICS_DEBUG = "tpu_metrics_debug"


@dataclass
class MyTrainingArguments:
    framework = "pt"
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
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

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )

    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for training."
            )
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_eval_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for evaluation."
            )
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " evaluation_strategy."
            )
        },
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    log_level: Optional[str] = field(
        default="passive",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": trainer_log_levels.keys(),
        },
    )
    log_level_replica: Optional[str] = field(
        default="warning",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": trainer_log_levels.keys(),
        },
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    logging_nan_inf_filter: bool = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})
    save_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_total_limit: Optional[int] = field(
        default=None,
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
    save_safetensors: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use safetensors saving and loading for state dicts instead of default torch.load and torch.save."
        },
    )
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )
    no_cuda: bool = field(
        default=False,
        metadata={"help": "This argument is deprecated. It will be removed in version 5.0 of 🤗 Transformers."},
    )
    use_cpu: bool = field(
        default=False,
        metadata={
            "help": " Whether or not to use cpu. If set to False, we will use cuda/tpu/mps/npu device if available."
        },
    )
    use_mps_device: bool = field(
        default=False,
        metadata={
            "help": "This argument is deprecated. `mps` device will be used if available similar to `cuda` device."
            " It will be removed in version 5.0 of 🤗 Transformers"
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: Optional[int] = field(default=None, metadata={"help": "Random seed to be used with data samplers."})
    jit_mode_eval: bool = field(
        default=False, metadata={"help": "Whether or not to use PyTorch jit trace for inference"}
    )
    use_ipex: bool = field(
        default=False,
        metadata={
            "help": (
                "Use Intel extension for PyTorch when it is available, installation:"
                " 'https://github.com/intel/intel-extension-for-pytorch'"
            )
        },
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    half_precision_backend: str = field(
        default="auto",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["auto", "cuda_amp", "apex", "cpu_amp"],
        },
    )
    bf16_full_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may"
                " change."
            )
        },
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit"},
    )
    tf32: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    ddp_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "The backend to be used for distributed training",
            "choices": ["nccl", "gloo", "mpi", "ccl", "hccl"],
        },
    )
    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={
            "help": (
                "Deprecated, the use of `--debug tpu_metrics_debug` is preferred. TPU: Whether to print debug metrics"
            )
        },
    )
    debug: Union[str, List[DebugOption]] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data."
            )
        },
    )
    sharded_ddp: Optional[Union[List[ShardedDDPOption], str]] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to use sharded DDP training (in distributed training only). The base option should be"
                " `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` like"
                " this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or `zero_dp_3`"
                " with the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`."
            ),
        },
    )
    fsdp: Optional[Union[List[FSDPOption], str]] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training"
                " only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add"
                " CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op"
                " offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard"
                " auto_wrap` or `shard_grad_op auto_wrap`."
            ),
        },
    )
    fsdp_min_num_params: int = field(
        default=0,
        metadata={
            "help": (
                "This parameter is deprecated. FSDP's minimum number of parameters for Default Auto Wrapping. (useful"
                " only when `fsdp` field is passed)."
            )
        },
    )
    # Do not touch this type annotation or it will stop working in CLI
    fsdp_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with FSDP (Pytorch Fully Sharded  Data Parallel). The value is either a"
                "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "This parameter is deprecated. Transformer layer class name (case-sensitive) to wrap, e.g,"
                " `BertLayer`, `GPTJBlock`, `T5Block` .... (useful only when `fsdp` flag is passed)."
            )
        },
    )
    # Do not touch this type annotation or it will stop working in CLI
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )

    default_optim = "adamw_torch"
    # XXX: enable when pytorch==2.0.1 comes out - we want to give it time to get all the bugs sorted out
    # if is_torch_available() and version.parse(version.parse(torch.__version__).base_version) >= version.parse("2.1.0"):
    #     default_optim = "adamw_torch_fused"
    # and update the doc above to:
    # optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch_fused"` (for torch<2.1.0 `"adamw_torch"`):
    optim: Union[OptimizerNames, str] = field(
        default=default_optim,
        metadata={"help": "The optimizer to use."},
    )
    optim_args: Optional[str] = field(default=None, metadata={"help": "Optional arguments to supply to optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    length_column_name: Optional[str] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_bucket_cap_mb: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `bucket_cap_mb` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_broadcast_buffers: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `broadcast_buffers` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    use_legacy_prediction_loop: bool = field(
        default=False, metadata={"help": "Whether or not to use the legacy prediction_loop in the Trainer."}
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_strategy: Union[HubStrategy, str] = field(
        default="every_save",
        metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
    )
    hub_token: Optional[str] = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    hub_private_repo: bool = field(default=False, metadata={"help": "Whether the model repository is private or not."})
    hub_always_push: bool = field(
        default=False,
        metadata={"help": "Unless `True`, the Trainer will skip pushes if the previous one wasn't finished yet."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    include_inputs_for_metrics: bool = field(
        default=False, metadata={"help": "Whether or not the inputs will be passed to the `compute_metrics` function."}
    )
    # Deprecated arguments
    fp16_backend: str = field(
        default="auto",
        metadata={
            "help": "Deprecated. Use half_precision_backend instead",
            "choices": ["auto", "cuda_amp", "apex", "cpu_amp"],
        },
    )
    push_to_hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to which push the `Trainer`."}
    )
    push_to_hub_organization: Optional[str] = field(
        default=None, metadata={"help": "The name of the organization in with to which push the `Trainer`."}
    )
    push_to_hub_token: Optional[str] = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    _n_gpu: int = field(init=False, repr=False, default=-1)
    mp_parameters: str = field(
        default="",
        metadata={"help": "Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer"},
    )

    auto_find_batch_size: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to automatically decrease the batch size in half and rerun the training loop again each time"
                " a CUDA Out-of-Memory was reached"
            )
        },
    )
    full_determinism: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
                " training. Important: this will negatively impact the performance, so only use it for debugging."
            )
        },
    )
    torchdynamo: Optional[str] = field(
        default=None,
        metadata={
            "help": "This argument is deprecated, use `--torch_compile_backend` instead.",
        },
    )
    ray_scope: Optional[str] = field(
        default="last",
        metadata={
            "help": (
                'The scope to use when doing hyperparameter search with Ray. By default, `"last"` will be used. Ray'
                " will then use the last checkpoint of all trials, compare those, and select the best one. However,"
                " other options are also available. See the Ray documentation"
                " (https://docs.ray.io/en/latest/tune/api_docs/analysis.html"
                "#ray.tune.ExperimentAnalysis.get_best_trial)"
                " for more options."
            )
        },
    )
    ddp_timeout: Optional[int] = field(
        default=1800,
        metadata={
            "help": "Overrides the default timeout for distributed training (value should be given in seconds)."
        },
    )
    torch_compile: bool = field(
        default=False, metadata={"help": "If set to `True`, the model will be wrapped in `torch.compile`."}
    )
    torch_compile_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which backend to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )
    torch_compile_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which mode to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )

    dispatch_batches: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to dispatch batches across devices in distributed training. If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process"
            "and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose"
            "underlying dataset is an `IterableDataset`, `False` otherwise."
        },
    )

    include_tokens_per_second: Optional[bool] = field(
        default=False,
        metadata={"help": "If set to `True`, the speed metrics will include `tgs` (tokens per second per device)."},
    )


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
    for i, sample in tqdm(enumerate(batch_input)):
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
    parser = HfArgumentParser((
        MyTrainingArguments,
        RejectSamplingArguments
    ))
    training_args, rs_args = parser.parse_args_into_dataclasses()

    os.makedirs(rs_args.output_reward_path, exist_ok=True)

    log_file = os.path.join(training_args.output_dir, "print_log.txt")
    local_rank = accelerator.local_process_index

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
        model.config.use_cache = True
        model.eval()

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
    main()

