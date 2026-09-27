from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrainConfig:
    # Data
    dataset_name: str
    dataset_config: str | None
    dataset_split: str
    text_column: str
    total_documents: int
    chunk_size: int
    streaming: bool
    shuffle_buffer: int
    seed: int

    # Tokenizer
    tokenizer_name: str

    # Model and dataloader
    context_length: int
    batch_size: int
    num_workers: int
    block_num: int
    embed_dim: int
    num_heads: int
    dropout: float

    # MoE 설정
    moe_layers: list[int]
    num_experts: int
    top_k: int
    expert_hidden_dim: int | None

    # Training
    output_dir: str
    epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    log_every_steps: int
    max_steps: int
    early_stop_step: int
    device: str
    resume_from: str | None

    # MoE training
    moe_aux_loss_coef: float

    # Automatic early stopping
    early_stop_metric: str
    early_stop_mode: str
    early_stop_threshold: float | None
    early_stop_patience_steps: int
    early_stop_min_delta: float
    early_stop_warmup_steps: int

    # Checkpointing
    save_every_steps: int
    save_each_chunk: bool
    save_optimizer_state: bool
    save_best_checkpoint: bool
    best_checkpoint_metric: str
    best_checkpoint_mode: str
    best_checkpoint_min_delta: float
    save_threshold_checkpoint: bool
    threshold_checkpoint_metric: str
    threshold_checkpoint_value: float | None
    threshold_checkpoint_mode: str
    threshold_checkpoint_once: bool

    # Metrics and visualization
    metrics_every_steps: int
    metric_smoothing_window: int
    jsonl_metrics: bool
    tensorboard: bool
    save_plots: bool
    plot_every_steps: int
    plot_metrics: str
    max_plot_points: int
    log_parameter_norm: bool
    histogram_every_steps: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the from-scratch decoder-only language model in data chunks."
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset_name", type=str, default="wikimedia/wikipedia")
    data.add_argument("--dataset_config", type=str, default="20231101.en")
    data.add_argument("--dataset_split", type=str, default="train")
    data.add_argument("--text_column", type=str, default="text")
    data.add_argument(
        "--total_documents",
        type=int,
        default=100_000,
        help="Total number of non-empty documents used across all chunks.",
    )
    data.add_argument(
        "--chunk_size",
        type=int,
        default=10_000,
        help="Maximum number of non-empty documents held in memory at one time.",
    )
    data.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Hugging Face streaming. Recommended for very large datasets.",
    )
    data.add_argument("--shuffle_buffer", type=int, default=10_000)
    data.add_argument("--seed", type=int, default=42)

    tokenizer = parser.add_argument_group("tokenizer")
    tokenizer.add_argument(
        "--tokenizer_name",
        type=str,
        default="openai-community/gpt2",
        help="Pretrained tokenizer name or local tokenizer directory.",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--context_length", type=int, default=1024)
    model.add_argument("--block_num", type=int, default=8)
    model.add_argument("--embed_dim", type=int, default=512)
    model.add_argument("--num_heads", type=int, default=8)
    model.add_argument("--dropout", type=float, default=0.1)

    model.add_argument(
    "--moe_layers",
    type=int,
    nargs="*",
    default=[],
    help="0-based Transformer block indices that use MoE instead of dense FFN.",
    )

    model.add_argument(
        "--num_experts",
        type=int,
        default=4,
    )

    model.add_argument(
        "--top_k",
        type=int,
        default=2,
    )

    model.add_argument(
        "--expert_hidden_dim",
        type=int,
        default=None,
    )

    train = parser.add_argument_group("training")
    train.add_argument("--output_dir", type=str, default="outputs/chunked_pretrain")
    train.add_argument("--batch_size", type=int, default=8)
    train.add_argument("--num_workers", type=int, default=0)
    train.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes over each in-memory data chunk.",
    )
    train.add_argument("--learning_rate", type=float, default=2e-4)
    train.add_argument("--weight_decay", type=float, default=0.01)
    train.add_argument("--max_grad_norm", type=float, default=1.0)
    
    train.add_argument(
        "--moe_aux_loss_coef",
        type=float,
        default=0.01,
    )

    train.add_argument(
        "--log_every_steps",
        type=int,
        default=10,
        help="Print the current training status every N optimizer steps.",
    )
    train.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Absolute global optimizer-step limit. 0 disables this limit.",
    )
    train.add_argument(
        "--early_stop_step",
        type=int,
        default=0,
        help="Manual global step at which training stops. 0 disables it.",
    )
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--resume_from", type=str, default=None)

    early_stop = parser.add_argument_group("automatic early stopping")
    early_stop.add_argument(
        "--early_stop_metric",
        type=str,
        default="train_loss",
        help="Metric monitored by threshold and patience early stopping.",
    )
    early_stop.add_argument(
        "--early_stop_mode",
        choices=["min", "max"],
        default="min",
        help="Use min when lower is better and max when higher is better.",
    )
    early_stop.add_argument(
        "--early_stop_threshold",
        type=float,
        default=None,
        help="Stop after the monitored metric reaches this threshold.",
    )
    early_stop.add_argument(
        "--early_stop_patience_steps",
        type=int,
        default=0,
        help="Stop after this many steps without improvement. 0 disables it.",
    )
    early_stop.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum change counted as an improvement.",
    )
    early_stop.add_argument(
        "--early_stop_warmup_steps",
        type=int,
        default=0,
        help="Do not apply metric-based early stopping before this step.",
    )

    checkpoint = parser.add_argument_group("checkpointing")
    checkpoint.add_argument(
        "--save_every_steps",
        "--save_every",
        dest="save_every_steps",
        type=int,
        default=1_000,
        help="Save a periodic checkpoint every N steps. 0 disables it.",
    )
    checkpoint.add_argument(
        "--save_each_chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a checkpoint after a complete chunk finishes.",
    )
    checkpoint.add_argument(
        "--save_optimizer_state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include optimizer state in every checkpoint.",
    )
    checkpoint.add_argument(
        "--save_best_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    checkpoint.add_argument(
        "--best_checkpoint_metric", type=str, default="train_loss"
    )
    checkpoint.add_argument(
        "--best_checkpoint_mode", choices=["min", "max"], default="min"
    )
    checkpoint.add_argument(
        "--best_checkpoint_min_delta", type=float, default=0.0
    )
    checkpoint.add_argument(
        "--save_threshold_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    checkpoint.add_argument(
        "--threshold_checkpoint_metric", type=str, default="train_loss"
    )
    checkpoint.add_argument(
        "--threshold_checkpoint_value", type=float, default=None
    )
    checkpoint.add_argument(
        "--threshold_checkpoint_mode", choices=["min", "max"], default="min"
    )
    checkpoint.add_argument(
        "--threshold_checkpoint_once",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    monitor = parser.add_argument_group("metrics and visualization")
    monitor.add_argument("--metrics_every_steps", type=int, default=10)
    monitor.add_argument("--metric_smoothing_window", type=int, default=20)
    monitor.add_argument(
        "--jsonl_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    monitor.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    monitor.add_argument(
        "--save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    monitor.add_argument(
        "--plot_every_steps",
        type=int,
        default=500,
        help="Regenerate PNG plots every N steps. 0 means only at the end.",
    )
    monitor.add_argument(
        "--plot_metrics",
        type=str,
        default=(
            "train_loss,perplexity,grad_norm,learning_rate,"
            "tokens_per_second,samples_per_second,"
            "gpu_allocated_gb,gpu_reserved_gb,gpu_peak_allocated_gb"
        ),
        help="Comma-separated metrics saved as separate PNG files.",
    )
    monitor.add_argument("--max_plot_points", type=int, default=5_000)
    monitor.add_argument(
        "--log_parameter_norm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    monitor.add_argument(
        "--histogram_every_steps",
        type=int,
        default=0,
        help="TensorBoard parameter/gradient histogram interval. 0 disables it.",
    )

    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_names = (
        "total_documents",
        "chunk_size",
        "context_length",
        "batch_size",
        "block_num",
        "embed_dim",
        "num_heads",
        "epochs",
        "log_every_steps",
        "metrics_every_steps",
        "metric_smoothing_window",
        "max_plot_points",
        "num_experts",
        "top_k",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")

    nonnegative_names = (
        "shuffle_buffer",
        "num_workers",
        "max_steps",
        "early_stop_step",
        "early_stop_patience_steps",
        "early_stop_min_delta",
        "early_stop_warmup_steps",
        "save_every_steps",
        "best_checkpoint_min_delta",
        "plot_every_steps",
        "histogram_every_steps",
        "weight_decay",
        "dropout",
        "moe_aux_loss_coef",
    )
    for name in nonnegative_names:
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")

    if args.embed_dim % args.num_heads != 0:
        parser.error("--embed_dim must be divisible by --num_heads")
    if (args.embed_dim // args.num_heads) % 2 != 0:
        parser.error("head_dim = embed_dim / num_heads must be even for RoPE")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must satisfy 0 <= dropout < 1")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    if args.max_grad_norm <= 0:
        parser.error("--max_grad_norm must be positive")
    if not args.tokenizer_name.strip():
        parser.error("--tokenizer_name must not be empty")
    if not args.plot_metrics.strip():
        parser.error("--plot_metrics must contain at least one metric name")
    if args.save_threshold_checkpoint and args.threshold_checkpoint_value is None:
        parser.error(
            "--threshold_checkpoint_value is required when "
            "--save_threshold_checkpoint is enabled"
        )
    if args.histogram_every_steps > 0 and not args.tensorboard:
        parser.error("--histogram_every_steps requires --tensorboard")
    if args.plot_every_steps > 0 and not args.save_plots:
        parser.error("--plot_every_steps requires --save_plots")

    if args.top_k > args.num_experts:
        parser.error("--top_k must be <= --num_experts")

    if args.expert_hidden_dim is not None and args.expert_hidden_dim <= 0:
        parser.error("--expert_hidden_dim must be positive")

    for layer_idx in args.moe_layers:
        if not 0 <= layer_idx < args.block_num:
            parser.error(
                f"--moe_layers contains invalid layer {layer_idx}; "
                f"valid range is 0 to {args.block_num - 1}"
            )
            
def parse_config(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    _validate(namespace, parser)

    if namespace.dataset_config == "":
        namespace.dataset_config = None
    if namespace.resume_from == "":
        namespace.resume_from = None

    return TrainConfig(**vars(namespace))


if __name__ == "__main__":
    toy_config = parse_config(
        [
            "--dataset_name",
            "toy",
            "--dataset_config",
            "",
            "--total_documents",
            "8",
            "--chunk_size",
            "4",
            "--block_num",
            "2",
            "--embed_dim",
            "32",
            "--num_heads",
            "4",
            "--no-tensorboard",
            "--no-save_plots",
            "--plot_every_steps",
            "0",
        ]
    )
    print(toy_config)
