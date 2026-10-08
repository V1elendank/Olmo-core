"""
Short SFT run of OLMo-2 7B on the UVA CS Slurm cluster (no Beaker / Weka dependencies).

Adapted from ``src/scripts/train/sft/Olmo-2-7B-SFT.py``. Launch with torchrun on one node, e.g.

    torchrun --standalone --nproc-per-node=4 uva/sft_olmo2_7b_uva.py train \
        --checkpoint=/bigtemp/$USER/olmo/ckpt/OLMo-2-1124-7B/model_and_optim \
        --dataset_path=/bigtemp/$USER/olmo/data/tulu3-sample \
        --save_folder=/bigtemp/$USER/olmo/runs/olmo2-7b-sft-test

Use ``dry_run`` instead of ``train`` to print the config without touching a GPU.
Any extra ``--a.b.c=value`` arguments are applied as config overrides (OLMo-core dot notation).
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import cast

from rich import print

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank, get_world_size
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import prepare_cli_environment, seed_all

log = logging.getLogger(__name__)


@dataclass
class UVASFTConfig(Config):
    run_name: str
    model: TransformerConfig
    dataset: NumpyPackedFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 33333


def build_config(args: argparse.Namespace, overrides: list[str]) -> UVASFTConfig:
    tokenizer_config = TokenizerConfig.dolma2()
    seq_len = args.seq_len
    world_size = get_world_size() if args.cmd == "train" else args.world_size

    # flash-attn supports intra-document masking for packed SFT data; PyTorch SDPA does not,
    # so with the torch backend we pack without doc-length masking (fine for a feasibility test).
    use_flash = args.attn_backend == "flash_2"

    dataset = NumpyPackedFSLDatasetConfig(
        tokenizer=tokenizer_config,
        work_dir=f"{args.save_folder}/dataset-cache",
        paths=[f"{args.dataset_path.rstrip('/')}/token_ids_part_*.npy"],
        expand_glob=True,
        label_mask_paths=[f"{args.dataset_path.rstrip('/')}/labels_mask_*.npy"],
        generate_doc_lengths=use_flash,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=seq_len,
    )

    rank_microbatch_tokens = args.microbatch_seqs * seq_len
    global_batch_tokens = args.global_batch_seqs * seq_len
    if global_batch_tokens % (rank_microbatch_tokens * world_size) != 0:
        raise ValueError(
            f"global_batch_seqs ({args.global_batch_seqs}) must be divisible by "
            f"microbatch_seqs * world_size ({args.microbatch_seqs} * {world_size})"
        )

    model_factory = {"7B": TransformerConfig.olmo2_7B, "190M": TransformerConfig.olmo2_190M}
    model = model_factory[args.model_size](
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName(args.attn_backend),
    )

    config = UVASFTConfig(
        run_name=args.run_name,
        model=model,
        dataset=dataset,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=global_batch_tokens,
            seed=34521,
            num_workers=4,
            work_dir=f"{args.save_folder}/dataset-cache",
        ),
        train_module=TransformerTrainModuleConfig(
            rank_microbatch_size=rank_microbatch_tokens,
            max_sequence_length=seq_len,
            z_loss_multiplier=None,
            compile_model=args.compile,
            optim=SkipStepAdamWConfig(
                lr=args.lr,
                weight_decay=0.0,
                betas=(0.9, 0.95),
                compile=False,
            ),
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
            ),
            ac_config=TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.full
                if args.full_ac
                else TransformerActivationCheckpointingMode.selected_modules,
                modules=None if args.full_ac else ["blocks.*.feed_forward"],
            ),
            scheduler=LinearWithWarmup(warmup_fraction=0.1, alpha_f=0.0),
            max_grad_norm=1.0,
        ),
        trainer=TrainerConfig(
            save_folder=args.save_folder,
            load_strategy=LoadStrategy.never,  # base checkpoint is loaded manually below
            save_overwrite=True,
            no_checkpoints=not args.save_checkpoints,
            metrics_collect_interval=1,
            cancel_check_interval=10,
            max_duration=Duration.steps(args.steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback()),
    ).merge(overrides)

    if os.environ.get("UVA_CPU_TEST"):
        config.trainer.callbacks.pop("gpu_monitor", None)

    return config


def train(checkpoint: str, config: UVASFTConfig):
    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()

    if checkpoint.lower() == "none":
        log.warning("No checkpoint given: training from RANDOM init (smoke test only)")
    else:
        log.info(f"Loading base model weights from '{checkpoint}'...")
        trainer.load_checkpoint(checkpoint, load_trainer_state=False, load_optim_state=False)

    trainer.fit()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cmd", choices=["train", "dry_run", "prep_data"])
    parser.add_argument("--run_name", default="olmo2-7b-sft-uva-test")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="OLMo-core checkpoint dir (…/model_and_optim), or 'none' for random init",
    )
    parser.add_argument(
        "--dataset_path", required=True, help="Dir with token_ids_part_*.npy + labels_mask_*.npy"
    )
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument(
        "--microbatch_seqs", type=int, default=1, help="Sequences per GPU per micro-step"
    )
    parser.add_argument(
        "--global_batch_seqs", type=int, default=32, help="Sequences per optimizer step"
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--attn_backend", default="torch", choices=["torch", "flash_2", "flash_3"])
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument(
        "--full_ac", action="store_true", help="Full activation checkpointing (less memory)"
    )
    parser.add_argument(
        "--save_checkpoints", action="store_true", help="Save checkpoints (~120GB each w/ optim)"
    )
    parser.add_argument(
        "--model_size", default="7B", choices=["7B", "190M"], help="190M = tiny smoke test"
    )
    parser.add_argument("--world_size", type=int, default=4, help="Only used for dry_run")
    args, overrides = parser.parse_known_args()

    if args.cmd in ("dry_run", "prep_data"):
        prepare_cli_environment()
    elif os.environ.get("UVA_CPU_TEST"):  # CPU-only smoke test of the code path (no GPU)
        prepare_training_environment(backend="gloo")
    else:
        prepare_training_environment()

    config = build_config(args, overrides)
    if get_local_rank() == 0:
        print(config)

    if args.cmd == "prep_data":
        # Single process, no GPU: pack documents into fixed-length instances and cache the result
        # in <save_folder>/dataset-cache so the torchrun job only reuses it. Fails fast here
        # instead of leaving the other GPU ranks waiting at a barrier.
        dataset = config.dataset.build()
        dataset.prepare()
        log.info(f"Dataset ready: {len(dataset):,d} instances of {args.seq_len} tokens")
    elif args.cmd == "train":
        try:
            train(args.checkpoint, config)
        finally:
            teardown_training_environment()


if __name__ == "__main__":
    main()
