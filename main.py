from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from config import TrainConfig, parse_config
from data.dataset import Train_Dataset, create_dataloader
from data.load_data import (
    load_next_text_chunk,
    load_pretraining_stream,
    skip_source_rows,
)
from model.model import DecoderLanguageModel
from train.trainer import PretrainingTrainer

from util.util import set_seed , load_progress, save_progress_atomic


def run(config: TrainConfig) -> dict[str, float | int | str]:
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(config.tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected tokenizer must define an EOS token")
    # GPT-2 has no native pad token. Dataset masking uses sequence lengths, so a
    # real EOS in the text is not mistaken for padding.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir / "tokenizer")

    model = DecoderLanguageModel(
        vocab_size=len(tokenizer),
        block_num=config.block_num,
        embed_dim=config.embed_dim,
        context_length=config.context_length,
        num_head=config.num_heads,
        dropout=config.dropout,

        moe_layers=config.moe_layers,
        num_experts=config.num_experts,
        top_k=config.top_k,
        expert_hidden_dim=config.expert_hidden_dim,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameter_count:,} tokenizer_vocab={len(tokenizer):,}")

    trainer = PretrainingTrainer(model=model, config=config)
    progress_path = output_dir / "progress.json"

    trained_documents = 0
    source_rows_consumed = 0
    chunk_id = 0

    if config.resume_from is not None:
        trainer.load_checkpoint(config.resume_from)
        progress = load_progress(progress_path)
        trained_documents = int(progress.get("trained_documents", 0))
        source_rows_consumed = int(progress.get("source_rows_consumed", 0))
        chunk_id = int(progress.get("next_chunk_id", 0))
        print(
            f"resumed checkpoint={config.resume_from} "
            f"documents={trained_documents:,} chunk={chunk_id} "
            f"step={trainer.global_step}"
        )

    stream = load_pretraining_stream(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
        streaming=config.streaming,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer,
    )
    stream = skip_source_rows(stream, source_rows_consumed)
    row_iterator = iter(stream)

    stop_reason = "total_documents"

    while trained_documents < config.total_documents:
        remaining_documents = config.total_documents - trained_documents
        current_chunk_size = min(config.chunk_size, remaining_documents)

        texts, consumed_rows = load_next_text_chunk(
            row_iterator=row_iterator,
            text_column=config.text_column,
            chunk_size=current_chunk_size,
        )
        if not texts:
            stop_reason = "dataset_exhausted"
            break

        chunk_document_count = len(texts)
        print(
            f"chunk={chunk_id} loaded_documents={chunk_document_count:,} "
            f"trained={trained_documents:,}/{config.total_documents:,}"
        )

        train_dataset = Train_Dataset(
            texts=texts,
            tokenizer=tokenizer,
            context_length=config.context_length,
        )
        del texts
        gc.collect()

        train_loader = create_dataloader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            pad_token_id=int(tokenizer.pad_token_id),
            shuffle=True,
            num_workers=config.num_workers,
        )
        print(f"chunk={chunk_id} sequences={len(train_dataset):,} batches={len(train_loader):,}")

        result = trainer.train_chunk(train_loader=train_loader, chunk_id=chunk_id)

        # Commit source progress only after the whole chunk has trained. If a stop
        # happens mid-chunk, a later resume re-reads this chunk instead of silently
        # skipping documents that were never trained.
        if bool(result["completed_chunk"]):
            trained_documents += chunk_document_count
            source_rows_consumed += consumed_rows

            chunk_checkpoint_path = ""
            if config.save_each_chunk:
                chunk_checkpoint = trainer.save_checkpoint(
                    category="chunk",
                    name=f"checkpoint-chunk-{chunk_id:06d}.pt",
                    chunk_id=chunk_id,
                )
                chunk_checkpoint_path = str(chunk_checkpoint)

            chunk_id += 1
            save_progress_atomic(
                progress_path,
                {
                    "trained_documents": trained_documents,
                    "source_rows_consumed": source_rows_consumed,
                    "next_chunk_id": chunk_id,
                    "global_step": trainer.global_step,
                    "checkpoint_path": chunk_checkpoint_path,
                },
            )

        print(json.dumps(result, indent=2))

        del train_loader
        del train_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if bool(result["should_stop"]):
            stop_reason = str(result["stop_reason"])
            break

    final_chunk_id = max(chunk_id - 1, 0)
    return trainer.finalize(
        chunk_id=final_chunk_id,
        stop_reason=stop_reason,
        trained_documents=trained_documents,
    )


if __name__ == "__main__":
    run_config = parse_config()
    print(json.dumps(run_config.to_dict(), indent=2))
    result = run(run_config)
    print(json.dumps(result, indent=2))
