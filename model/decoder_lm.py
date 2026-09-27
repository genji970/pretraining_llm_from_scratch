from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .stacking import Stacking
except ImportError:  # Allows: python model/decoder_lm.py
    from stacking import Stacking


class DecoderLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_num: int,
        embed_dim: int,
        context_length: int,
        num_head: int,
        dropout: float = 0.1,

        moe_layers: list[int] | None = None,
        num_experts: int = 4,
        top_k: int =2,
        expert_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )
        self.blocks = Stacking(
            block_num=block_num,
            embed_dim=embed_dim,
            context_length=context_length,
            num_head=num_head,
            dropout=dropout,

            moe_layeres=moe_layers,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
        )
        self.final_layernorm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch_size, sequence_length]"
            )
        if input_ids.size(1) > self.context_length:
            raise ValueError(
                "input sequence exceeds configured context_length"
            )

        x = self.token_embedding(input_ids)
        x, aux_loss, router_infos = self.blocks(x, attention_mask,)
        x = self.final_layernorm(x)
        logits=self.proj(x)
        return logits, aux_loss, router_infos


if __name__ == "__main__":
    model = DecoderLanguageModel(
        vocab_size=32,
        block_num=2,
        embed_dim=32,
        context_length=8,
        num_head=4,
        dropout=0.0,
    )
    input_ids = torch.randint(0, 32, (3, 7))
    attention_mask = torch.ones_like(input_ids)

    logits = model(input_ids, attention_mask)
    print(f"logits.shape={tuple(logits.shape)}")
