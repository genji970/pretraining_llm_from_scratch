from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .transformer_block import Transformer
except ImportError:  # Allows: python model/stacking.py
    from transformer_block import Transformer


class Stacking(nn.Module):
    def __init__(
        self,
        block_num: int,
        embed_dim: int,
        context_length: int,
        num_head: int,
        dropout: float = 0.1,

        moe_layers: list[int] | None = None,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if block_num <= 0:
            raise ValueError("block_num must be positive")

        moe_layers=set(moe_layers or [])

        self.transformer_block_list = nn.ModuleList(
            [
                Transformer(
                    embed_dim=embed_dim,
                    context_length=context_length,
                    num_head=num_head,
                    dropout=dropout,

                    use_moe=layer_idx in moe_layers,
                    num_experts=num_experts,
                    top_k=top_k,
                    expert_hidden_dim=expert_hidden_dim,
                )
                for layer_idx in range(block_num)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total_aux_loss = x.new_zeros(())
        router_infos = {}

        for layer_idx, transformer_block in enumerate(
            self.transformer_block_list
        ):
            x, aux_loss, router_info = transformer_block(
                x, 
                attention_mask,
            )

            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
                router_infos[layer_idx] = router_info
        return x, total_aux_loss, router_infos


if __name__ == "__main__":
    blocks = Stacking(
        block_num=2,
        embed_dim=32,
        context_length=8,
        num_head=4,
        dropout=0.0,
    )
    x = torch.randn(3, 7, 32)
    attention_mask = torch.ones(3, 7, dtype=torch.long)

    output = blocks(x, attention_mask)
    print(f"output.shape={tuple(output.shape)}")
