from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .attention import Self_Attention
except ImportError:  # Allows: python model/transformer_block.py
    from attention import Self_Attention

from .moe import SparseMoE

class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        num_head: int,
        dropout: float = 0.1,

        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2, 
        expert_dim_multiplier: int = 2,
        expert_hidden_dim: int | None = None
    ) -> None:
        super().__init__()

        self.use_moe = use_moe

        self.layernorm_list = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(2)]
        )
        self.causal_attention = Self_Attention(
            embed_dim=embed_dim,
            context_length=context_length,
            num_head=num_head,
            dropout=dropout,
        )

        if use_moe:
            hidden_dim = (
                expert_hidden_dim
                if expert_hidden_dim is not None
                else embed_dim * expert_dim_multiplier
            )
            
            self.feed_forward = SparseMoE(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
            )
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_x = self.layernorm_list[0](x)
        x = x + self.causal_attention(
            normalized_x,
            attention_mask,
        )

        normalized_x = self.layernorm_list[1](x)

        if self.use_moe:
            moe_output, aux_loss, router_info = self.feed_forward(
                normalized_x
            )
            x = x+ moe_output
            return x, aux_loss, router_info
        x = x + self.feed_forward(normalized_x)
        return x , None , None


if __name__ == "__main__":
    block = Transformer(
        embed_dim=32,
        context_length=8,
        num_head=4,
        dropout=0.0,
    )
    x = torch.randn(3, 7, 32)
    attention_mask = torch.ones(3, 7, dtype=torch.long)

    output = block(x, attention_mask)
    print(f"output.shape={tuple(output.shape)}")
