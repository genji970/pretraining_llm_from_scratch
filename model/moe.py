from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.network=nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class SparseMoE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int=1,
        dropout: float=0.0,
    ) -> None:
        super().__init__()

        if num_experts <= 0:
            raise ValueError("num experts must be positive.")
        
        if not 1<=top_k<=num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts.")
        
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # token -> expert logits
        self.router = nn.Linear(
            embed_dim,
            num_experts,
            bias=False,
        )

        self.experts = nn.ModuleList(
            [
                Expert(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )
    
    def auxiliary_loss(
        self,
        router_prob: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        router_prob: [total_tokens, num_experts]
        selected_experts: [total_tokens, top_k]
        """
        assignment = F.one_hot(
            selected_experts,
            num_classes=self.num_experts,
        ).float() #assignment.shape : [total_tokens,top_k,num_experts]

        # [total_tokens, top_k, num_experts] , num_experts : 총 몇번 사용되나 해당 expert가.
        """
        tokens_per_expert <-> mean_router_prob의 차이 : 
        hard routing 결과를 보느냐 soft router prob을 보느냐
        mean_router_prob의 의미 : router가 각 expert에게 평균적으로 얼마만큼의 확률을 줬는지 보는 soft preference이다.
        """
        tokens_per_expert = assignment.mean(dim=(0,1)) # -> 전체 token * top_k * ruting slot 중 expert e가 차지한 비율
        mean_router_prob=router_prob.mean(dim=0)

        load_balancing_loss=self.num_experts*torch.sum(
            tokens_per_expert*mean_router_prob
        )
        return load_balancing_loss,tokens_per_expert
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:

    """
    x: [batch_size,sequence_length,embed_dim]
    """
        batch_size, sequence_length, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"expected embed_dim={self.embed_dim},"
                f"received embed_dim={embed_dim}"
            )
        
        # [batch * sequence, embed_dim]
        x_flatten=x.reshape(-1,embed_dim)
        
        #[total_tokens,num_experts]
        router_logits=self.router(x_flatten)

        router_prob=torch.softmax(router_logits.float(),dim=-1,)

        # [total_tokens,top_k]
        router_weight, selected_experts=torch.topk(
            router_prob,
            k=self.top_k,
            dim=-1,
        )

        routing_weight=routing_weight/routing_weight.sum(
            dim=-1,
            keepdim=True,
        ).to(x_flatten.dtype)

        output=torch.zeros_like(x_flatten)

        

