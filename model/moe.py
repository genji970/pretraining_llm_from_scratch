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

        # fp32로 softmax 계산
        router_prob=torch.softmax(router_logits.float(),dim=-1,)

        # -------------------------------------------------
        # 2. top-k expert 선택, 
        # -------------------------------------------------

        # [total_tokens,top_k]
        routing_weight, selected_experts=torch.topk(
            router_prob,
            k=self.top_k,
            dim=-1,
        )

        routing_weight=routing_weight/routing_weight.sum(
            dim=-1,
            keepdim=True,
        ).to(x_flatten.dtype)

        # -------------------------------------------------
        # 3. Output buffer 
        # -------------------------------------------------

        output = torch.zeros_like(x_flatten)

        # -------------------------------------------------
        # 4. Expert dispatch
        # -------------------------------------------------
        
        for expert_idx, expert in enumerate(self.experts):
            
            # [total_tokenes, top_k]
            expert_mask = selected_experts == expert_idx # torch에서는 일종의 broadcasting 느낌으로, expert_idx가 torch tensor에 있나 없나 하나하나 확인한다.

            token_idx , topk_idx = torch.where(expert_mask)

            # numel()은 tensor 안에 들어있는 전체 원소 개수(number of elements)
            if token_idx.numel() == 0:
                continue
            
            expert_input = x_flatten[token_idx] # -> expert)mask가 true인 위치에서 얻은 token_idx를 이용해서, x_flatten에서 해당 token 행들만 뽑는다.
            """
            x_flatten =
            [
                token0_embedding,
                token1_embedding,
                token2_embedding,
                token3_embedding,
            ]

            expert_mask =
            [
                [ True, False],
                [False,  True],
                [False, False],
                [ True, False],
            ]

            expert_input = x_flatten[token_idx] ->
            expert_input =
            [
                x_flatten[0],
                x_flatten[1],
                x_flatten[3],
            ]
            """
            expert_output = expert(expert_input) # 이렇게하면 top-1이든 top-5든 똑같이 expert에 들어간다. 그러므로, router의 weight을 사용해서 top 몇인지를 반영할 수 있게 한다.
            
            #해당 token에서 이 expert가 가지는 routing weight
            # [num_selected_tokens]
            weight = routing_weight[token_idx, topk_idx]

            #[num_selected_tokens, 1]
            weight = weight.unsqueeze(dim=-1)
            
            #weighted expert output을 원래 token 위치에 더함.
            output.index_add_(
                0,
                token_idx,
                expert_output * weight,
            )
        
        # ----------------------------------
        # 5. Auxiliary load-balancing loss
        # ----------------------------------

        aux_loss, tokens_per_expert = self.auxiliary_loss(
            router_prob, 
            selected_experts,
        )

        # ----------------------------------
        # 6. 원래 shape 복원
        # ----------------------------------

        output = output.reshape(
            batch_size,
            sequence_length,
            embed_dim,
        )

        router_info = {
            "router_logits" :router_logits,
            "router_prob" : router_prob,
            "selected_experts" : selected_experts,
            "routing_weight" : routing_weight,
            "tokens_per_expert" : tokens_per_expert,
        }

        return output, aux_loss, router_info

            

            
            




