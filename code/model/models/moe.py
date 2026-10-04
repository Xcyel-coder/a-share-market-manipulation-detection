"""
Mixture of Experts implementation for time-series encoder.

Architecture inspired by DeepSeek-MoE:
- Shared experts: always activated (capture universal patterns)
- Routed experts: top-k gating (specialize in different patterns)
- Load balancing auxiliary loss to prevent expert collapse

This MoE layer replaces the standard FFN in Transformer blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert: SwiGLU-style two-layer FFN."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TopKRouter(nn.Module):
    """
    Top-K expert router with load balancing loss.

    For each token, computes routing probabilities over all experts
    and selects the top-k. Includes jitter noise during training
    for better exploration.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        jitter: float = 0.1,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.jitter = jitter
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            weights: [batch, seq_len, top_k]  — normalized routing weights
            indices: [batch, seq_len, top_k]  — selected expert indices
            balance_loss: scalar — load balancing auxiliary loss
        """
        # [batch, seq_len, num_experts]
        logits = self.gate(x)

        if self.training and self.jitter > 0:
            noise = torch.randn_like(logits) * self.jitter
            logits = logits + noise

        routing_probs = F.softmax(logits, dim=-1)

        top_k_probs, top_k_indices = routing_probs.topk(self.top_k, dim=-1)
        # renormalize
        weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        balance_loss = self._load_balance_loss(routing_probs, top_k_indices)

        return weights, top_k_indices, balance_loss

    def _load_balance_loss(
        self, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Switch Transformer style load balancing loss.
        L = N * sum_i(f_i * P_i), where:
          f_i = fraction of tokens routed to expert i
          P_i = mean routing probability for expert i
        """
        # f_i: fraction of tokens dispatched to each expert
        one_hot = F.one_hot(indices, self.num_experts).float()  # [..., top_k, N_exp]
        one_hot = one_hot.sum(dim=-2)  # [..., N_exp]  (count per expert)
        f = one_hot.mean(dim=[0, 1])  # [N_exp]

        # P_i: mean probability per expert
        p = probs.mean(dim=[0, 1])  # [N_exp]

        return self.num_experts * (f * p).sum()


class MoEFeedForward(nn.Module):
    """
    MoE feed-forward layer with shared + routed experts.

    Shared experts process all tokens (universal patterns).
    Routed experts specialize via top-k gating.
    Output = shared_output + routed_output.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_routed_experts: int = 8,
        num_shared_experts: int = 2,
        top_k: int = 2,
        router_jitter: float = 0.1,
    ):
        super().__init__()
        self.num_routed = num_routed_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k

        self.shared_experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_routed_experts)]
        )
        self.router = TopKRouter(d_model, num_routed_experts, top_k, router_jitter)

    def forward(
        self, x: torch.Tensor, metadata: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            metadata: optional [batch, metadata_dim] for metadata-aware routing
        Returns:
            output: [batch, seq_len, d_model]
            balance_loss: scalar
        """
        shared_out = torch.stack(
            [expert(x) for expert in self.shared_experts], dim=0
        ).mean(dim=0)

        if metadata is not None and hasattr(self.router, 'metadata_proj'):
            weights, indices, balance_loss = self.router(x, metadata=metadata)
        else:
            weights, indices, balance_loss = self.router(x)
        routed_out = self._dispatch_and_combine(x, weights, indices)

        return shared_out + routed_out, balance_loss

    def _dispatch_and_combine(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to selected experts and combine weighted outputs."""
        batch, seq_len, d_model = x.shape
        output = torch.zeros_like(x)

        flat_x = x.reshape(-1, d_model)              # [B*S, D]
        flat_w = weights.reshape(-1, self.top_k)      # [B*S, K]
        flat_idx = indices.reshape(-1, self.top_k)     # [B*S, K]

        for k in range(self.top_k):
            expert_indices = flat_idx[:, k]             # [B*S]
            expert_weights = flat_w[:, k].unsqueeze(-1) # [B*S, 1]

            for e_id in range(self.num_routed):
                mask = expert_indices == e_id
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = self.routed_experts[e_id](expert_input)
                    output.reshape(-1, d_model)[mask] += (
                        expert_weights[mask] * expert_output
                    )

        return output

    def get_expert_utilization(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Analyze expert activation patterns (for visualization in Phase 3)."""
        with torch.no_grad():
            logits = self.router.gate(x)
            probs = F.softmax(logits, dim=-1)
            _, indices, _ = self.router(x)

            # Count per expert
            counts = torch.zeros(self.num_routed, device=x.device)
            for e_id in range(self.num_routed):
                counts[e_id] = (indices == e_id).float().sum()

            return {
                "routing_probs": probs.mean(dim=[0, 1]),  # [num_experts]
                "expert_counts": counts,
                "expert_utilization": counts / counts.sum(),
            }
