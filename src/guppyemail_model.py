"""guppyemail decoder-only transformer model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.guppyemail_config import GuppyEmailConfig


class CausalSelfAttention(nn.Module):
    """Represent CausalSelfAttention behavior."""
    def __init__(self, config: GuppyEmailConfig):
        """Initialize the instance."""
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).view(
            batch_size, seq_len, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out(y)


class FeedForward(nn.Module):
    """Represent FeedForward behavior."""
    def __init__(self, config: GuppyEmailConfig):
        """Initialize the instance."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_hidden),
            nn.ReLU(),
            nn.Linear(config.ffn_hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        return self.net(x)


class Block(nn.Module):
    """Represent Block behavior."""
    def __init__(self, config: GuppyEmailConfig):
        """Initialize the instance."""
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GuppyEmailLM(nn.Module):
    """Represent GuppyEmailLM behavior."""
    def __init__(self, config: GuppyEmailConfig):
        """Initialize the instance."""
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Internal helper for init weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the forward pass."""
        _, seq_len = idx.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} > max_seq_len {self.config.max_seq_len}"
            )

        pos = torch.arange(seq_len, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Handle generate."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == self.config.eos_id:
                break
        return idx

    def parameter_count(self) -> int:
        """Handle parameter count."""
        return sum(param.numel() for param in self.parameters())
