# small_transformer_prior.py

import torch
import torch.nn as nn


class SmallTransformerPrior(nn.Module):
    """
    A lightweight Transformer-based autoregressive prior for discrete index sequences.

    Args:
        num_embeddings (int): Size of the codebook (number of discrete classes K).
        emb_dim (int): Dimension of token, position, and condition embeddings.
        condition_dim (int): Number of distinct condition labels (e.g., 3 for three classes).
        n_layers (int): Number of TransformerDecoder layers.
        n_heads (int): Number of attention heads in each layer.
        ff_hidden (int): Hidden dimension of the feed‑forward layers.
        max_len (int): Maximum sequence length (T). Must be >= actual T during both train and inference.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        emb_dim: int = 64,
        condition_dim: int = 3,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_hidden: int = 128,
        max_len: int = 30,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb_dim = emb_dim
        self.condition_dim = condition_dim
        self.max_len = max_len

        # Token embedding: maps each discrete index (0..K-1) to an emb_dim vector
        self.token_emb = nn.Embedding(num_embeddings, emb_dim)

        # Positional embedding: one embedding per position 0..max_len-1
        self.pos_emb = nn.Embedding(max_len, emb_dim)

        # Condition embedding: maps each condition label (0..condition_dim-1) to emb_dim vector
        self.cond_emb = nn.Embedding(condition_dim, emb_dim)

        # Build a stack of TransformerDecoder layers
        # We use the same tensor as both "tgt" and "memory" to implement a masked autoregressive
        #             TransformerDecoder without a separate encoder.
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_hidden,
            batch_first=True,      # (batch, seq, feature)
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        # Final linear layer to map back to logits over discrete classes
        self.fc_out = nn.Linear(emb_dim, num_embeddings)

    def forward(self, idx_seq: torch.LongTensor, cond: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass of the autoregressive prior.

        Args:
            idx_seq (LongTensor): [B, T] tensor of integer indices (each ∈ [0, num_embeddings-1]).
            cond (LongTensor):   [B] tensor of condition labels (each ∈ [0, condition_dim-1]).

        Returns:
            logits (Tensor): [B, T, num_embeddings] unnormalized scores for next-index classification
                             at each time step.
        """
        B, T = idx_seq.shape
        assert T <= self.max_len, f"Sequence length T={T} exceeds max_len={self.max_len}"

        # 1) Token embedding for the discrete indices: [B, T, emb_dim]
        token_embeddings = self.token_emb(idx_seq)  # [B, T, emb_dim]

        # 2) Positional embedding: need to create a [B, T] index for positions 0..T-1
        pos_indices = torch.arange(T, device=idx_seq.device).unsqueeze(0)  # [1, T]
        pos_embeddings = self.pos_emb(pos_indices)                         # [1, T, emb_dim]
        pos_embeddings = pos_embeddings.expand(B, -1, -1)                  # [B, T, emb_dim]

        # 3) Condition embedding: [B, emb_dim] --> expand to [B, T, emb_dim]
        cond_embeddings = self.cond_emb(cond).unsqueeze(1).expand(-1, T, -1)  # [B, T, emb_dim]

        # 4) Combine token + position + condition embeddings
        x = token_embeddings + pos_embeddings + cond_embeddings  # [B, T, emb_dim]

        # 5) Create a causal mask of shape [T, T] where True indicates masked positions
        #    We want to mask out positions j > i when computing output at position i.
        #    torch.triu with diagonal=1 creates ones for positions (i, j) where j > i.
        causal_mask = torch.triu(torch.ones((T, T), device=idx_seq.device), diagonal=1).bool()
        #    The TransformerDecoder expects a "tgt_mask" where masked positions are True.

        # 6) Pass through the TransformerDecoder.
        #    We feed x as both the "tgt" and "memory" arguments so that each position
        #    attends only to itself and prior positions (causal).
        out = self.transformer(
            tgt=x,        # [B, T, emb_dim]
            memory=x,     # [B, T, emb_dim], shared for masked self-attention
            tgt_mask=causal_mask,
        )  # out: [B, T, emb_dim]

        # 7) Predict logits for next discrete index at each position: [B, T, num_embeddings]
        logits = self.fc_out(out)

        return logits
