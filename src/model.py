"""ProteinContactNet: sequence -> distogram.

The network has three stages.

1. **1D encoder.** A transformer over residues. Attention is the right
   inductive bias here because contacts are long-range by definition -- a purely
   convolutional 1D encoder would need impractical depth to relate residue 5 to
   residue 200.

2. **Pair featurisation.** Per-residue features ``h`` are lifted to a pairwise
   tensor. We concatenate ``h_i``, ``h_j`` and a relative-position encoding
   rather than only using an outer product: the raw features preserve identity
   information that a product destroys, and relative separation is strongly
   predictive (|i-j| < 6 is almost always in contact).

3. **2D tower.** Residual convolutions with a cycling dilation schedule. Each
   block sees a wider neighbourhood than the last, so by the end of the tower a
   pair's prediction is informed by the whole surrounding patch of the map --
   this is what enforces the local consistency that makes predicted maps
   physically plausible (contacts come in helical bands and sheet stripes, never
   as isolated pixels).

The output is symmetrised, because a distance matrix must be.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD_IDX, VOCAB_SIZE


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal encoding."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels only, applied independently at each (i, j).

    InstanceNorm2d / GroupNorm pool statistics across the whole L x L map, which
    makes a residue pair's normalised features depend on how much *padding*
    happens to be in its batch -- two identical proteins get different
    predictions depending on their neighbours. Normalising per spatial position
    removes that coupling entirely and is what makes the tower exactly
    padding-invariant (see ``tests/test_model.py``).
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class ResidualBlock2D(nn.Module):
    """Pre-activation residual block with a dilated 3x3 convolution.

    ``pair_mask`` is re-applied before every convolution. Because padded
    positions are then held at exactly zero, a convolution reading into the
    padded region sees the same zeros that ``F.conv2d``'s own zero-padding would
    supply for an unpadded input -- so batching changes nothing numerically.
    """

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = ChannelLayerNorm2d(channels)
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.norm2 = ChannelLayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        h = F.elu(self.norm1(x))
        if pair_mask is not None:
            h = h * pair_mask
        h = self.dropout(self.conv1(h))
        h = F.elu(self.norm2(h))
        if pair_mask is not None:
            h = h * pair_mask
        out = x + self.conv2(h)
        if pair_mask is not None:
            out = out * pair_mask
        return out


class ProteinContactNet(nn.Module):
    """Predicts a distogram over residue pairs.

    Args:
        d_model: width of the per-residue representation.
        embed_dim: if given, the input is assumed to be pre-computed embeddings
            (e.g. ESM-2) of this width instead of amino-acid indices.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        n_conv_blocks: int = 8,
        conv_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        num_bins: int = 24,
        dropout: float = 0.1,
        embed_dim: int | None = None,
        max_relative_position: int = 32,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.max_relative_position = max_relative_position

        if embed_dim is None:
            self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
            self.input_proj = nn.Identity()
        else:
            self.embedding = None
            self.input_proj = nn.Linear(embed_dim, d_model)

        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and only emits a
        # warning; disable it explicitly to keep the logs clean.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, n_encoder_layers, enable_nested_tensor=False
        )

        # Relative sequence separation, bucketed and embedded.
        self.rel_pos_embed = nn.Embedding(2 * max_relative_position + 2, 16)

        pair_in = 2 * d_model + 16
        self.pair_proj = nn.Conv2d(pair_in, conv_channels, 1)
        self.tower = nn.ModuleList([
            ResidualBlock2D(conv_channels, dilations[i % len(dilations)], dropout)
            for i in range(n_conv_blocks)
        ])
        self.out_norm = ChannelLayerNorm2d(conv_channels)
        self.head = nn.Conv2d(conv_channels, num_bins, 1)

    def _relative_positions(self, length: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(length, device=device)
        rel = idx[None, :] - idx[:, None]
        rel = rel.clamp(-self.max_relative_position, self.max_relative_position)
        return rel + self.max_relative_position

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return distogram logits of shape ``(B, num_bins, L, L)``.

        ``sequence`` is ``(B, L)`` long indices, or ``(B, L, embed_dim)`` floats
        when the model was built with ``embed_dim``. ``mask`` is ``(B, L)`` and
        True for real residues.
        """
        if self.embedding is not None:
            h = self.embedding(sequence)
            pad_mask = sequence == PAD_IDX
            if mask is not None:
                pad_mask = pad_mask | ~mask
        else:
            h = self.input_proj(sequence)
            pad_mask = ~mask if mask is not None else None

        h = self.pos_encoding(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        b, length, d = h.shape
        # (B, 1, L, L): a pair is real only if both of its residues are.
        if pad_mask is not None and pad_mask.any():
            residue_ok = (~pad_mask).to(h.dtype)
            pair_mask = (residue_ok[:, :, None] * residue_ok[:, None, :]).unsqueeze(1)
        else:
            pair_mask = None

        # Broadcast per-residue features across both axes of the pair tensor.
        h_i = h.unsqueeze(2).expand(b, length, length, d)
        h_j = h.unsqueeze(1).expand(b, length, length, d)
        rel = self.rel_pos_embed(self._relative_positions(length, h.device))
        rel = rel.unsqueeze(0).expand(b, -1, -1, -1)

        pair = torch.cat([h_i, h_j, rel], dim=-1).permute(0, 3, 1, 2)
        x = self.pair_proj(pair)
        if pair_mask is not None:
            x = x * pair_mask
        for block in self.tower:
            x = block(x, pair_mask)
        head_in = F.elu(self.out_norm(x))
        if pair_mask is not None:
            head_in = head_in * pair_mask
        logits = self.head(head_in)

        # A distance matrix is symmetric, so the logits must be too. Averaging
        # with the transpose is cheaper and more stable than penalising asymmetry.
        return 0.5 * (logits + logits.transpose(-1, -2))

    @torch.no_grad()
    def predict_contacts(
        self,
        sequence: torch.Tensor,
        num_bins: int | None = None,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        threshold: float = 8.0,
    ) -> torch.Tensor:
        """Contact probability per pair: total mass in bins below ``threshold``."""
        from .data import bin_centers

        logits = self(sequence)
        probs = F.softmax(logits, dim=1)
        centers = bin_centers(num_bins or self.num_bins, min_dist, max_dist)
        centers = centers.to(probs.device)
        contact_bins = (centers < threshold).view(1, -1, 1, 1)
        return (probs * contact_bins).sum(dim=1)


def distogram_loss(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    pair_mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Masked cross-entropy over residue pairs.

    Only pairs where both residues are resolved contribute. Returning a
    mask-weighted mean (rather than summing) keeps the loss scale independent of
    protein length, so the learning rate does not need retuning per crop size.
    """
    loss = F.cross_entropy(
        logits, target_bins, reduction="none", label_smoothing=label_smoothing
    )
    mask = pair_mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom
