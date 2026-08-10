"""Contact-prediction metrics.

Plain accuracy is useless here: a protein of length L has O(L) true contacts
among O(L^2) pairs, so "predict no contact" scores >98%. The field therefore
reports **precision of the top-L/k most confident predictions**, split by
sequence separation, because only long-range contacts constrain the fold.
"""

from __future__ import annotations

from typing import Dict

import torch

# Standard CASP separation ranges (|i - j|).
RANGES: Dict[str, tuple[int, int]] = {
    "short": (6, 11),
    "medium": (12, 23),
    "long": (24, 10 ** 9),
}


def separation_mask(length: int, lo: int, hi: int, device=None) -> torch.Tensor:
    """(L, L) bool mask selecting pairs with lo <= |i-j| <= hi."""
    idx = torch.arange(length, device=device)
    sep = (idx[None, :] - idx[:, None]).abs()
    return (sep >= lo) & (sep <= hi)


def top_k_precision(
    pred_probs: torch.Tensor,
    true_contacts: torch.Tensor,
    valid: torch.Tensor,
    k_divisor: float = 1.0,
    sep_range: tuple[int, int] = (24, 10 ** 9),
) -> float:
    """Precision among the top ``L/k_divisor`` predictions in a separation band.

    Args:
        pred_probs: (L, L) predicted contact probability.
        true_contacts: (L, L) bool, ground-truth contacts.
        valid: (L, L) bool, pairs that are supervised (both residues resolved).
        k_divisor: 1 for P@L, 2 for P@L/2, 5 for P@L/5.
        sep_range: inclusive |i-j| bounds.

    Returns:
        Precision in [0, 1], or ``nan`` when no candidate pairs exist -- which
        happens for short crops with no long-range pairs at all. Returning nan
        rather than 0 keeps those crops from dragging the average down.
    """
    length = pred_probs.shape[0]
    lo, hi = sep_range
    band = separation_mask(length, lo, hi, pred_probs.device) & valid
    # Only the upper triangle: the matrix is symmetric, so counting both halves
    # would double-count every contact.
    band = band & torch.triu(torch.ones_like(band), diagonal=1).bool()

    n_candidates = int(band.sum())
    if n_candidates == 0:
        return float("nan")

    k = max(1, int(length / k_divisor))
    k = min(k, n_candidates)

    scores = pred_probs.masked_fill(~band, float("-inf")).flatten()
    top_idx = torch.topk(scores, k).indices
    hits = true_contacts.flatten()[top_idx]
    return float(hits.float().mean())


def evaluate_contacts(
    pred_probs: torch.Tensor,
    distances: torch.Tensor,
    pair_mask: torch.Tensor,
    contact_threshold: float = 8.0,
) -> Dict[str, float]:
    """Full metric suite for one batch, averaged over proteins.

    ``pred_probs`` is (B, L, L); ``distances`` and ``pair_mask`` match.
    """
    true_contacts = (distances < contact_threshold) & pair_mask
    results: Dict[str, list[float]] = {}

    for range_name, sep in RANGES.items():
        for divisor, label in ((1.0, "L"), (2.0, "L_2"), (5.0, "L_5")):
            key = f"P@{label}_{range_name}"
            values = [
                top_k_precision(
                    pred_probs[b], true_contacts[b], pair_mask[b], divisor, sep
                )
                for b in range(pred_probs.shape[0])
            ]
            values = [v for v in values if v == v]  # drop nan
            results[key] = values

    return {
        k: (sum(v) / len(v) if v else float("nan")) for k, v in results.items()
    }


def mean_absolute_distance_error(
    pred_distances: torch.Tensor,
    true_distances: torch.Tensor,
    pair_mask: torch.Tensor,
    max_dist: float = 22.0,
) -> float:
    """MAE in angstroms over supervised pairs that are genuinely close.

    Pairs beyond ``max_dist`` are excluded: they all fall in the same catch-all
    bin, so the model cannot distinguish 30 A from 80 A and scoring it on those
    pairs measures nothing.
    """
    mask = pair_mask & (true_distances < max_dist)
    if not mask.any():
        return float("nan")
    err = (pred_distances - true_distances).abs()[mask]
    return float(err.mean())
