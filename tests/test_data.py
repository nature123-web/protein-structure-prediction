"""Tests for sequence encoding, distance binning, cropping and collation."""

import numpy as np
import torch

from src.data import (
    PAD_IDX,
    SyntheticProteinDataset,
    bin_centers,
    collate,
    CroppedDataset,
    decode_sequence,
    distance_to_bins,
    encode_sequence,
    pairwise_distances,
    random_crop,
)


def test_encode_decode_roundtrip():
    seq = "MKTAYIAKQRQ"
    assert decode_sequence(encode_sequence(seq)) == seq


def test_unknown_residue_maps_to_x():
    encoded = encode_sequence("MZKB")
    assert decode_sequence(encoded) == "MXKX"


def test_pad_index_is_reserved():
    """No real amino acid may collide with the padding index."""
    assert PAD_IDX not in encode_sequence("ACDEFGHIKLMNPQRSTVWYX").tolist()


def test_pairwise_distances_are_symmetric_with_zero_diagonal():
    coords = np.random.default_rng(0).normal(size=(10, 3))
    d = pairwise_distances(coords)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0, atol=1e-8)


def test_distance_binning_is_monotonic():
    """Larger distances must never land in a smaller bin."""
    dist = torch.linspace(0, 40, 200)
    bins = distance_to_bins(dist, num_bins=24, min_dist=2.0, max_dist=22.0)
    assert (bins[1:] >= bins[:-1]).all()
    assert bins.min() == 0 and bins.max() == 23


def test_bin_centers_are_increasing_and_correct_length():
    centers = bin_centers(24, 2.0, 22.0)
    assert centers.shape == (24,)
    assert (centers[1:] > centers[:-1]).all()


def test_synthetic_dataset_is_deterministic():
    a = SyntheticProteinDataset(4, seed=7)[2]
    b = SyntheticProteinDataset(4, seed=7)[2]
    assert torch.equal(a.sequence, b.sequence)
    assert torch.allclose(a.distances, b.distances)


def test_synthetic_record_is_self_consistent():
    record = SyntheticProteinDataset(4, min_length=50, max_length=80, seed=1)[0]
    length = record.sequence.shape[0]
    assert 50 <= length <= 80
    assert record.distances.shape == (length, length)
    assert record.mask.shape == (length,)
    assert torch.allclose(record.distances, record.distances.T, atol=1e-4)


def test_synthetic_proteins_have_realistic_contact_density():
    """Real proteins are compact: a few contacts per residue, not a full graph."""
    record = SyntheticProteinDataset(1, min_length=120, max_length=120, seed=3)[0]
    length = record.sequence.shape[0]
    sep = (torch.arange(length)[None, :] - torch.arange(length)[:, None]).abs()
    long_range = (record.distances < 8.0) & (sep >= 12)
    contacts_per_residue = long_range.sum().item() / length
    assert 0.05 < contacts_per_residue < 10.0


def test_random_crop_preserves_submatrix_alignment():
    record = SyntheticProteinDataset(1, min_length=100, max_length=100, seed=5)[0]
    rng = np.random.default_rng(0)
    cropped = random_crop(record, 32, rng)
    n = cropped.sequence.shape[0]
    assert n == 32
    assert cropped.distances.shape == (32, 32)
    # The crop must still be a valid distance matrix.
    assert torch.allclose(cropped.distances, cropped.distances.T, atol=1e-4)
    assert torch.allclose(torch.diagonal(cropped.distances), torch.zeros(n), atol=1e-4)


def test_crop_is_noop_for_short_proteins():
    record = SyntheticProteinDataset(1, min_length=20, max_length=20, seed=0)[0]
    cropped = random_crop(record, 128, np.random.default_rng(0))
    assert cropped.sequence.shape[0] == 20


def test_deterministic_crop_is_stable_across_calls():
    ds = CroppedDataset(
        SyntheticProteinDataset(2, min_length=100, max_length=100, seed=0),
        crop_size=48, deterministic=True,
    )
    assert torch.equal(ds[0].sequence, ds[0].sequence)


def test_collate_pads_and_builds_pair_mask():
    base = SyntheticProteinDataset(3, min_length=40, max_length=90, seed=2)
    batch = collate([base[0], base[1], base[2]])
    b, max_len = batch["sequence"].shape
    assert b == 3
    assert batch["distances"].shape == (3, max_len, max_len)
    assert batch["pair_mask"].shape == (3, max_len, max_len)

    for i in range(3):
        n = int(batch["mask"][i].sum())
        # Padding region must be zeroed and unmasked.
        assert (batch["sequence"][i, n:] == PAD_IDX).all()
        assert not batch["pair_mask"][i, n:, :].any()
        assert batch["pair_mask"][i, :n, :n].all()
