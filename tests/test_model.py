"""Shape, symmetry and masking tests for ProteinContactNet."""

import torch

from src.data import PAD_IDX, VOCAB_SIZE
from src.model import ProteinContactNet, distogram_loss


def make_model(**kwargs):
    defaults = dict(
        d_model=32, n_heads=2, n_encoder_layers=1, dim_feedforward=64,
        n_conv_blocks=2, conv_channels=16, num_bins=10, dropout=0.0,
    )
    defaults.update(kwargs)
    return ProteinContactNet(**defaults)


def test_output_shape():
    model = make_model()
    seq = torch.randint(1, VOCAB_SIZE, (2, 20))
    logits = model(seq)
    assert logits.shape == (2, 10, 20, 20)


def test_logits_are_symmetric():
    """A distance matrix is symmetric, so the predicted distogram must be."""
    model = make_model().eval()
    seq = torch.randint(1, VOCAB_SIZE, (1, 16))
    with torch.no_grad():
        logits = model(seq)
    assert torch.allclose(logits, logits.transpose(-1, -2), atol=1e-5)


def test_variable_length_runs():
    model = make_model().eval()
    for length in (8, 33, 64):
        seq = torch.randint(1, VOCAB_SIZE, (1, length))
        with torch.no_grad():
            out = model(seq)
        assert out.shape[-1] == length


def test_padding_does_not_change_real_residue_predictions():
    """Appending padding must not alter predictions for the real prefix."""
    torch.manual_seed(0)
    model = make_model().eval()
    seq = torch.randint(1, VOCAB_SIZE, (1, 12))
    padded = torch.cat([seq, torch.full((1, 6), PAD_IDX)], dim=1)
    mask = torch.cat([torch.ones(1, 12), torch.zeros(1, 6)], dim=1).bool()
    with torch.no_grad():
        a = model(seq)
        b = model(padded, mask=mask)[:, :, :12, :12]
    assert torch.allclose(a, b, atol=1e-4)


def test_embedding_mode_accepts_float_input():
    """embed_dim lets a pretrained LM embedding replace the amino-acid table."""
    model = make_model(embed_dim=48).eval()
    feats = torch.randn(1, 15, 48)
    mask = torch.ones(1, 15, dtype=torch.bool)
    with torch.no_grad():
        out = model(feats, mask=mask)
    assert out.shape == (1, 10, 15, 15)


def test_loss_ignores_masked_pairs():
    """Changing predictions on masked-out pairs must not change the loss."""
    logits_a = torch.randn(1, 10, 6, 6, requires_grad=False)
    logits_b = logits_a.clone()
    target = torch.randint(0, 10, (1, 6, 6))
    mask = torch.zeros(1, 6, 6, dtype=torch.bool)
    mask[:, :3, :3] = True

    logits_b[:, :, 4:, 4:] = torch.randn(1, 10, 2, 2)
    assert torch.isclose(
        distogram_loss(logits_a, target, mask),
        distogram_loss(logits_b, target, mask),
    )


def test_loss_is_scale_free_in_length():
    """Mask-weighted mean keeps the loss comparable across crop sizes."""
    torch.manual_seed(0)
    small = distogram_loss(
        torch.zeros(1, 10, 8, 8), torch.zeros(1, 8, 8, dtype=torch.long),
        torch.ones(1, 8, 8, dtype=torch.bool),
    )
    large = distogram_loss(
        torch.zeros(1, 10, 40, 40), torch.zeros(1, 40, 40, dtype=torch.long),
        torch.ones(1, 40, 40, dtype=torch.bool),
    )
    assert torch.isclose(small, large, atol=1e-6)


def test_backward_pass_produces_gradients():
    model = make_model()
    seq = torch.randint(1, VOCAB_SIZE, (2, 24))
    target = torch.randint(0, 10, (2, 24, 24))
    mask = torch.ones(2, 24, 24, dtype=torch.bool)
    loss = distogram_loss(model(seq), target, mask)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_predict_contacts_returns_probabilities():
    model = make_model().eval()
    seq = torch.randint(1, VOCAB_SIZE, (1, 20))
    contacts = model.predict_contacts(seq)
    assert contacts.shape == (1, 20, 20)
    assert (contacts >= 0).all() and (contacts <= 1).all()
