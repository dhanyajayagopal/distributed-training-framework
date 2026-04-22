"""Unit tests for CheckpointManager (no distributed required)."""
from __future__ import annotations

import torch
import torch.nn as nn

from trainer.checkpoint import CheckpointManager


def _make():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, opt


def test_save_and_load_roundtrip(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model, opt = _make()

    for _ in range(3):
        x = torch.randn(8, 4)
        y = torch.randn(8, 2)
        loss = nn.MSELoss()(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    saved = mgr.save(model=model, optimizer=opt, step=3, rank=0)
    assert saved is not None
    assert saved.exists()

    before = {k: v.clone() for k, v in model.state_dict().items()}
    fresh_model, fresh_opt = _make()
    info = mgr.load(model=fresh_model, optimizer=fresh_opt, path=saved)

    assert info["step"] == 3
    for k, v in fresh_model.state_dict().items():
        assert torch.allclose(v, before[k])


def test_latest_pointer(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model, opt = _make()
    mgr.save(model=model, optimizer=opt, step=1, rank=0)
    mgr.save(model=model, optimizer=opt, step=5, rank=0)

    fresh_model, fresh_opt = _make()
    info = mgr.load(model=fresh_model, optimizer=fresh_opt, path="latest")
    assert info["step"] == 5


def test_missing_file_raises(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model, opt = _make()
    try:
        mgr.load(model=model, optimizer=opt, path=tmp_path / "nope.pt")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_atomic_save_leaves_no_tmp(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model, opt = _make()
    mgr.save(model=model, optimizer=opt, step=0, rank=0)
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())


def test_non_zero_rank_does_not_write(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model, opt = _make()
    path = mgr.save(model=model, optimizer=opt, step=7, rank=1)
    assert path is None
    assert not any(p.suffix == ".pt" for p in tmp_path.iterdir())
