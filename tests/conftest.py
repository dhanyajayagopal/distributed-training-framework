"""Shared pytest fixtures."""
from __future__ import annotations

import socket
from contextlib import closing

import pytest


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def free_port() -> int:
    return _free_port()


@pytest.fixture(autouse=True)
def _disable_cuda_for_tests(monkeypatch):
    """All tests run on CPU to keep them portable across machines."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    yield


@pytest.fixture()
def tmp_work_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path
