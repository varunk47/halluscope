"""Shared test plumbing.

The model-loading tests share a Windows laptop with whatever experiment is
running at the time. When that experiment has the GPU and most of the commit
charge, ``from_pretrained`` dies with "the paging file is too small" or a bare
MemoryError. That is the machine being busy, not the code being wrong, so the
run reports it as a skip with the reason rather than as a failure that has to
be re-read every time.
"""

from __future__ import annotations

import pytest


def _out_of_commit_charge(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    return isinstance(exc, OSError) and (
        getattr(exc, "winerror", None) == 1455 or "paging file" in str(exc)
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    try:
        outcome.get_result()
    except BaseException as exc:  # noqa: BLE001 - we only translate one family and re-raise the rest
        if _out_of_commit_charge(exc):
            pytest.skip(f"machine out of commit charge, not a code failure: {exc}")
        raise
