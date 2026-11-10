"""Ascend-only dynamic rows and gradient-demand regressions.

Small controlled launch tiles exercise all six kernel bodies, independently of
hardware-dependent UB heuristics. Existing RMSNorm tests cover natural dispatch.
"""

import importlib

import pytest
import torch

from liger_kernel.ops.backends._ascend.dispatch_common import DEFAULT_TOLERANCES
from liger_kernel.testing import pytorch_reference_rms_norm
from liger_kernel.utils import infer_device


@pytest.fixture
def ascend_rms():
    if infer_device() != "npu":
        pytest.skip("requires Ascend NPU and triton-ascend")
    return importlib.import_module("liger_kernel.ops.backends._ascend.ops.rms_norm")


def record_launches(module, monkeypatch):
    calls = []

    class Spy:
        def __init__(self, kernel, name):
            self.kernel = kernel
            self.name = name

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def run(**kwargs):
                compiled = launch(**kwargs)
                calls.append((self.name, compiled.hash, kwargs["n_rows"] % 16 == 0))
                return compiled

            return run

    for direction in ["forward", "backward"]:
        for path in ["row", "fused_2d", "tiled"]:
            name = f"_rms_norm_{direction}_{path}"
            monkeypatch.setattr(module, name, Spy(getattr(module, name), name))
    return calls


@pytest.mark.parametrize(
    "path,tile", [("row", (512, 1, True)), ("fused_2d", (512, 4, True)), ("tiled", (256, 1, False))]
)
@pytest.mark.parametrize(
    "dtype,casting_mode", [(torch.bfloat16, "gemma"), (torch.float32, "llama"), (torch.float32, "none")]
)
def test_ascend_dynamic_rows(ascend_rms, monkeypatch, path, tile, dtype, casting_mode):
    calls = record_launches(ascend_rms, monkeypatch)
    monkeypatch.setattr(ascend_rms, "_fused_tile", lambda *args: tile)
    # Cover a row-block boundary and both directions across the alignment classes.
    for rows in [16, 17, 31, 32]:
        x = torch.randn(rows, 512, device="npu", dtype=dtype).requires_grad_()
        w = torch.randn(512, device="npu", dtype=dtype, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        rw = w.detach().clone().requires_grad_()
        dy = torch.randn_like(x)
        y = ascend_rms.LigerRMSNormFunction.apply(x, w, 1e-6, 0.0, casting_mode, False)
        ref = pytorch_reference_rms_norm(rx, rw, 1e-6, casting_mode=casting_mode)
        grads = torch.autograd.grad(y, (x, w), dy)
        refs = torch.autograd.grad(ref, (rx, rw), dy)
        tol = DEFAULT_TOLERANCES[dtype]
        for index, (actual, expected) in enumerate(zip((y, *grads), (ref, *refs))):
            suffix = "fwd" if index == 0 else "bwd"
            assert actual.dtype == expected.dtype
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, atol=tol[f"atol_{suffix}"], rtol=tol[f"rtol_{suffix}"])

    assert {c[0] for c in calls} == {f"_rms_norm_forward_{path}", f"_rms_norm_backward_{path}"}
    # This checks artifact identity, rather than counting backend compilations.
    for name in {c[0] for c in calls}:
        # Older runtimes may retain alignment classes for runtime integers.
        for aligned in (False, True):
            assert len({c[1] for c in calls if c[0] == name and c[2] == aligned}) == 1


def test_ascend_row_specialization_contract(ascend_rms):
    for direction in ["forward", "backward"]:
        for path in ["row", "fused_2d", "tiled"]:
            kernel = getattr(ascend_rms, f"_rms_norm_{direction}_{path}")
            param = next(p for p in kernel.params if p.name == "n_rows")
            assert not param.is_constexpr


def test_ascend_public_dispatch_accepts_noncontiguous_input(ascend_rms, monkeypatch):
    from liger_kernel.functional import rms_norm

    calls = record_launches(ascend_rms, monkeypatch)
    x = torch.randn(17, 576, device="npu")[:, :512].requires_grad_()
    assert not x.is_contiguous()
    w = torch.ones(512, device="npu")
    y = rms_norm(x, w, eps=1e-6, casting_mode="gemma", in_place=False, impl="ascend-triton")
    rx = x.detach().clone().requires_grad_()
    ref = pytorch_reference_rms_norm(rx, w, 1e-6, casting_mode="gemma")
    tol = DEFAULT_TOLERANCES[x.dtype]
    torch.testing.assert_close(y, ref, atol=tol["atol_fwd"], rtol=tol["rtol_fwd"])
    torch.testing.assert_close(
        torch.autograd.grad(y.sum(), x)[0],
        torch.autograd.grad(ref.sum(), rx)[0],
        atol=tol["atol_bwd"],
        rtol=tol["rtol_bwd"],
    )
    assert any("forward" in c[0] for c in calls)
    assert any("backward" in c[0] for c in calls)
