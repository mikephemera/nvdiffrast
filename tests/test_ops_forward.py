from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.musa


def _import_dr():
    import nvdiffrast.torch as dr

    return dr


def _to_device(tensor: torch.Tensor, device: str, requires_grad: bool = False) -> torch.Tensor:
    tensor = tensor.to(device)
    if requires_grad:
        tensor.requires_grad_()
    return tensor


def _extension_diagnostic() -> str:
    try:
        import _nvdiffrast_c
    except Exception as exc:
        return f"extension_import_error={exc}"

    path = Path(_nvdiffrast_c.__file__).resolve()
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"extension={path}, arch_read_error={exc}"
    archs = sorted({match.decode("ascii") for match in re.findall(rb"musa-mtgpu-mt-musa--mp_([0-9]+)", data)})
    return (
        f"extension={path}, embedded_archs={archs or 'unknown'}, "
        f"TORCH_MUSA_ARCH_LIST={os.environ.get('TORCH_MUSA_ARCH_LIST')}"
    )


def _fail_on_invalid_device_function(exc: RuntimeError) -> None:
    if "invalid device function" in str(exc):
        pytest.fail(
            "nvdiffrast MUSA kernel failed with invalid device function. "
            "This usually means the extension does not contain code for the runtime arch, "
            "or the migrated kernel did not compile into valid device code. "
            "Reinstall on the target machine with a matching TORCH_MUSA_ARCH_LIST, e.g. "
            "TORCH_MUSA_ARCH_LIST=<capability> python -m pip install -e . "
            "--no-build-isolation --force-reinstall. "
            f"Diagnostic: {_extension_diagnostic()}. "
            f"Original error: {exc}"
        )


def _dr_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as exc:
        _fail_on_invalid_device_function(exc)
        raise


def _make_context(dr, device: str):
    try:
        return dr.RasterizeCudaContext(device)
    except RuntimeError as exc:
        _fail_on_invalid_device_function(exc)
        raise


def _triangle_scene(torch_mod, device: str, requires_grad: bool = False):
    pos = torch.tensor(
        [[[-0.75, -0.65, 0.0, 1.0], [0.70, -0.55, 0.0, 1.0], [-0.55, 0.70, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    attr = torch.tensor(
        [[[1.0, 0.0, 0.2], [0.1, 1.0, 0.3], [0.0, 0.2, 1.0]]],
        dtype=torch.float32,
    )
    return _to_device(pos, device, requires_grad), _to_device(tri, device), _to_device(attr, device)


def _rectangle_scene(torch_mod, device: str):
    base = torch.tensor(
        [
            [-0.60, -0.55, 0.10, 1.0],
            [0.55, -0.55, 0.10, 1.0],
            [0.55, 0.55, 0.10, 1.0],
            [-0.60, 0.55, 0.10, 1.0],
        ],
        dtype=torch.float32,
    )
    shifted = base.clone()
    shifted[:, 0] += 0.18
    shifted[:, 1] -= 0.08
    shifted[:, 2] += 0.05
    pos = torch.stack([base, shifted], dim=0)
    tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    attr = torch.tensor(
        [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9], [0.9, 0.9, 0.1]],
        dtype=torch.float32,
    )
    attr_batched = torch.stack([attr, attr.flip(0)], dim=0)
    return (
        _to_device(pos, device),
        _to_device(tri, device),
        _to_device(attr, device),
        _to_device(attr_batched, device),
    )


def _triangle_barycentric_reference(pos: torch.Tensor, resolution: tuple[int, int]):
    height, width = resolution
    pos = pos.detach().cpu().to(torch.float64)
    xy = pos[:, :2] / pos[:, 3:4]
    x0, y0 = xy[0]
    x1, y1 = xy[1]
    x2, y2 = xy[2]

    rows, cols = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    x = (cols + 0.5) * (2.0 / width) - 1.0
    y = (rows + 0.5) * (2.0 / height) - 1.0

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    w2 = 1.0 - w0 - w1
    bary = torch.stack([w0, w1, w2], dim=-1)
    mask = torch.all(bary >= -1e-7, dim=-1)
    return bary, mask


def _interpolate_triangle_reference(
    pos: torch.Tensor, attr: torch.Tensor, resolution: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bary, mask = _triangle_barycentric_reference(pos, resolution)
    attr = attr.detach().cpu().to(torch.float64)
    out = torch.einsum("hwv,vc->hwc", bary, attr)
    out = torch.where(mask[..., None], out, torch.zeros_like(out))
    return out.to(torch.float32), mask, bary


def _fetch_texel(tex: torch.Tensor, batch: int, y: int, x: int, boundary_mode: str) -> torch.Tensor:
    height, width = tex.shape[1:3]
    if boundary_mode == "wrap":
        y %= height
        x %= width
    elif boundary_mode == "clamp":
        y = min(max(y, 0), height - 1)
        x = min(max(x, 0), width - 1)
    elif boundary_mode == "zero":
        if y < 0 or y >= height or x < 0 or x >= width:
            return torch.zeros(tex.shape[-1], dtype=tex.dtype)
    else:
        raise AssertionError(f"unsupported reference boundary mode: {boundary_mode}")
    return tex[batch, y, x]


def _texture_linear_reference(tex: torch.Tensor, uv: torch.Tensor, boundary_mode: str = "wrap") -> torch.Tensor:
    tex = tex.detach().cpu().to(torch.float64)
    uv = uv.detach().cpu().to(torch.float64)
    batch, out_h, out_w, _ = uv.shape
    tex_batch, tex_h, tex_w, channels = tex.shape
    out = torch.empty((batch, out_h, out_w, channels), dtype=torch.float64)

    for b in range(batch):
        tb = 0 if tex_batch == 1 else b
        for y in range(out_h):
            for x in range(out_w):
                u = float(uv[b, y, x, 0])
                v = float(uv[b, y, x, 1])
                if boundary_mode == "wrap":
                    u -= math.floor(u)
                    v -= math.floor(v)
                px = u * tex_w - 0.5
                py = v * tex_h - 0.5
                if boundary_mode == "clamp":
                    px = min(max(px, 0.0), tex_w - 1.0)
                    py = min(max(py, 0.0), tex_h - 1.0)
                x0 = math.floor(px)
                y0 = math.floor(py)
                x1 = x0 + 1
                y1 = y0 + 1
                wx = px - x0
                wy = py - y0
                c00 = _fetch_texel(tex, tb, y0, x0, boundary_mode)
                c10 = _fetch_texel(tex, tb, y0, x1, boundary_mode)
                c01 = _fetch_texel(tex, tb, y1, x0, boundary_mode)
                c11 = _fetch_texel(tex, tb, y1, x1, boundary_mode)
                out[b, y, x] = (1.0 - wy) * ((1.0 - wx) * c00 + wx * c10) + wy * (
                    (1.0 - wx) * c01 + wx * c11
                )
    return out.to(torch.float32)


def test_context_creation_accepts_musa_and_legacy_cuda_device(musa_runtime):
    dr = _import_dr()

    musa_ctx = _make_context(dr, musa_runtime.device)
    legacy_ctx = _make_context(dr, f"cuda:{musa_runtime.index}")

    assert musa_ctx.cpp_wrapper is not None
    assert legacy_ctx.cpp_wrapper is not None


def test_triangle_rasterize_and_interpolate_match_cpu_reference(musa_runtime):
    dr = _import_dr()
    torch_mod = musa_runtime.torch
    resolution = (40, 40)
    pos, tri, attr = _triangle_scene(torch_mod, musa_runtime.device)
    glctx = _make_context(dr, musa_runtime.device)

    rast, _ = _dr_call(dr.rasterize, glctx, pos, tri, resolution=resolution)
    color, _ = _dr_call(dr.interpolate, attr, rast, tri)
    torch_mod.musa.synchronize()

    rast_cpu = rast.detach().cpu()
    color_cpu = color.detach().cpu()[0]
    expected, cpu_mask, bary = _interpolate_triangle_reference(pos.detach().cpu()[0], attr.detach().cpu()[0], resolution)
    device_mask = rast_cpu[0, ..., 3] > 0
    interior = cpu_mask & (bary.min(dim=-1).values > 0.08)

    assert int(device_mask.sum()) > 0
    assert abs(int(device_mask.sum()) - int(cpu_mask.sum())) <= resolution[0] * 2
    assert int(interior.sum()) > 20
    assert torch.allclose(color_cpu[interior], expected[interior], atol=3e-3, rtol=3e-3)
    assert torch.all(rast_cpu[0, ..., 3][device_mask] == 1)
    assert torch.isfinite(rast_cpu).all()
    assert torch.isfinite(color_cpu).all()


def test_instanced_rectangle_batch_matches_individual_renders(musa_runtime):
    dr = _import_dr()
    torch_mod = musa_runtime.torch
    resolution = (36, 36)
    pos, tri, attr, attr_batched = _rectangle_scene(torch_mod, musa_runtime.device)
    scalar_attr = _to_device(
        torch.tensor([[[0.2], [0.5], [0.8], [1.0]], [[1.0], [0.8], [0.5], [0.2]]], dtype=torch.float32),
        musa_runtime.device,
    )
    glctx = _make_context(dr, musa_runtime.device)

    rast_batch, _ = _dr_call(dr.rasterize, glctx, pos, tri, resolution=resolution)
    color_batch, _ = _dr_call(dr.interpolate, attr_batched, rast_batch, tri)
    broadcast_color, _ = _dr_call(dr.interpolate, attr, rast_batch, tri)
    scalar_batch, _ = _dr_call(dr.interpolate, scalar_attr, rast_batch, tri)
    torch_mod.musa.synchronize()

    assert rast_batch.shape == (2, *resolution, 4)
    assert color_batch.shape == (2, *resolution, 3)
    assert broadcast_color.shape == (2, *resolution, 3)
    assert scalar_batch.shape == (2, *resolution, 1)
    rast_batch_cpu = rast_batch.detach().cpu()
    color_batch_cpu = color_batch.detach().cpu()
    broadcast_color_cpu = broadcast_color.detach().cpu()
    scalar_batch_cpu = scalar_batch.detach().cpu()
    assert torch.isfinite(rast_batch_cpu).all()
    assert torch.isfinite(color_batch_cpu).all()
    assert torch.isfinite(broadcast_color_cpu).all()
    assert torch.isfinite(scalar_batch_cpu).all()
    assert ((rast_batch_cpu[..., 3] > 0).sum(dim=(1, 2)).min() > 0).item()

    for batch_idx in range(2):
        rast_single, _ = _dr_call(
            dr.rasterize, glctx, pos[batch_idx : batch_idx + 1], tri, resolution=resolution
        )
        color_single, _ = _dr_call(
            dr.interpolate, attr_batched[batch_idx : batch_idx + 1], rast_single, tri
        )
        broadcast_color_single, _ = _dr_call(dr.interpolate, attr, rast_single, tri)
        scalar_single, _ = _dr_call(
            dr.interpolate, scalar_attr[batch_idx : batch_idx + 1], rast_single, tri
        )
        torch_mod.musa.synchronize()
        assert torch.allclose(
            rast_batch_cpu[batch_idx : batch_idx + 1], rast_single.detach().cpu(), atol=0, rtol=0
        )
        assert torch.allclose(
            color_batch_cpu[batch_idx : batch_idx + 1], color_single.detach().cpu(), atol=1e-6, rtol=1e-6
        )
        assert torch.allclose(
            broadcast_color_cpu[batch_idx : batch_idx + 1],
            broadcast_color_single.detach().cpu(),
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            scalar_batch_cpu[batch_idx : batch_idx + 1],
            scalar_single.detach().cpu(),
            atol=1e-6,
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    ("name", "attr"),
    [
        ("xyz", [[[0.0, 0.0, 1.2], [0.2, 0.0, 1.4], [0.0, 0.3, 1.6]]]),
        ("normal", [[[0.0, 0.0, 1.0], [0.1, 0.0, 0.95], [0.0, 0.2, 0.9]]]),
        ("vertex_color", [[1.0, 0.2, 0.1], [0.1, 1.0, 0.2], [0.2, 0.1, 1.0]]),
        ("uv", [[0.15, 0.20], [0.85, 0.25], [0.30, 0.90]]),
        ("diffuse_scalar", [[[0.2], [0.6], [1.0]]]),
    ],
)
def test_interpolate_foundationpose_attribute_shapes_and_values(musa_runtime, name, attr):
    dr = _import_dr()
    torch_mod = musa_runtime.torch
    resolution = (34, 34)
    pos, tri, _ = _triangle_scene(torch_mod, musa_runtime.device)
    attr_tensor = _to_device(torch.tensor(attr, dtype=torch.float32), musa_runtime.device)
    glctx = _make_context(dr, musa_runtime.device)

    rast, _ = _dr_call(dr.rasterize, glctx, pos, tri, resolution=resolution)
    out, out_da = _dr_call(dr.interpolate, attr_tensor, rast, tri)
    torch_mod.musa.synchronize()

    attr_ref = attr_tensor.detach().cpu()[0] if attr_tensor.ndim == 3 else attr_tensor.detach().cpu()
    expected, cpu_mask, bary = _interpolate_triangle_reference(pos.detach().cpu()[0], attr_ref, resolution)
    interior = cpu_mask & (bary.min(dim=-1).values > 0.10)

    assert out.shape == (1, *resolution, attr_ref.shape[-1]), name
    assert out_da.shape == (1, *resolution, 0), name
    out_cpu = out.detach().cpu()
    assert torch.isfinite(out_cpu).all(), name
    assert int(interior.sum()) > 10, name
    assert torch.allclose(out_cpu[0, interior], expected[interior], atol=3e-3, rtol=3e-3), name


def test_linear_texture_sampling_matches_cpu_reference(musa_runtime):
    dr = _import_dr()
    torch_mod = musa_runtime.torch
    tex_cpu = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.1], [0.2, 0.0, 0.2], [0.5, 0.0, 0.3], [0.8, 0.0, 0.4]],
                [[0.0, 0.3, 0.2], [0.2, 0.3, 0.3], [0.5, 0.3, 0.4], [0.8, 0.3, 0.5]],
                [[0.0, 0.6, 0.3], [0.2, 0.6, 0.4], [0.5, 0.6, 0.5], [0.8, 0.6, 0.6]],
                [[0.0, 0.9, 0.4], [0.2, 0.9, 0.5], [0.5, 0.9, 0.6], [0.8, 0.9, 0.7]],
            ]
        ],
        dtype=torch.float32,
    )
    uv_cpu = torch.tensor(
        [
            [
                [[0.25, 0.25], [0.50, 0.25], [0.75, 0.25]],
                [[0.25, 0.50], [0.50, 0.50], [0.75, 0.50]],
                [[0.25, 0.75], [0.50, 0.75], [0.75, 0.75]],
            ]
        ],
        dtype=torch.float32,
    )
    tex = _to_device(tex_cpu, musa_runtime.device)
    uv = _to_device(uv_cpu, musa_runtime.device)

    out = _dr_call(dr.texture, tex, uv, filter_mode="linear")
    torch_mod.musa.synchronize()

    expected = _texture_linear_reference(tex_cpu, uv_cpu, boundary_mode="wrap")
    assert out.shape == (1, 3, 3, 3)
    out_cpu = out.detach().cpu()
    assert torch.allclose(out_cpu, expected, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(out_cpu).all()


def test_unsupported_paths_raise_clear_errors(musa_runtime):
    dr = _import_dr()
    torch_mod = musa_runtime.torch
    pos, tri, _ = _triangle_scene(torch_mod, musa_runtime.device)
    glctx = _make_context(dr, musa_runtime.device)
    tex = _to_device(torch.ones((1, 4, 4, 3), dtype=torch.float32), musa_runtime.device)
    uv = _to_device(torch.full((1, 2, 2, 2), 0.5, dtype=torch.float32), musa_runtime.device)

    with pytest.raises(NotImplementedError, match="mip"):
        dr.texture(tex, uv, uv_da=_to_device(torch.zeros((1, 2, 2, 4)), musa_runtime.device))
    with pytest.raises(NotImplementedError, match="mipmap"):
        dr.texture(tex, uv, filter_mode="linear-mipmap-linear")
    with pytest.raises(NotImplementedError, match="mipmap"):
        dr.texture_construct_mip(tex)
    with pytest.raises(NotImplementedError, match="antialias"):
        dr.antialias(_to_device(torch.zeros((1, 8, 8, 3)), musa_runtime.device), None, None, None)
    with pytest.raises(NotImplementedError, match="depth peeling"):
        dr.DepthPeeler(glctx, pos, tri, [8, 8])


def test_backward_paths_are_forward_only(musa_runtime):
    dr = _import_dr()
    tex = _to_device(torch.ones((1, 4, 4, 3), dtype=torch.float32), musa_runtime.device, True)
    uv = _to_device(torch.full((1, 2, 2, 2), 0.5, dtype=torch.float32), musa_runtime.device, True)

    out = _dr_call(dr.texture, tex, uv, filter_mode="linear")
    grad = _to_device(torch.ones(out.shape, dtype=out.dtype), musa_runtime.device)
    with pytest.raises(RuntimeError, match="forward inference only"):
        out.backward(grad)
