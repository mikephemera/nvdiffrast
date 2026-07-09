from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest


pytestmark = [pytest.mark.musa, pytest.mark.foundationpose]


def _import_dr():
    import nvdiffrast.torch as dr

    return dr


def _make_context(dr, device: str):
    try:
        return dr.RasterizeCudaContext(device)
    except RuntimeError as exc:
        if "invalid device function" in str(exc):
            pytest.fail(
                "nvdiffrast MUSA extension was built for an incompatible MUSA arch. "
                "Rebuild/reinstall with a matching TORCH_MUSA_ARCH_LIST. "
                f"Diagnostic: {_extension_diagnostic()}. "
                f"Original error: {exc}"
            )
        raise


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


def _call_render(foundationpose_utils, **kwargs):
    try:
        return foundationpose_utils.nvdiffrast_render(**kwargs)
    except RuntimeError as exc:
        if "invalid device function" in str(exc):
            pytest.fail(
                "FoundationPose render hit an nvdiffrast MUSA invalid-device-function error. "
                "Rebuild/reinstall nvdiffrast_musa with a matching TORCH_MUSA_ARCH_LIST. "
                f"Diagnostic: {_extension_diagnostic()}. "
                f"Original error: {exc}"
            )
        raise


def _make_square_mesh(textured: bool):
    import trimesh
    from PIL import Image

    vertices = np.array(
        [
            [-0.45, -0.45, 0.0],
            [0.45, -0.45, 0.0],
            [0.45, 0.45, 0.0],
            [-0.45, 0.45, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if textured:
        uv = np.array([[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]], dtype=np.float32)
        tex = np.array(
            [
                [[255, 40, 40], [40, 255, 40], [40, 40, 255], [255, 255, 40]],
                [[255, 80, 80], [80, 255, 80], [80, 80, 255], [255, 255, 80]],
                [[255, 120, 120], [120, 255, 120], [120, 120, 255], [255, 255, 120]],
                [[255, 160, 160], [160, 255, 160], [160, 160, 255], [255, 255, 160]],
            ],
            dtype=np.uint8,
        )
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=Image.fromarray(tex))
    else:
        colors = np.array(
            [[255, 40, 40, 255], [40, 255, 40, 255], [40, 40, 255, 255], [255, 255, 40, 255]],
            dtype=np.uint8,
        )
        mesh.visual.vertex_colors = colors

    return mesh


def _camera_inputs(torch_mod, device: str, batch_size: int = 1):
    import torch

    height = 64
    width = 64
    k_mat = np.array(
        [[58.0, 0.0, width / 2.0], [0.0, 58.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    poses = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    poses[:, 2, 3] = 1.8
    if batch_size > 1:
        poses[1, 0, 3] = 0.08
        poses[1, 1, 3] = -0.06
    return height, width, k_mat, poses.to(device)


def _assert_render_outputs(torch_mod, color, depth, normal, batch_size: int, height: int, width: int):
    assert color.shape == (batch_size, height, width, 3)
    assert depth.shape == (batch_size, height, width)
    assert normal is not None
    assert normal.shape == (batch_size, height, width, 3)
    assert color.device.type == "musa"
    assert depth.device.type == "musa"
    assert normal.device.type == "musa"

    color_cpu = color.detach().cpu()
    depth_cpu = depth.detach().cpu()
    normal_cpu = normal.detach().cpu()
    assert torch_mod.isfinite(color_cpu).all().item()
    assert torch_mod.isfinite(depth_cpu).all().item()
    assert torch_mod.isfinite(normal_cpu).all().item()

    mask = color_cpu.abs().sum(dim=-1) > 1e-6
    assert mask.sum().item() > 0
    assert depth_cpu[mask].mean().item() > 0.1
    normal_norm = normal_cpu[mask].norm(dim=-1)
    assert normal_norm.mean().item() > 0.5


def _assert_mask_iou(mask_a, mask_b, min_iou: float, name: str):
    intersection = mask_a & mask_b
    union = mask_a | mask_b
    union_count = int(union.sum().item())
    intersection_count = int(intersection.sum().item())
    assert union_count > 0, f"{name}: empty union mask"
    iou = intersection_count / union_count
    assert iou >= min_iou, (
        f"{name}: mask IoU {iou:.4f} is below {min_iou:.4f}; "
        f"intersection={intersection_count}, union={union_count}"
    )
    return intersection


def _assert_mean_abs_diff(name: str, actual, expected, mask, max_mean_abs: float, max_abs: float | None = None):
    selected = (actual - expected).abs()[mask]
    assert selected.numel() > 0, f"{name}: empty comparison mask"
    mean_abs = selected.mean().item()
    worst_abs = selected.max().item()
    max_abs_ok = max_abs is None or worst_abs <= max_abs
    assert mean_abs <= max_mean_abs and max_abs_ok, (
        f"{name}: diff too large; mean_abs={mean_abs:.6g} "
        f"(limit {max_mean_abs:.6g}), max_abs={worst_abs:.6g}"
        + ("" if max_abs is None else f" (limit {max_abs:.6g})")
    )


def _assert_first_pose_matches_single_render(
    color_batch,
    depth_batch,
    normal_batch,
    color_single,
    depth_single,
    normal_single,
):
    color_first = color_batch[:1].detach().cpu()
    depth_first = depth_batch[:1].detach().cpu()
    normal_first = normal_batch[:1].detach().cpu()
    color_single = color_single.detach().cpu()
    depth_single = depth_single.detach().cpu()
    normal_single = normal_single.detach().cpu()

    mask_first = depth_first > 0
    mask_single = depth_single > 0
    overlap = _assert_mask_iou(mask_first, mask_single, min_iou=0.95, name="first-pose batch vs single")

    _assert_mean_abs_diff("depth", depth_first, depth_single, overlap, max_mean_abs=1e-3, max_abs=1e-2)
    _assert_mean_abs_diff("normal", normal_first, normal_single, overlap, max_mean_abs=2e-3, max_abs=2e-2)
    _assert_mean_abs_diff("color", color_first, color_single, overlap, max_mean_abs=3e-2)


@pytest.mark.parametrize("textured", [False, True])
def test_foundationpose_nvdiffrast_render_single_pose_contract(
    musa_runtime,
    foundationpose_utils,
    cuda_factory_alias_to_musa,
    textured: bool,
):
    dr = _import_dr()
    torch_mod = musa_runtime.torch
    mesh = _make_square_mesh(textured=textured)
    mesh_tensors = foundationpose_utils.make_mesh_tensors(mesh, device=musa_runtime.device)
    height, width, k_mat, poses = _camera_inputs(torch_mod, musa_runtime.device, batch_size=1)
    glctx = _make_context(dr, musa_runtime.device)

    color, depth, normal = _call_render(
        foundationpose_utils,
        K=k_mat,
        H=height,
        W=width,
        ob_in_cams=poses,
        glctx=glctx,
        context="cuda",
        get_normal=True,
        mesh_tensors=mesh_tensors,
        output_size=np.asarray([height, width]),
        use_light=True,
        extra={},
    )
    torch_mod.musa.synchronize()

    _assert_render_outputs(torch_mod, color, depth, normal, 1, height, width)


@pytest.mark.parametrize("textured", [False, True])
def test_foundationpose_nvdiffrast_render_batch_bbox_matches_single_first_pose(
    musa_runtime,
    foundationpose_utils,
    cuda_factory_alias_to_musa,
    textured: bool,
):
    dr = _import_dr()
    torch_mod = musa_runtime.torch
    mesh = _make_square_mesh(textured=textured)
    mesh_tensors = foundationpose_utils.make_mesh_tensors(mesh, device=musa_runtime.device)
    height, width, k_mat, poses = _camera_inputs(torch_mod, musa_runtime.device, batch_size=2)
    output_size = np.asarray([32, 32])
    import torch

    bbox = torch.tensor(
        [[8.0, 8.0, 56.0, 56.0], [8.0, 8.0, 56.0, 56.0]],
        dtype=torch.float32,
    ).to(musa_runtime.device)

    glctx_batch = _make_context(dr, musa_runtime.device)
    color_batch, depth_batch, normal_batch = _call_render(
        foundationpose_utils,
        K=k_mat,
        H=height,
        W=width,
        ob_in_cams=poses,
        glctx=glctx_batch,
        context="cuda",
        get_normal=True,
        mesh_tensors=mesh_tensors,
        bbox2d=bbox,
        output_size=output_size,
        use_light=True,
        extra={},
    )

    glctx_single = _make_context(dr, musa_runtime.device)
    color_single, depth_single, normal_single = _call_render(
        foundationpose_utils,
        K=k_mat,
        H=height,
        W=width,
        ob_in_cams=poses[:1],
        glctx=glctx_single,
        context="cuda",
        get_normal=True,
        mesh_tensors=mesh_tensors,
        bbox2d=bbox[:1],
        output_size=output_size,
        use_light=True,
        extra={},
    )
    torch_mod.musa.synchronize()

    out_h, out_w = int(output_size[0]), int(output_size[1])
    _assert_render_outputs(torch_mod, color_batch, depth_batch, normal_batch, 2, out_h, out_w)
    _assert_render_outputs(torch_mod, color_single, depth_single, normal_single, 1, out_h, out_w)
    _assert_first_pose_matches_single_render(
        color_batch,
        depth_batch,
        normal_batch,
        color_single,
        depth_single,
        normal_single,
    )
