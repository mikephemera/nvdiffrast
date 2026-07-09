from __future__ import annotations

import importlib
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class MusaRuntime:
    torch: Any
    device: str
    index: int


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("nvdiffrast-musa")
    group.addoption(
        "--musa-device",
        action="store",
        default="musa:0",
        help="MUSA device used by runtime tests, e.g. musa:0. Legacy cuda:0 is accepted.",
    )
    group.addoption(
        "--require-musa",
        action="store_true",
        help="Fail instead of skipping when no usable MUSA runtime is available.",
    )
    group.addoption(
        "--run-foundationpose",
        action="store_true",
        help="Run FoundationPose render-path contract tests.",
    )
    group.addoption(
        "--foundationpose-root",
        action="store",
        default=str(Path(__file__).resolve().parents[2] / "FoundationPose_musa"),
        help="Path to the FoundationPose checkout used by contract tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "musa: requires a working torch_musa runtime")
    config.addinivalue_line(
        "markers",
        "foundationpose: exercises the FoundationPose nvdiffrast render contract",
    )


def _normalize_musa_device(device: Any) -> str:
    text = str(device)
    if text == "cuda":
        return "musa:0"
    if text.startswith("cuda:"):
        return "musa:" + text.split(":", 1)[1]
    if text == "musa":
        return "musa:0"
    return text


def _device_index(device: str) -> int:
    if ":" not in device:
        return 0
    return int(device.split(":", 1)[1])


def _fail_or_skip(request: pytest.FixtureRequest, msg: str) -> None:
    if request.config.getoption("--require-musa"):
        pytest.fail(msg)
    pytest.skip(msg)


def _musa_runtime_summary(torch: Any, device: str) -> str:
    parts = [f"device={device}"]
    try:
        parts.append(f"name={torch.musa.get_device_name(device)}")
    except Exception:
        pass
    try:
        parts.append(f"capability={torch.musa.get_device_capability(device)}")
    except Exception:
        pass
    try:
        parts.append(f"torch_musa_arch_list={torch.musa.get_arch_list()}")
    except Exception:
        pass
    if os.environ.get("TORCH_MUSA_ARCH_LIST"):
        parts.append(f"TORCH_MUSA_ARCH_LIST={os.environ['TORCH_MUSA_ARCH_LIST']}")
    return ", ".join(parts)


def _runtime_musa_arch(torch: Any, device: str) -> str | None:
    try:
        capability = torch.musa.get_device_capability(device)
    except Exception:
        return None
    return f"{capability[0]}{capability[1]}"


def _extension_musa_archs() -> tuple[Path | None, list[str]]:
    try:
        module = importlib.import_module("_nvdiffrast_c")
    except Exception:
        return None, []

    path = Path(module.__file__).resolve()
    try:
        data = path.read_bytes()
    except OSError:
        return path, []
    archs = sorted({match.decode("ascii") for match in re.findall(rb"musa-mtgpu-mt-musa--mp_([0-9]+)", data)})
    return path, archs


def _validate_extension_arch(request: pytest.FixtureRequest, torch: Any, device: str) -> None:
    runtime_arch = _runtime_musa_arch(torch, device)
    extension_path, extension_archs = _extension_musa_archs()
    if runtime_arch is None or not extension_archs:
        return
    if runtime_arch in extension_archs:
        return

    _fail_or_skip(
        request,
        "nvdiffrast MUSA extension does not contain code for the selected device arch. "
        f"device={device}, runtime_arch={runtime_arch}, extension={extension_path}, "
        f"embedded_archs={extension_archs}, TORCH_MUSA_ARCH_LIST={os.environ.get('TORCH_MUSA_ARCH_LIST')}. "
        "Rebuild with TORCH_MUSA_ARCH_LIST including the runtime arch.",
    )


def _ensure_stub_module(name: str, *, package: bool = False) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if package and not hasattr(module, "__path__"):
        module.__path__ = []
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = _ensure_stub_module(parent_name, package=True)
        setattr(parent, child_name, module)
    return module


def _is_importable(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _unused_foundationpose_dependency(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("FoundationPose render contract touched an import stub for an unused dependency")


def _install_foundationpose_render_import_stubs() -> None:
    """Stub Utils.py imports that are irrelevant to nvdiffrast_render()."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nvdiffrast-musa-matplotlib")

    if not _is_importable("trimesh"):
        class TextureVisuals:
            def __init__(self, uv=None, image=None, material=None):
                self.uv = uv
                self.material = material if material is not None else types.SimpleNamespace(image=image)

        class Trimesh:
            def __init__(self, vertices, faces, process=False):
                del process
                import numpy as np

                self.vertices = vertices
                self.faces = faces
                self.visual = types.SimpleNamespace(vertex_colors=None)
                self.vertex_normals = np.tile(
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(vertices), 1)
                )

        trimesh = _ensure_stub_module("trimesh", package=True)
        visual = _ensure_stub_module("trimesh.visual", package=True)
        texture = _ensure_stub_module("trimesh.visual.texture")
        texture.TextureVisuals = TextureVisuals
        visual.texture = texture
        trimesh.visual = visual
        trimesh.Trimesh = Trimesh

    for module_name in ("imageio", "pandas", "open3d", "cv2", "matplotlib", "matplotlib.pyplot"):
        if not _is_importable(module_name):
            _ensure_stub_module(module_name, package=module_name in {"matplotlib"})

    if not _is_importable("transformations"):
        transformations = _ensure_stub_module("transformations")
        transformations.__all__ = []

    if not _is_importable("ruamel.yaml"):
        ruamel = _ensure_stub_module("ruamel", package=True)
        yaml_module = _ensure_stub_module("ruamel.yaml")
        yaml_module.YAML = lambda: types.SimpleNamespace()
        ruamel.yaml = yaml_module

    if not _is_importable("pytorch3d.transforms"):
        transforms = _ensure_stub_module("pytorch3d.transforms")
        for name in (
            "so3_log_map",
            "so3_exp_map",
            "se3_exp_map",
            "se3_log_map",
            "matrix_to_axis_angle",
            "matrix_to_euler_angles",
            "euler_angles_to_matrix",
            "rotation_6d_to_matrix",
        ):
            setattr(transforms, name, _unused_foundationpose_dependency)

    if not _is_importable("pytorch3d.renderer"):
        renderer = _ensure_stub_module("pytorch3d.renderer", package=True)
        for name in (
            "FoVPerspectiveCameras",
            "PerspectiveCameras",
            "look_at_view_transform",
            "look_at_rotation",
            "RasterizationSettings",
            "MeshRenderer",
            "MeshRasterizer",
            "BlendParams",
            "SoftSilhouetteShader",
            "HardPhongShader",
            "PointLights",
            "TexturesVertex",
        ):
            setattr(renderer, name, _unused_foundationpose_dependency)

    mesh = _ensure_stub_module("pytorch3d.renderer.mesh", package=True)
    if not _is_importable("pytorch3d.renderer.mesh.rasterize_meshes"):
        rasterize_meshes = _ensure_stub_module("pytorch3d.renderer.mesh.rasterize_meshes")
        rasterize_meshes.barycentric_coordinates = _unused_foundationpose_dependency
        mesh.rasterize_meshes = rasterize_meshes
    if not _is_importable("pytorch3d.renderer.mesh.shader"):
        shader = _ensure_stub_module("pytorch3d.renderer.mesh.shader")
        shader.SoftDepthShader = _unused_foundationpose_dependency
        shader.HardFlatShader = _unused_foundationpose_dependency
        mesh.shader = shader
    if not _is_importable("pytorch3d.renderer.mesh.textures"):
        textures = _ensure_stub_module("pytorch3d.renderer.mesh.textures")
        textures.Textures = _unused_foundationpose_dependency
        mesh.textures = textures
    if not _is_importable("pytorch3d.structures"):
        structures = _ensure_stub_module("pytorch3d.structures")
        structures.Meshes = _unused_foundationpose_dependency

    if not _is_importable("scipy.interpolate"):
        interpolate = _ensure_stub_module("scipy.interpolate")
        interpolate.griddata = _unused_foundationpose_dependency
    if not _is_importable("scipy.spatial"):
        spatial = _ensure_stub_module("scipy.spatial")
        spatial.cKDTree = _unused_foundationpose_dependency


@pytest.fixture(scope="session")
def musa_runtime(request: pytest.FixtureRequest) -> MusaRuntime:
    try:
        torch = importlib.import_module("torch")
        importlib.import_module("torch_musa")
    except Exception as exc:  # pragma: no cover - depends on target host.
        msg = f"torch_musa is not importable: {exc}"
        if request.config.getoption("--require-musa"):
            pytest.fail(msg)
        pytest.skip(msg)

    if not hasattr(torch, "musa"):
        msg = "torch_musa did not register torch.musa"
        if request.config.getoption("--require-musa"):
            pytest.fail(msg)
        pytest.skip(msg)

    try:
        is_available = bool(torch.musa.is_available())
        device_count = int(torch.musa.device_count())
    except Exception as exc:  # pragma: no cover - depends on target host.
        _fail_or_skip(request, f"torch.musa runtime check failed: {exc}")

    if not is_available or device_count <= 0:
        _fail_or_skip(
            request, f"no usable MUSA device found (is_available={is_available}, count={device_count})"
        )

    device = _normalize_musa_device(request.config.getoption("--musa-device"))
    index = _device_index(device)
    if index < 0 or index >= device_count:
        _fail_or_skip(request, f"requested {device}, but only {device_count} MUSA device(s) are visible")

    torch.musa.set_device(device)
    try:
        sample = torch.tensor([1.0, 2.0, 3.0]).to(device)
        result = (sample + 1.0).cpu()
        torch.musa.synchronize()
        if result.tolist() != [2.0, 3.0, 4.0]:
            raise RuntimeError(f"unexpected MUSA smoke result: {result}")
    except Exception as exc:
        _fail_or_skip(
            request,
            "MUSA tensor smoke test failed before nvdiffrast ran. "
            f"{_musa_runtime_summary(torch, device)}. Original error: {exc}",
        )
    _validate_extension_arch(request, torch, device)
    return MusaRuntime(torch=torch, device=device, index=index)


@pytest.fixture(scope="session")
def foundationpose_utils(request: pytest.FixtureRequest):
    if not request.config.getoption("--run-foundationpose"):
        pytest.skip("pass --run-foundationpose to run FoundationPose render contract tests")

    root = Path(request.config.getoption("--foundationpose-root")).resolve()
    if not root.is_dir():
        pytest.fail(f"FoundationPose root does not exist: {root}")

    _install_foundationpose_render_import_stubs()
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module("Utils")
    except ModuleNotFoundError as exc:
        pytest.skip(
            "FoundationPose render contract dependencies are missing "
            f"({exc.name}). Install FoundationPose requirements to enable these tests."
        )
    except Exception as exc:
        pytest.fail(f"failed to import FoundationPose Utils.py from {root}: {exc}")


@pytest.fixture
def cuda_factory_alias_to_musa(monkeypatch: pytest.MonkeyPatch, musa_runtime: MusaRuntime) -> None:
    """Route FoundationPose's hard-coded CUDA tensor factories to MUSA for tests."""

    torch = musa_runtime.torch
    musa_device = torch.device(musa_runtime.device)

    def normalize_device(device: Any) -> Any:
        if device is None:
            return None
        if isinstance(device, str):
            if device == "cuda" or device.startswith("cuda:"):
                return musa_device
            return device
        if isinstance(device, torch.device) and device.type == "cuda":
            return musa_device
        return device

    def wrap_factory(name: str) -> None:
        original = getattr(torch, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            redirected = False
            if "device" in kwargs:
                requested = normalize_device(kwargs["device"])
                if requested == musa_device:
                    redirected = True
                    kwargs["device"] = torch.device("cpu")
                else:
                    kwargs["device"] = requested
            out = original(*args, **kwargs)
            return out.to(musa_device) if redirected else out

        monkeypatch.setattr(torch, name, wrapped)

    for factory_name in ("arange", "as_tensor", "empty", "eye", "full", "ones", "tensor", "zeros"):
        wrap_factory(factory_name)
