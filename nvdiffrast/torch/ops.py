# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch_musa
import warnings
import _nvdiffrast_c

_FORWARD_ONLY_ERROR = "nvdiffrast_musa supports forward inference only"

def _normalize_musa_device(device):
    if device is None:
        return None
    if isinstance(device, str):
        if device == 'cuda':
            return 'musa'
        if device.startswith('cuda:'):
            return 'musa:' + device.split(':', 1)[1]
    if isinstance(device, torch.device) and device.type == 'cuda':
        return torch.device('musa' if device.index is None else f'musa:{device.index}')
    return device

#----------------------------------------------------------------------------
# Log level.
#----------------------------------------------------------------------------

def get_log_level():
    '''Get current log level.

    Returns:
      Current log level in nvdiffrast. See `set_log_level()` for possible values.
    '''
    return _nvdiffrast_c.get_log_level()

def set_log_level(level):
    '''Set log level.

    Log levels follow the convention on the C++ side of Torch:
      0 = Info,
      1 = Warning,
      2 = Error,
      3 = Fatal.
    The default log level is 1.

    Args:
      level: New log level as integer. Internal nvdiffrast messages of this 
             severity or higher will be printed, while messages of lower
             severity will be silent.
    '''
    _nvdiffrast_c.set_log_level(level)

#----------------------------------------------------------------------------
# CudaRaster state wrapper.
#----------------------------------------------------------------------------

class RasterizeCudaContext:
    def __init__(self, device=None):
        '''Create a new MUSA rasterizer context.

        The context is deleted and internal storage is released when the object is
        destroyed.

        Args:
          device (Optional): MUSA device on which the context is created. Type can be
                             `torch.device`, string (e.g., `'musa:1'` or legacy
                             `'cuda:1'`), or int. If not specified, context will be
                             created on currently active MUSA device.
        Returns:
          The newly created MUSA rasterizer context.
        '''
        device = _normalize_musa_device(device)
        if device is None:
            musa_device_idx = torch.musa.current_device()
        else:
            with torch.musa.device(device):
                musa_device_idx = torch.musa.current_device()
        self.cpp_wrapper = _nvdiffrast_c.RasterizeCRStateWrapper(musa_device_idx)
        self.active_depth_peeler = None

RasterizeMusaContext = RasterizeCudaContext


#----------------------------------------------------------------------------
# Rasterize.
#----------------------------------------------------------------------------

class _rasterize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raster_ctx, pos, tri, resolution, ranges, grad_db, peeling_idx):
        out, out_db = _nvdiffrast_c.rasterize_fwd_cuda(raster_ctx.cpp_wrapper, pos, tri, resolution, ranges, peeling_idx)
        ctx.save_for_backward(pos, tri, out)
        ctx.saved_grad_db = grad_db
        return out, out_db

    @staticmethod
    def backward(ctx, dy, ddb):
        raise RuntimeError(_FORWARD_ONLY_ERROR)

# Op wrapper.
def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    '''Rasterize triangles.

    All input tensors must be contiguous and reside in GPU memory except for
    the `ranges` tensor that, if specified, has to reside in CPU memory. The
    output tensors will be contiguous and reside in GPU memory.

    Args:
        glctx: Rasterizer context of type `RasterizeCudaContext`.
        pos: Vertex position tensor with dtype `torch.float32`. To enable range
             mode, this tensor should have a 2D shape [num_vertices, 4]. To enable
             instanced mode, use a 3D shape [minibatch_size, num_vertices, 4].
        tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
        resolution: Output resolution as integer tuple (height, width).
        ranges: In range mode, tensor with shape [minibatch_size, 2] and dtype
                `torch.int32`, specifying start indices and counts into `tri`.
                Ignored in instanced mode.
        grad_db: Propagate gradients of image-space derivatives of barycentrics
                 into `pos` in backward pass.

    Returns:
        A tuple of two tensors, both with shape [minibatch_size, height, width, 4].
        The first output tensor contains the main rasterizer output in order
        (u, v, z/w, triangle_id). The second output tensor contains image-space
        derivatives of barycentrics in order (du/dX, du/dY, dv/dX, dv/dY).
    '''
    assert isinstance(glctx, RasterizeCudaContext)
    assert grad_db is True or grad_db is False

    # Sanitize inputs.
    assert isinstance(pos, torch.Tensor) and isinstance(tri, torch.Tensor)
    resolution = tuple(resolution)
    if ranges is None:
        ranges = torch.empty(size=(0, 2), dtype=torch.int32, device='cpu')
    else:
        assert isinstance(ranges, torch.Tensor)

    # Check that context is not currently reserved for depth peeling.
    if glctx.active_depth_peeler is not None:
        raise RuntimeError("Cannot call rasterize() during depth peeling operation, use rasterize_next_layer() instead")

    # Instantiate the function.
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)

#----------------------------------------------------------------------------
# Depth peeler context manager for rasterizing multiple depth layers.
#----------------------------------------------------------------------------

class DepthPeeler:
    def __init__(self, glctx, pos, tri, resolution, ranges=None, grad_db=True):
        '''Create a depth peeler object for rasterizing multiple depth layers.

        Arguments are the same as in `rasterize()`.

        Returns:
          The newly created depth peeler.
        '''
        raise NotImplementedError("nvdiffrast_musa does not support depth peeling")
        assert isinstance(glctx, RasterizeCudaContext)
        assert grad_db is True or grad_db is False

        # Sanitize inputs as usual.
        assert isinstance(pos, torch.Tensor) and isinstance(tri, torch.Tensor)
        resolution = tuple(resolution)
        if ranges is None:
            ranges = torch.empty(size=(0, 2), dtype=torch.int32, device='cpu')
        else:
            assert isinstance(ranges, torch.Tensor)

        # Store all the parameters.
        self.raster_ctx = glctx
        self.pos = pos
        self.tri = tri
        self.resolution = resolution
        self.ranges = ranges
        self.grad_db = grad_db
        self.peeling_idx = None

    def __enter__(self):
        if self.raster_ctx is None:
            raise RuntimeError("Cannot re-enter a terminated depth peeling operation")
        if self.raster_ctx.active_depth_peeler is not None:
            raise RuntimeError("Cannot have multiple depth peelers active simultaneously in a rasterization context")
        self.raster_ctx.active_depth_peeler = self
        self.peeling_idx = 0
        return self

    def __exit__(self, *args):
        assert self.raster_ctx.active_depth_peeler is self
        self.raster_ctx.active_depth_peeler = None
        self.raster_ctx = None # Remove all references to input tensor so they're not left dangling.
        self.pos = None
        self.tri = None
        self.resolution = None
        self.ranges = None
        self.grad_db = None
        self.peeling_idx = None
        return None

    def rasterize_next_layer(self):
        '''Rasterize next depth layer.

        Operation is equivalent to `rasterize()` except that previously reported
        surface points are culled away.

        Returns:
          A tuple of two tensors as in `rasterize()`.
        '''
        assert self.raster_ctx.active_depth_peeler is self
        assert self.peeling_idx >= 0
        result = _rasterize_func.apply(self.raster_ctx, self.pos, self.tri, self.resolution, self.ranges, self.grad_db, self.peeling_idx)
        self.peeling_idx += 1
        return result

#----------------------------------------------------------------------------
# Interpolate.
#----------------------------------------------------------------------------

# Output pixel differentials for at least some attributes.
class _interpolate_func_da(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attr, rast, tri, rast_db, diff_attrs_all, diff_attrs_list):
        out, out_da = _nvdiffrast_c.interpolate_fwd_da(attr, rast, tri, rast_db, diff_attrs_all, diff_attrs_list)
        ctx.save_for_backward(attr, rast, tri, rast_db)
        ctx.saved_misc = diff_attrs_all, diff_attrs_list
        return out, out_da

    @staticmethod
    def backward(ctx, dy, dda):
        raise RuntimeError(_FORWARD_ONLY_ERROR)

# No pixel differential for any attribute.
class _interpolate_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attr, rast, tri):
        out, out_da = _nvdiffrast_c.interpolate_fwd(attr, rast, tri)
        ctx.save_for_backward(attr, rast, tri)
        return out, out_da

    @staticmethod
    def backward(ctx, dy, _):
        raise RuntimeError(_FORWARD_ONLY_ERROR)

# Op wrapper.
def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """Interpolate vertex attributes.

    All input tensors must be contiguous and reside in GPU memory. The output tensors
    will be contiguous and reside in GPU memory.

    Args:
        attr: Attribute tensor with dtype `torch.float32`. 
              Shape is [num_vertices, num_attributes] in range mode, or 
              [minibatch_size, num_vertices, num_attributes] in instanced mode.
              Broadcasting is supported along the minibatch axis.
        rast: Main output tensor from `rasterize()`.
        tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
        rast_db: (Optional) Tensor containing image-space derivatives of barycentrics, 
                 i.e., the second output tensor from `rasterize()`. Enables computing
                 image-space derivatives of attributes.
        diff_attrs: (Optional) List of attribute indices for which image-space
                    derivatives are to be computed. Special value 'all' is equivalent
                    to list [0, 1, ..., num_attributes - 1].

    Returns:
        A tuple of two tensors. The first output tensor contains interpolated
        attributes and has shape [minibatch_size, height, width, num_attributes].
        If `rast_db` and `diff_attrs` were specified, the second output tensor contains
        the image-space derivatives of the selected attributes and has shape
        [minibatch_size, height, width, 2 * len(diff_attrs)]. The derivatives of the
        first selected attribute A will be on channels 0 and 1 as (dA/dX, dA/dY), etc.
        Otherwise, the second output tensor will be an empty tensor with shape
        [minibatch_size, height, width, 0].
    """
    # Sanitize the list of pixel differential attributes.
    if diff_attrs is None:
        diff_attrs = []
    elif diff_attrs != 'all':
        diff_attrs = np.asarray(diff_attrs, np.int32)
        assert len(diff_attrs.shape) == 1
        diff_attrs = diff_attrs.tolist()

    diff_attrs_all = int(diff_attrs == 'all')
    diff_attrs_list = [] if diff_attrs_all else diff_attrs

    # Check inputs.
    assert all(isinstance(x, torch.Tensor) for x in (attr, rast, tri))
    if diff_attrs:
        assert isinstance(rast_db, torch.Tensor)

    # Choose stub.
    if diff_attrs:
        return _interpolate_func_da.apply(attr, rast, tri, rast_db, diff_attrs_all, diff_attrs_list)
    else:
        return _interpolate_func.apply(attr, rast, tri)

#----------------------------------------------------------------------------
# Texture
#----------------------------------------------------------------------------

# Linear-mipmap-linear and linear-mipmap-nearest: Mipmaps enabled.
class _texture_func_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, filter_mode, tex, uv, uv_da, mip_level_bias, mip_wrapper, filter_mode_enum, boundary_mode_enum, *mip_stack):
        raise NotImplementedError("nvdiffrast_musa does not support mipmapped texture sampling")

    @staticmethod
    def backward(ctx, dy):
        raise RuntimeError(_FORWARD_ONLY_ERROR)

# Linear and nearest: Mipmaps disabled.
class _texture_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum):
        out = _nvdiffrast_c.texture_fwd(tex, uv, filter_mode_enum, boundary_mode_enum)
        ctx.save_for_backward(tex, uv)
        ctx.saved_misc = filter_mode, filter_mode_enum, boundary_mode_enum
        return out

    @staticmethod
    def backward(ctx, dy):
        raise RuntimeError(_FORWARD_ONLY_ERROR)

# Op wrapper.
def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode='auto', boundary_mode='wrap', max_mip_level=None):
    """Perform texture sampling.

    All input tensors must be contiguous and reside in GPU memory. The output tensor
    will be contiguous and reside in GPU memory.

    Args:
        tex: Texture tensor with dtype `torch.float32`. For 2D textures, must have shape
             [minibatch_size, tex_height, tex_width, tex_channels]. For cube map textures,
             must have shape [minibatch_size, 6, tex_height, tex_width, tex_channels] where
             tex_width and tex_height are equal. Note that `boundary_mode` must also be set
             to 'cube' to enable cube map mode. Broadcasting is supported along the minibatch axis.
        uv: Tensor containing per-pixel texture coordinates. When sampling a 2D texture,
            must have shape [minibatch_size, height, width, 2]. When sampling a cube map
            texture, must have shape [minibatch_size, height, width, 3].
        uv_da: (Optional) Tensor containing image-space derivatives of texture coordinates.
               Must have same shape as `uv` except for the last dimension that is to be twice
               as long.
        mip_level_bias: (Optional) Per-pixel bias for mip level selection. If `uv_da` is omitted,
                        determines mip level directly. Must have shape [minibatch_size, height, width].
        mip: (Optional) Preconstructed mipmap stack from a `texture_construct_mip()` call, or a list
                        of tensors specifying a custom mipmap stack. When specifying a custom mipmap stack,
                        the tensors in the list must follow the same format as `tex` except for width and
                        height that must follow the usual rules for mipmap sizes. The base level texture
                        is still supplied in `tex` and must not be included in the list. Gradients of a
                        custom mipmap stack are not automatically propagated to base texture but the mipmap
                        tensors will receive gradients of their own. If a mipmap stack is not specified
                        but the chosen filter mode requires it, the mipmap stack is constructed internally
                        and discarded afterwards.
        filter_mode: Texture filtering mode to be used. Valid values are 'auto', 'nearest',
                     'linear', 'linear-mipmap-nearest', and 'linear-mipmap-linear'. Mode 'auto'
                     selects 'linear' if neither `uv_da` or `mip_level_bias` is specified, and
                     'linear-mipmap-linear' when at least one of them is specified, these being
                     the highest-quality modes possible depending on the availability of the
                     image-space derivatives of the texture coordinates or direct mip level information.
        boundary_mode: Valid values are 'wrap', 'clamp', 'zero', and 'cube'. If `tex` defines a
                       cube map, this must be set to 'cube'. The default mode 'wrap' takes fractional
                       part of texture coordinates. Mode 'clamp' clamps texture coordinates to the
                       centers of the boundary texels. Mode 'zero' virtually extends the texture with
                       all-zero values in all directions.
        max_mip_level: If specified, limits the number of mipmaps constructed and used in mipmap-based
                       filter modes.

    Returns:
        A tensor containing the results of the texture sampling with shape
        [minibatch_size, height, width, tex_channels]. Cube map fetches with invalid uv coordinates
        (e.g., zero vectors) output all zeros and do not propagate gradients.
    """

    if uv_da is not None or mip_level_bias is not None or mip is not None:
        raise NotImplementedError("nvdiffrast_musa does not support mipmapped texture sampling")

    # Default filter mode.
    if filter_mode == 'auto':
        filter_mode = 'linear'
    if filter_mode not in ['nearest', 'linear']:
        raise NotImplementedError("nvdiffrast_musa texture() supports only nearest, linear, and non-mipmap auto filter modes")

    # Sanitize inputs.
    if max_mip_level is not None:
        max_mip_level = int(max_mip_level)
        assert max_mip_level >= 0
        if max_mip_level != 0:
            raise NotImplementedError("nvdiffrast_musa does not support mipmapped texture sampling")

    # Check inputs.
    assert isinstance(tex, torch.Tensor) and isinstance(uv, torch.Tensor)
    if 'mipmap' in filter_mode:
        raise NotImplementedError("nvdiffrast_musa does not support mipmapped texture sampling")

    # Convert filter mode to internal enumeration.
    filter_mode_dict = {'nearest': 0, 'linear': 1, 'linear-mipmap-nearest': 2, 'linear-mipmap-linear': 3}
    filter_mode_enum = filter_mode_dict[filter_mode]

    # Convert boundary mode to internal enumeration.
    boundary_mode_dict = {'cube': 0, 'wrap': 1, 'clamp': 2, 'zero': 3}
    boundary_mode_enum = boundary_mode_dict[boundary_mode]

    # Choose stub.
    return _texture_func.apply(filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum)

# Mipmap precalculation for cases where the texture stays constant.
def texture_construct_mip(tex, max_mip_level=None, cube_mode=False):
    """Construct a mipmap stack for a texture.

    This function can be used for constructing a mipmap stack for a texture that is known to remain
    constant. This avoids reconstructing it every time `texture()` is called.

    Args:
        tex: Texture tensor with the same constraints as in `texture()`.
        max_mip_level: If specified, limits the number of mipmaps constructed.
        cube_mode: Must be set to True if `tex` specifies a cube map texture.

    Returns:
        An opaque object containing the mipmap stack. This can be supplied in a call to `texture()` 
        in the `mip` argument.
    """

    raise NotImplementedError("nvdiffrast_musa does not support mipmap construction")

#----------------------------------------------------------------------------
# Antialias.
#----------------------------------------------------------------------------

class _antialias_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, color, rast, pos, tri, topology_hash, pos_gradient_boost):
        raise NotImplementedError("nvdiffrast_musa does not support antialias")

    @staticmethod
    def backward(ctx, dy):
        raise RuntimeError(_FORWARD_ONLY_ERROR)

# Op wrapper.
def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    """Perform antialiasing.

    All input tensors must be contiguous and reside in GPU memory. The output tensor
    will be contiguous and reside in GPU memory.

    Note that silhouette edge determination is based on vertex indices in the triangle
    tensor. For it to work properly, a vertex belonging to multiple triangles must be
    referred to using the same vertex index in each triangle. Otherwise, nvdiffrast will always
    classify the adjacent edges as silhouette edges, which leads to bad performance and
    potentially incorrect gradients. If you are unsure whether your data is good, check
    which pixels are modified by the antialias operation and compare to the example in the
    documentation.

    Args:
        color: Input image to antialias with shape [minibatch_size, height, width, num_channels].
        rast: Main output tensor from `rasterize()`.
        pos: Vertex position tensor used in the rasterization operation.
        tri: Triangle tensor used in the rasterization operation.
        topology_hash: (Optional) Preconstructed topology hash for the triangle tensor. If not
                       specified, the topology hash is constructed internally and discarded afterwards.
        pos_gradient_boost: (Optional) Multiplier for gradients propagated to `pos`.

    Returns:
        A tensor containing the antialiased image with the same shape as `color` input tensor.
    """

    raise NotImplementedError("nvdiffrast_musa does not support antialias")

# Topology hash precalculation for cases where the triangle array stays constant.
def antialias_construct_topology_hash(tri):
    """Construct a topology hash for a triangle tensor.

    This function can be used for constructing a topology hash for a triangle tensor that is 
    known to remain constant. This avoids reconstructing it every time `antialias()` is called.

    Args:
        tri: Triangle tensor with shape [num_triangles, 3]. Must be contiguous and reside in
             GPU memory.

    Returns:
        An opaque object containing the topology hash. This can be supplied in a call to 
        `antialias()` in the `topology_hash` argument.
    """
    raise NotImplementedError("nvdiffrast_musa does not support antialias")

#----------------------------------------------------------------------------
# Legacy OpenGL context stub for backwards compatibility.
#----------------------------------------------------------------------------

class RasterizeGLContext(RasterizeCudaContext):
    def __init__(self, output_db=True, mode='automatic', device=None):
        warnings.warn("RasterizeGLContext has been deprecated and uses RasterizeCudaContext internally", DeprecationWarning, stacklevel=2)
        super().__init__(device=device)

    def set_context(self):
        pass

    def release_context(self):
        pass

#----------------------------------------------------------------------------
