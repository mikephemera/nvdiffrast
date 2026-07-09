// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "torch_common.inl"
#include "torch_types.h"
#include <tuple>

//------------------------------------------------------------------------
// Op prototypes. Return type macros for readability.

#define OP_RETURN_T     torch::Tensor
#define OP_RETURN_TT    std::tuple<torch::Tensor, torch::Tensor>
#define OP_RETURN_TTT   std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
#define OP_RETURN_TTTT  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
#define OP_RETURN_TTV   std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor> >
#define OP_RETURN_TTTTV std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<torch::Tensor> >

OP_RETURN_TT        rasterize_fwd_cuda                  (RasterizeCRStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges, int peeling_idx);
OP_RETURN_TT        interpolate_fwd                     (torch::Tensor attr, torch::Tensor rast, torch::Tensor tri);
OP_RETURN_TT        interpolate_fwd_da                  (torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor rast_db, bool diff_attrs_all, std::vector<int>& diff_attrs_vec);
OP_RETURN_T         texture_fwd                         (torch::Tensor tex, torch::Tensor uv, int filter_mode, int boundary_mode);

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // State classes.
    pybind11::class_<RasterizeCRStateWrapper>(m, "RasterizeCRStateWrapper").def(pybind11::init<int>());
    pybind11::class_<TextureMipWrapper>(m, "TextureMipWrapper").def(pybind11::init<>());

    // Plumbing to torch/c10 logging system.
    m.def("get_log_level", [](void)     { return FLAGS_caffe2_log_level;  }, "get log level");
    m.def("set_log_level", [](int level){ FLAGS_caffe2_log_level = level; }, "set log level");

    // Ops.
    m.def("rasterize_fwd_cuda",                 &rasterize_fwd_cuda,                    "rasterize forward op (cuda)");
    m.def("interpolate_fwd",                    &interpolate_fwd,                       "interpolate forward op with attribute derivatives");
    m.def("interpolate_fwd_da",                 &interpolate_fwd_da,                    "interpolate forward op without attribute derivatives");
    m.def("texture_fwd",                        &texture_fwd,                           "texture forward op without mipmapping");
}

//------------------------------------------------------------------------
