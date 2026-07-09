// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

// Framework-specific macros to enable code sharing.

//------------------------------------------------------------------------
// PyTorch.

#ifdef NVDR_TORCH
#ifndef __MUSACC__
#include <torch/extension.h>
#include <ATen/musa/MUSAContext.h>
#include <c10/musa/MUSAGuard.h>
#include <pybind11/numpy.h>
#endif
#define NVDR_CHECK(COND, ERR) do { TORCH_CHECK(COND, ERR) } while(0)
#define NVDR_CHECK_MUSA_ERROR(MUSA_CALL) do { musaError_t err = MUSA_CALL; TORCH_CHECK(err == musaSuccess, "MUSA error: ", musaGetErrorString(err), " [", #MUSA_CALL, ";]"); } while(0)
#endif

//------------------------------------------------------------------------
