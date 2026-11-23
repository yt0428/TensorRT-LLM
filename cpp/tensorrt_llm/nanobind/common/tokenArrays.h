/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <vector>

namespace NanobindUtils
{

namespace nb = nanobind;

//! Copies Python token ids into a vector. A 1-D contiguous host array of TokenId (a numpy array, or
//! anything nanobind imports through DLPack or the buffer protocol) is copied with one memcpy.
//! Anything else, e.g. list[int], goes through nanobind's sequence cast, one element at a time.
template <typename TokenId>
std::vector<TokenId> tokensFromPython(nb::handle tokens)
{
    // An array import tried on a list or tuple always fails, and failing costs more than casting a
    // short list, so they skip it.
    if (!PyList_CheckExact(tokens.ptr()) && !PyTuple_CheckExact(tokens.ptr()))
    {
        nb::ndarray<TokenId const, nb::ndim<1>, nb::c_contig, nb::device::cpu> array;
        if (nb::try_cast(tokens, array, /*convert=*/false))
        {
            std::vector<TokenId> result(array.shape(0));
            if (!result.empty())
            {
                std::memcpy(result.data(), array.data(), result.size() * sizeof(TokenId));
            }
            return result;
        }
    }
    std::vector<TokenId> result;
    if (!nb::try_cast(tokens, result))
    {
        throw nb::type_error("token ids must be a sequence of integers or a 1-D integer array");
    }
    return result;
}

} // namespace NanobindUtils
