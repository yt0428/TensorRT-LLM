#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# Mistral Large 4 SM107 persistent whole-layer megakernel native target.
# Supports both the modeling_v2 family CMake helper (add_modeling_v2_family) and
# standalone shared-library compilation.

set(ml4_mega_cu
    "${CMAKE_CURRENT_SOURCE_DIR}/../mistral_large_4_nvfp4__sm_107__tp4/csrc/ml4Megakernel.cu"
)
set(ml4_mega_op
    "${CMAKE_CURRENT_SOURCE_DIR}/../mistral_large_4_nvfp4__sm_107__tp4/csrc/ml4MegakernelOp.cpp"
)

set_property(
  SOURCE "${ml4_mega_cu}"
  APPEND
  PROPERTY COMPILE_DEFINITIONS ML4_SMEM_MAIN=200704 ML4_SETUP_MODE=1 ML4_LEAN=1)
set_property(
  SOURCE "${ml4_mega_cu}" APPEND
  PROPERTY COMPILE_OPTIONS --maxrregcount=200 --ftz=false --prec-div=true
           --prec-sqrt=true -Xptxas=-v)

if(COMMAND add_modeling_v2_family)
  if(107 IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
    set_property(
      SOURCE "${ml4_mega_cu}" APPEND
      PROPERTY COMPILE_OPTIONS --generate-code=arch=compute_107a,code=sm_107a)
  endif()
  add_modeling_v2_family(mistral_large4)
  add_modeling_v2_target(
    mistral_large4
    mistral_large_4_nvfp4__sm_107__tp4
    ARCHS
    100f
    SOURCES
    ml4Megakernel.cu
    ml4MegakernelOp.cpp
    DEPS
    cuda_driver
    REQUIRES
    MULTI_DEVICE)
else()
  set(_ml4_models_root "${CMAKE_CURRENT_SOURCE_DIR}/../..")

  add_library(th_ml4_megakernel_cuda OBJECT "${ml4_mega_cu}")
  set_property(TARGET th_ml4_megakernel_cuda PROPERTY POSITION_INDEPENDENT_CODE
                                                      ON)
  set_property(TARGET th_ml4_megakernel_cuda
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_include_directories(
    th_ml4_megakernel_cuda
    PRIVATE "${_ml4_models_root}" "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}")
  if(COMMAND set_cuda_architectures)
    set_cuda_architectures(th_ml4_megakernel_cuda 107 100f)
  endif()

  add_library(th_ml4_megakernel MODULE $<TARGET_OBJECTS:th_ml4_megakernel_cuda>
                                       "${ml4_mega_op}")
  set_property(TARGET th_ml4_megakernel PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(
    th_ml4_megakernel
    PROPERTIES PREFIX "lib" CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
               LINKER_LANGUAGE CXX BUILD_RPATH
                                   "$ORIGIN;$ORIGIN/../../nvidia/nccl/lib")
  target_include_directories(
    th_ml4_megakernel
    PRIVATE "${_ml4_models_root}" "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}")
  target_link_libraries(
    th_ml4_megakernel PRIVATE ${TORCH_LIBRARIES} th_utils ${Python3_LIBRARIES}
                              ${SHARED_TARGET} ${CUDA_RT_LIB} ${CUDA_DRV_LIB})
endif()
