# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTOSA
  ConvertONNXToTOSA.cpp

  Math/Elementwise.cpp
  Tensor/ArgMax.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTosa
  MLIRTransforms
  )
