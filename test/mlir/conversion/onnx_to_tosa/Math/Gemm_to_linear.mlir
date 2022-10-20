// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// because the optional bias is not yet implemented in the upstream TorchToLinAlg pass we will instead
// create a bias with values of 0. Our testing relies on the lowering from torch to host code and this
// will allow us to run the lowering passes.
module attributes {}  {
  func.func @Gemm_to_linear(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4xf32>) -> tensor<1x4xf32> attributes {input_names = ["a", "b"], output_names = ["y"]} {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
//CHECK-LABEL:  @Gemm_to_linear(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4xf32>) -> tensor<1x4xf32>
//CHECK-DAG:    "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }

 func.func @Gemm_to_linear_opt(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x4xf32> attributes {input_names = ["a", "b"], output_names = ["y"]} {
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, none) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
//CHECK-LABEL:  @Gemm_to_linear_opt(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x4xf32>
//CHECK-NOT:    %none = "onnx.NoValue"() {value} : () -> none
//CHECK-DAG:    "tosa.const"() {value = dense<0.000000e+00> : tensor<5xf32>} : () -> tensor<5xf32>
//CHECK-DAG:    "tosa.fully_connected"(%arg0, %arg1, %0) : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<5xf32>) -> tensor<1x4xf32>
  }
}