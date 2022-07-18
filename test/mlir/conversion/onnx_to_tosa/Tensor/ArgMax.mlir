// RUN: onnx-mlir-opt --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_argmax(%arg0: tensor<8x16x32xf32>) -> tensor<8x16x32xi64> {
  %0 = "onnx.ArgMax"(%arg0) {axis = 2 : si64, keepdims = 1 : si64, onnx_node_name = "ArgMax_0"} : (tensor<8x16x32xf32>) -> tensor<8x16x32xi64>
  return %0 : tensor<8x16x32xi64>
// CHECK-LABEL:  func @test_argmax
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8x16x32xf32>) -> tensor<8x16x32xi64> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.argmax"([[PARAM_0_]]) {axis = 2 : i64} : (tensor<8x16x32xf32>) -> tensor<8x16x32xi64>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<8x16x32xi64>
// CHECK-NEXT:   }
}
