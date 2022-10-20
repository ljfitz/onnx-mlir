// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_alpha(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.618 : f32} : (tensor<3x6xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
// CHECK-LABEL:  @test_alpha(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
// CHECK-DAG:    %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 3, 6]} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 6, 4]} : (tensor<6x4xf32>) -> tensor<1x6x4xf32>
// CHECK-DAG:    %[[ALPHA:.*]] = "tosa.const"() {value = dense<1.618000e+00> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG:    %[[ALPHA_A:.*]] = "tosa.mul"(%[[ALPHA]], %[[A]]) {shift = 0 : i32} : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[ALPHA_AB:.*]] = "tosa.matmul"(%[[ALPHA_A]], %[[B]]) : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:    %[[ALPHA_ABC:.*]] = "tosa.add"(%[[ALPHA_AB]], %arg2) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:    %[[RES:.*]] = "tosa.reshape"(%[[ALPHA_ABC]]) {new_shape = [3, 4]} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:    return %[[RES]] : tensor<3x4xf32>