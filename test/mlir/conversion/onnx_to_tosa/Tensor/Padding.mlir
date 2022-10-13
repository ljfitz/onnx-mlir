// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_pad(%arg0: tensor<20x16x44x32xf32>) ->  tensor<24x22x56x46xf32>     {
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[4.5000]> : tensor<1xf32>} : () -> tensor<1xf32> 
    %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<24x22x56x46xf32> 
    return %2 :   tensor<24x22x56x46xf32> 
// CHECK-LABEL: test_pad
// CHECK: %[[VAR0:.*]] = "tosa.const"() {value = dense<4.500000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK: %[[VAR1:.*]] = "tosa.const"() {value = dense<[{{\[}}0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>} : () -> tensor<4x2xi64>
// CHECK: %[[VAR2:.*]] = "tosa.pad"(%arg0, %[[VAR1]], %[[VAR0]])
}
