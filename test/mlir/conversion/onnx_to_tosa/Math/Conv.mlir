// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s


func.func @test_onnx_conv2d_stride_13(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x75x75xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x75x75xf32>
  return %0 : tensor<5x2x75x75xf32>
// CHECK-LABEL:  func @test_onnx_conv2d_stride_13
// CHECK: [[PERM1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: [[TRANSINPUT:%.+]] = "tosa.transpose"(%arg0, [[PERM1]]) : (tensor<5x3x1024x1024xf32>, tensor<4xi64>) -> tensor<5x1024x1024x3xf32>
// CHECK: [[PERM2:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: [[TRANSKERNEL:%.+]] = "tosa.transpose"(%arg1, [[PERM2]]) : (tensor<2x3x64x64xf32>, tensor<4xi64>) -> tensor<2x64x64x3xf32>
// CHECK: [[OUTPUT:%.+]] = "tosa.conv2d"([[TRANSINPUT]], [[TRANSKERNEL]], %arg2) {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [13, 13]} : (tensor<5x1024x1024x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x75x75x2xf32>
// CHECK: [[PERM3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: {{%.+}} = "tosa.transpose"([[OUTPUT]], [[PERM3]]) : (tensor<5x75x75x2xf32>, tensor<4xi64>) -> tensor<5x2x75x75xf32>
}

// -----
func.func @test_onnx_conv2d_novalue(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32>
  return %0 : tensor<5x2x965x967xf32>
// CHECK-LABEL:  func @test_onnx_conv2d_novalue
// CHECK: [[PERM1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: [[TRANSINPUT:%.+]] = "tosa.transpose"(%arg0, [[PERM1]]) : (tensor<5x3x1024x1024xf32>, tensor<4xi64>) -> tensor<5x1024x1024x3xf32>
// CHECK: [[PERM2:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: [[TRANSKERNEL:%.+]] = "tosa.transpose"(%arg1, [[PERM2]]) : (tensor<2x3x64x64xf32>, tensor<4xi64>) -> tensor<2x64x64x3xf32>
// CHECK: [[BIAS:%.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: [[OUTPUT:%.+]] = "tosa.conv2d"([[TRANSINPUT]], [[TRANSKERNEL]], [[BIAS]]) {dilation = [1, 1], pad =  [1, 3, 2, 4], stride = [1, 1]} : (tensor<5x1024x1024x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x965x967x2xf32>
// CHECK: [[PERM3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: {{%.+}} = "tosa.transpose"([[OUTPUT]], [[PERM3]]) : (tensor<5x965x967x2xf32>, tensor<4xi64>) -> tensor<5x2x965x967xf32>
}

// -----
func.func @test_onnx_conv2d_no_dilation_pad(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<7x3x64x64xf32>, %arg2: none) ->   tensor<5x7x74x74xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {strides = [13, 13]} : (tensor<5x3x1024x1024xf32>, tensor<7x3x64x64xf32>, none) ->  tensor<5x7x74x74xf32>
  return %0 :  tensor<5x7x74x74xf32>
// CHECK-LABEL:  func @test_onnx_conv2d_no_dilation_pad
// CHECK: [[PERM1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: [[TRANSINPUT:%.+]] = "tosa.transpose"(%arg0, [[PERM1]]) : (tensor<5x3x1024x1024xf32>, tensor<4xi64>) -> tensor<5x1024x1024x3xf32>
// CHECK: [[PERM2:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: [[TRANSKERNEL:%.+]] = "tosa.transpose"(%arg1, [[PERM2]]) : (tensor<7x3x64x64xf32>, tensor<4xi64>) -> tensor<7x64x64x3xf32>
// CHECK: [[BIAS:%.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<7xf32>} : () -> tensor<7xf32>
// CHECK: [[OUTPUT:%.+]] = "tosa.conv2d"([[TRANSINPUT]], [[TRANSKERNEL]], [[BIAS]]) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [13, 13]} : (tensor<5x1024x1024x3xf32>, tensor<7x64x64x3xf32>, tensor<7xf32>) -> tensor<5x74x74x7xf32>
// CHECK: [[PERM3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: {{%.+}} = "tosa.transpose"([[OUTPUT]], [[PERM3]]) : (tensor<5x74x74x7xf32>, tensor<4xi64>) -> tensor<5x7x74x74xf32>
}
// -----
func.func @test_onnx_conv2d_no_dilation_pad_stride(%arg0: tensor<5x3x1024x1050xf32>, %arg1 : tensor<2x3x60x64xf32>, %arg2: none) ->  tensor<5x2x965x987xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) : (tensor<5x3x1024x1050xf32>, tensor<2x3x60x64xf32>, none) ->  tensor<5x2x965x987xf32>
  return %0 : tensor<5x2x965x987xf32>
// CHECK-LABEL:  func @test_onnx_conv2d_no_dilation_pad_stride
// CHECK: [[PERM1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: [[TRANSINPUT:%.+]] = "tosa.transpose"(%arg0, [[PERM1]]) : (tensor<5x3x1024x1050xf32>, tensor<4xi64>) -> tensor<5x1024x1050x3xf32>
// CHECK: [[PERM2:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: [[TRANSKERNEL:%.+]] = "tosa.transpose"(%arg1, [[PERM2]]) : (tensor<2x3x60x64xf32>, tensor<4xi64>) -> tensor<2x60x64x3xf32>
// CHECK: [[BIAS:%.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: [[OUTPUT:%.+]] = "tosa.conv2d"([[TRANSINPUT]], [[TRANSKERNEL]], [[BIAS]]) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<5x1024x1050x3xf32>, tensor<2x60x64x3xf32>, tensor<2xf32>) -> tensor<5x965x987x2xf32>
// CHECK: [[PERM3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: {{%.+}} = "tosa.transpose"([[OUTPUT]], [[PERM3]]) : (tensor<5x965x987x2xf32>, tensor<4xi64>) -> tensor<5x2x965x987xf32>
}