// RUN: onnx-mlir-opt --maccel=NNPA --constprop-zhigh %s -split-input-file | FileCheck %s

// -----

// COM: Test constant stickify for layout 1D.
// CHECK: func @remove_stick_1d() -> tensor<6xf32, #zhigh.encoding<{dataLayout = "1D"}>> {
func.func @remove_stick_1d() -> tensor<6xf32, #zhigh.encoding<{dataLayout = "1D"}>> {
  %inp = "onnx.Constant"() {value = dense<[0., 1., 2., 3., 4., 5.]> : tensor<6xf32>} : () -> tensor<6xf32>
  %res = "zhigh.Stick"(%inp) {layout = "1D"} : (tensor<6xf32>) -> tensor<6xf32, #zhigh.encoding<{dataLayout = "1D"}>>
  return %res : tensor<6xf32, #zhigh.encoding<{dataLayout = "1D"}>>

  // CHECK-NEXT: %0 = "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<4096xi8>} : () -> tensor<6xf32, #zhigh.encoding<{dataLayout = "1D"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.Stick"
}

// -----

// COM: Test constant stickify for layout 2D.
// CHECK: func @remove_stick_2d() -> tensor<2x3xf32> {
func.func @remove_stick_2d() -> tensor<2x3xf32> {
  %inp = "onnx.Constant"() {value = dense<[[0., 1., 2.], [3., 4., 5.]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %st= "zhigh.Stick"(%inp) {layout = "2D"} : (tensor<2x3xf32>) -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "2D"}>>
  %res = "zhigh.Unstick"(%st) {layout = "2D"} : (tensor<2x3xf32, #zhigh.encoding<{dataLayout = "2D"}>>) -> tensor<2x3xf32>
  return %res : tensor<2x3xf32>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<4096xi8>} : () -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "2D"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.Stick"
}

// -----

// COM: Test constant stickify for layout 2DS.
// CHECK: func @remove_stick_2ds() -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "2DS"}>> {
func.func @remove_stick_2ds() -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "2DS"}>> {
  %inp = "onnx.Constant"() {value = dense<[[0., 1., 2.], [3., 4., 5.]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %res = "zhigh.Stick"(%inp) {layout = "2DS"} : (tensor<2x3xf32>) -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "2DS"}>>
  return %res : tensor<2x3xf32, #zhigh.encoding<{dataLayout = "2DS"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<8192xi8>} : () -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "2DS"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.Stick"
}

// -----

// COM: Test constant stickify for layout 3D. 
// CHECK: func @remove_stick_3d() -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
func.func @remove_stick_3d() -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
  %inp = "onnx.Constant"() {value = dense<[[[0., 1., 2.], [3., 4., 5.]]]> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
  %res = "zhigh.Stick"(%inp) {layout = "3D"} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "3D"}>>
  return %res : tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "3D"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<4096xi8>} : () -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "3D"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.Stick"
}

// -----

// COM: Test constant stickify for layout 3DS. 
// CHECK: func @remove_stick_3ds() -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "3DS"}>> {
func.func @remove_stick_3ds() -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "3DS"}>> {
  %inp = "onnx.Constant"() {value = dense<[[[0., 1., 2.], [3., 4., 5.]]]> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
  %res = "zhigh.Stick"(%inp) {layout = "3DS"} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
  return %res : tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "3DS"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<4096xi8>} : () -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "3DS"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.Stick"
}

// -----

// COM: Test constant stickify for layout 4D. 
// CHECK: func @remove_stick_4d() -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
func.func @remove_stick_4d() -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "4D"}>> {
  %inp = "onnx.Constant"() {value = dense<[[[[0., 1., 2.], [3., 4., 5.]]]]> : tensor<1x1x2x3xf32>} : () -> tensor<1x1x2x3xf32>
  %res = "zhigh.Stick"(%inp) {layout = "4D"} : (tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "4D"}>>
  return %res : tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "4D"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<4096xi8>} : () -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "4D"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.Stick"
}

// -----

// COM: Test constant stickify for layout NHWC. 
// CHECK: func @remove_stick_nhwc() -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
func.func @remove_stick_nhwc() -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
  %inp = "onnx.Constant"() {value = dense<[[[[0., 1., 2.], [3., 4., 5.]]]]> : tensor<1x1x2x3xf32>} : () -> tensor<1x1x2x3xf32>
  %res = "zhigh.Stick"(%inp) {layout = "NHWC"} : (tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
  return %res : tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<4096xi8>} : () -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.Stick"
}

// -----

// COM: Test constant stickify for layout NCHW. 
// CHECK: func @remove_stick_nchw() -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "NCHW"}>> {
func.func @remove_stick_nchw() -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "NCHW"}>> {
  %inp = "onnx.Constant"() {value = dense<[[[[0., 1., 2.], [3., 4., 5.]]]]> : tensor<1x1x2x3xf32>} : () -> tensor<1x1x2x3xf32>
  %res = "zhigh.Stick"(%inp) {layout = "NCHW"} : (tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "NCHW"}>>
  return %res : tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "NCHW"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<8192xi8>} : () -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "NCHW"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.Stick"
}

// -----

// COM: Test constant stickify for layout CNNK_HWCK. 
// CHECK: func @remove_stick_cnnk_hwck() -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "HWCK"}>> {
func.func @remove_stick_cnnk_hwck() -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "HWCK"}>> {
  %inp = "onnx.Constant"() {value = dense<[[[[0., 1., 2.], [3., 4., 5.]]]]> : tensor<1x1x2x3xf32>} : () -> tensor<1x1x2x3xf32>
  %res = "zhigh.Stick"(%inp) {layout = "HWCK"} : (tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>
  return %res : tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<4096xi8>} : () -> tensor<1x1x2x3xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.Stick"
}

// -----

// COM: Test constant stickify for layout ZRH used in GRU.
// Biases are 2D tensors.
// CHECK: func @remove_stick_zrh_2d() -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "ZRH"}>> {
func.func @remove_stick_zrh_2d() -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "ZRH"}>> {
  %z = "onnx.Constant"() {value = dense<[[0., 1., 2.], [3., 4., 5.]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %r = "onnx.Constant"() {value = dense<[[0., 1., 2.], [3., 4., 5.]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %h = "onnx.Constant"() {value = dense<[[0., 1., 2.], [3., 4., 5.]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %res = "zhigh.StickForGRU"(%z, %r, %h) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "ZRH"}>>
  return %res : tensor<2x3xf32, #zhigh.encoding<{dataLayout = "ZRH"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<24576xi8>} : () -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "ZRH"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.StickForGRU"
}

// -----

// COM: Test constant stickify for layout ZRH used in GRU.
// Weights are 3D tensors.
// CHECK: func @remove_stick_zrh_3d() -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "ZRH"}>> {
func.func @remove_stick_zrh_3d() -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "ZRH"}>> {
  %z = "onnx.Constant"() {value = dense<[[[0., 1., 2.], [3., 4., 5.]]]> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
  %r = "onnx.Constant"() {value = dense<[[[0., 1., 2.], [3., 4., 5.]]]> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
  %h = "onnx.Constant"() {value = dense<[[[0., 1., 2.], [3., 4., 5.]]]> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
  %res = "zhigh.StickForGRU"(%z, %r, %h) : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "ZRH"}>>
  return %res : tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "ZRH"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<12288xi8>} : () -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "ZRH"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.StickForGRU"
}

// -----

// COM: Test constant stickify for layout FICO used in LSTM.
// CHECK: func @remove_stick_fico_2d() -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "FICO"}>> {
// Biases are 2D tensors.
func.func @remove_stick_fico_2d() -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "FICO"}>> {
  %f = "onnx.Constant"() {value = dense<[[0., 1., 2.], [3., 4., 5.]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %i = "onnx.Constant"() {value = dense<[[0., 1., 2.], [3., 4., 5.]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %c = "onnx.Constant"() {value = dense<[[0., 1., 2.], [3., 4., 5.]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %o = "onnx.Constant"() {value = dense<[[0., 1., 2.], [3., 4., 5.]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %res = "zhigh.StickForLSTM"(%f, %i, %c, %o) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "FICO"}>>
  return %res : tensor<2x3xf32, #zhigh.encoding<{dataLayout = "FICO"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<32768xi8>} : () -> tensor<2x3xf32, #zhigh.encoding<{dataLayout = "FICO"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.StickForLSTM"
}

// -----

// COM: Test constant stickify for layout FICO used in LSTM.
// CHECK: func @remove_stick_fico_3d() -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "FICO"}>> {
// Weights are 3D tensors.
func.func @remove_stick_fico_3d() -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "FICO"}>> {
  %f = "onnx.Constant"() {value = dense<[[[0., 1., 2.], [3., 4., 5.]]]> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
  %i = "onnx.Constant"() {value = dense<[[[0., 1., 2.], [3., 4., 5.]]]> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
  %c = "onnx.Constant"() {value = dense<[[[0., 1., 2.], [3., 4., 5.]]]> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
  %o = "onnx.Constant"() {value = dense<[[[0., 1., 2.], [3., 4., 5.]]]> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
  %res = "zhigh.StickForLSTM"(%f, %i, %c, %o) : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "FICO"}>>
  return %res : tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "FICO"}>>

  // CHECK-NEXT: "zhigh.StickifiedConstant"() {alignment = 4096 : i64, dense_resource<zhigh> : tensor<16384xi8>} : () -> tensor<1x2x3xf32, #zhigh.encoding<{dataLayout = "FICO"}>>

  // CHECK-NOT: "onnx.Constant"
  // CHECK-NOT: "zhigh.StickForLSTM"
}
