//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x1x15x15xf32>) -> tensor<1x1x673x673xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [3, 3], onnx_node_name = "MaxPool_0", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x1x15x15xf32>) -> tensor<1x1x7x7xf32>
    %1 = "onnx.Constant"() {value = dense<[0, 0, 333, 333, 0, 0, 333, 333]> : tensor<8xi64>} : () -> tensor<8xi64>
    %2 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK: [[VAL:%[^ ]*]] = torch.constant.float 0.000000e+00
    //CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int333, %int333{{_*[0-9]*}}, %int333{{_*[0-9]*}}, %int333{{_*[0-9]*}} :
    //CHECK: torch.aten.constant_pad_nd %6, [[PAD]], [[VAL]] : !torch.vtensor<[1,1,7,7],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,1,673,673],f32>
    %3 = "onnx.Pad"(%0, %1, %2) {mode = "constant"} : (tensor<1x1x7x7xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x1x673x673xf32>
    return %3 : tensor<1x1x673x673xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 15 , 15] , \22name\22 : \220\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 673 , 673] , \22name\22 : \222\22 }\0A\0A]\00"} : () -> ()
}
