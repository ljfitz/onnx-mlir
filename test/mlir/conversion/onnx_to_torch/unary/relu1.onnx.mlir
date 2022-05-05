//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x10x5x5xf32>) -> tensor<1x5x3x3xf32> attributes {input_names = ["input"], output_names = ["5"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<5x10x3x3xf32>} : () -> tensor<5x10x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.0760196224, 0.054027766, 0.0197769403, -0.0698400959, 8.678320e-02]> : tensor<5xf32>} : () -> tensor<5xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x10x5x5xf32>, tensor<5x10x3x3xf32>, tensor<5xf32>) -> tensor<1x5x5x5xf32>
    %3 = "onnx.MaxPoolSingleOut"(%2) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_1", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x5x5x5xf32>) -> tensor<1x5x3x3xf32>
    %4 = "onnx.Relu"(%3) {onnx_node_name = "Relu_2"} : (tensor<1x5x3x3xf32>) -> tensor<1x5x3x3xf32>
//CHECK: torch.aten.relu %13 : !torch.vtensor<[1,5,3,3],f32> -> !torch.vtensor<[1,5,3,3],f32>
//CHECK: torch.tensor_static_info_cast %14 : !torch.vtensor<[1,5,3,3],f32> to !torch.vtensor<[1,5,3,3],f32>
//CHECK: builtin.unrealized_conversion_cast %15 : !torch.vtensor<[1,5,3,3],f32> to tensor<1x5x3x3xf32>
    return %4 : tensor<1x5x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10 , 5 , 5] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 5 , 3 , 3] , \22name\22 : \225\22 }\0A\0A]\00"} : () -> ()
}
