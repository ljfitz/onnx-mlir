//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32> attributes {input_names = ["0"], output_names = ["1"]} {

    //CHECK: [[ALPHA:%[^ ]*]] = torch.constant.float 0.0099999997764825821
    //CHECK: torch.aten.leaky_relu %arg0, [[ALPHA]] : !torch.vtensor<[1,3,2,2],f32>, !torch.float -> !torch.vtensor<[1,3,2,2],f32>
    %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.00999999977 : f32, onnx_node_name = "LeakyRelu_0"} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    return %0 : tensor<1x3x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 2 , 2] , \22name\22 : \220\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 2 , 2] , \22name\22 : \221\22 }\0A\0A]\00"} : () -> ()
}
