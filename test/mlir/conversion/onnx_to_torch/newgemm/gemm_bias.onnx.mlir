//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK: %[[TRANSB:.*]] = torch.constant.int 1
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) : (tensor<3x6xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
//CHECK: torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[3,6],f32>, !torch.vtensor<[6,4],f32> -> !torch.vtensor<[3,4],f32>
//CHECK: torch.aten.add.Tensor %6, %arg1, %[[TRANSB]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[6,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
    return %0 : tensor<3x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 6] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [6 , 4] , \22name\22 : \22b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4] , \22name\22 : \22c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4] , \22name\22 : \22y\22 }\0A\0A]\00"} : () -> ()
}
