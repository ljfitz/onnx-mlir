// This is the precise Gemm that is being generated by the TensorFlow version of ResNet50.
//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x2048xf32>, %arg1: tensor<1000x2048xf32>, %arg2: tensor<1000xf32>) -> tensor<1x1000xf32> attributes {input_names = ["x", "fc.weight", "fc.bias"], output_names = ["y"]} {
%0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<1x2048xf32>, tensor<1000x2048xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
// We now generate a linear instead of the a transpose, mul and add as it can 
// be represented by a linear function in torch, e.g. Y = XA^T + B
//CHECK: [[RES1:%.]] = torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[1,2048],f32>, !torch.vtensor<[1000,2048],f32>, !torch.vtensor<[1000],f32> -> !torch.vtensor<[1,1000],f32>       
return %0 : tensor<1x1000xf32>
  }
}