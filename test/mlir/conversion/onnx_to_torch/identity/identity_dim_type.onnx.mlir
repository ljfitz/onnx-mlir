//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32> attributes {input_names = ["input"], output_names = ["3"]} {
 %1 = "onnx.Identity"(%arg0) : (tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32> 
//CHECK: (%arg0: !torch.vtensor<[1,1,8,8],f32>) -> !torch.vtensor<[1,1,8,8],f32>
//CHECK: %arg0 : !torch.vtensor<[1,1,8,8],f32>
   return %1 : tensor<1x1x8x8xf32>
 }
}
