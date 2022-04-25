//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x1x8x8xf32>) -> tensor<1x6x3x3xf32> attributes {input_names = ["input"], output_names = ["5"]} {
    %0 = "onnx.Constant"() {value = dense<[[[[0.0222365465, -0.29355225, 0.0908823832], [-0.173412487, -0.330785275, 0.078472577], [0.126281425, 0.0459890366, -3.23017448E-4]]], [[[0.063176237, -0.0846533775, -0.112484738], [0.0113253593, 0.233641356, 0.103301331], [-0.329823136, -0.0590218306, -0.208348915]]], [[[-0.261517525, -0.191946432, 0.198983356], [0.132580444, 8.046230e-04, -3.141380e-01], [0.222769305, 0.283797562, 0.208771154]]], [[[0.219519019, 0.101525187, 0.0909941643], [-0.330224216, 0.0290210247, 0.0994548052], [-0.191296935, 0.0392216444, -0.238734767]]], [[[0.110565707, 0.00200573611, -0.0272888355], [0.261777252, 0.10111364, 0.0839024409], [-0.324077696, -0.301391691, 0.308608264]]], [[[0.104720831, -0.294060647, -0.244227812], [-0.0826112851, -0.256598711, 0.133390278], [0.0902876481, 0.021089714, -0.279151887]]]]> : tensor<6x1x3x3xf32>} : () -> tensor<6x1x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<[0.291346878, 0.201920122, -0.253405273, -0.016338747, -0.303526759, 0.0325766429]> : tensor<6xf32>} : () -> tensor<6xf32>
    //CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
    //CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int1, %int1{{_*[0-9]*}} :
    //CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int0, %int0{{_*[0-9]*}} :
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_0", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x8x8xf32>, tensor<6x1x3x3xf32>, tensor<6xf32>) -> tensor<1x6x6x6xf32>
//CHECK: torch.aten.conv2d %arg0, %{{[0-9]}}, %{{[0-9]}}, [[STRIDE]], [[PAD]], [[DILATION]], %int1{{_*[0-9]*}} : !torch.vtensor<[1,1,8,8],f32>, !torch.vtensor<[6,1,3,3],f32>, !torch.vtensor<[6],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,6,6,6],f32>
    %3 = "onnx.MaxPoolSingleOut"(%2) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_1", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x6x6x6xf32>) -> tensor<1x6x3x3xf32>
    %4 = "onnx.LeakyRelu"(%3) {alpha = 0.00999999977 : f32, onnx_node_name = "LeakyRelu_2"} : (tensor<1x6x3x3xf32>) -> tensor<1x6x3x3xf32>
    return %4 : tensor<1x6x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 8 , 8] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 6 , 3 , 3] , \22name\22 : \225\22 }\0A\0A]\00"} : () -> ()
}
