//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x3x75x75xf32>) -> tensor<f32> attributes {input_names = ["input.1"], output_names = ["15"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<75x3x3x3xf32>} : () -> tensor<75x3x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<[0.0964951366, -0.0757777392, 0.164070264, -0.0851661637, -0.0532653518, -0.0846414789, 0.138772383, -0.157166913, 0.131651714, -0.189847916, -0.0981560796, 0.0917280092, 0.0877307578, 5.092120e-02, -0.0909967646, -0.160857961, -0.115096614, -0.156773329, 0.0469386801, 0.0576468073, -0.012520073, 0.0872224643, -0.189858168, 0.065727517, 0.0522207655, 0.0676124915, 0.153921276, -0.170313492, -0.0185590982, 0.00902692508, 8.722900e-03, -0.157349676, 0.0620967186, 0.0918546915, -0.0916046723, 0.00762272393, -0.125012606, 0.15406087, 0.0330839679, 0.151479885, -0.09943787, 0.186132073, 0.0626699254, 0.00155355246, 0.022335669, -0.153754413, 0.15888232, -0.0093089724, -0.1912027, -0.0171422102, -0.00277747656, 0.0335581526, 0.0364526771, 0.157430515, -0.162931398, -0.0896333307, -0.14063631, 0.116748422, 0.176921383, -0.105432868, 0.0229430087, 0.0513861626, 0.107943505, 0.0624648444, 0.0266540125, 0.0153248962, -0.0784423649, 6.70704641E-4, 0.126774415, -0.0908973068, 0.0237670802, 0.16966176, -0.111446612, -0.0349170901, -0.108628549]> : tensor<75xf32>} : () -> tensor<75xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x3x75x75xf32>, tensor<75x3x3x3xf32>, tensor<75xf32>) -> tensor<1x75x75x75xf32>
    %3 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<38x75x3x3xf32>} : () -> tensor<38x75x3x3xf32>
    %4 = "onnx.Constant"() {value = dense<[0.00546126813, -0.0196392834, -0.00680329232, -0.00536963856, 0.0208435748, -0.0130821206, 0.00688991183, 0.0203272365, -0.0281473156, 0.0375216939, -0.00790663436, -0.00937498081, 0.0322414711, -0.00816204119, -0.00912685133, -0.0263613071, 0.0287726372, 0.03483456, -0.0109078307, 0.0072224536, 0.0287012924, -0.0105988225, 0.0265569966, 0.0243289303, 0.0286068134, -0.0294105168, 0.00387686794, 0.0150157642, -0.00733148213, 0.0158965383, -5.93863253E-4, -0.0215825532, -0.00102285738, 0.0337187685, 0.027797265, -0.0225700587, 0.0205344558, -0.0263472106]> : tensor<38xf32>} : () -> tensor<38xf32>
    %5 = "onnx.Conv"(%2, %3, %4) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_1", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x75x75x75xf32>, tensor<38x75x3x3xf32>, tensor<38xf32>) -> tensor<1x38x38x38xf32>
    %6 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<17x38x7x7xf32>} : () -> tensor<17x38x7x7xf32>
    %7 = "onnx.Constant"() {value = dense<[-0.0116659543, -0.00456158444, 0.020029692, 0.0173839275, -0.00753006897, -0.0080922693, 0.007518908, -0.00651095714, 4.618840e-03, -0.0171762761, 0.00581417046, 0.0173386596, -0.00731108198, 0.00438685762, -0.0223495141, 0.0134883206, -0.00964440312]> : tensor<17xf32>} : () -> tensor<17xf32>
    %8 = "onnx.Conv"(%5, %6, %7) {dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], onnx_node_name = "Conv_2", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x38x38x38xf32>, tensor<17x38x7x7xf32>, tensor<17xf32>) -> tensor<1x17x17x17xf32>
    %9 = "onnx.MaxPoolSingleOut"(%8) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_3", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x17x17x17xf32>) -> tensor<1x17x8x8xf32>
    %10 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<10x1088xf32>} : () -> tensor<10x1088xf32>
    %11 = "onnx.Constant"() {value = dense<[-0.00859574787, 0.025370162, 0.00614910666, -0.0131458696, -0.0181427132, -0.022160003, -0.0188410394, 0.0217594896, -0.0069691157, 0.00782693177]> : tensor<10xf32>} : () -> tensor<10xf32>
//CHECK: %int[[AVAL:[^ ]*]] = torch.constant.int 0
//CHECK: %int[[BVAL:[^ ]*]] = torch.constant.int 1
//CHECK: torch.aten.transpose.int %28, %int[[AVAL:[^ ]*]], %int[[BVAL:[^ ]*]] :
    %12 = "onnx.Gemm"(%9, %10, %11) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_5", transB = 1 : si64} : (tensor<1x17x8x8xf32>, tensor<10x1088xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    %13 = "onnx.ReduceMean"(%12) {keepdims = 0 : si64, onnx_node_name = "ReduceMean_6"} : (tensor<1x10xf32>) -> tensor<f32>
    return %13 : tensor<f32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 75 , 75] , \22name\22 : \22input.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [] , \22name\22 : \2215\22 }\0A\0A]\00"} : () -> ()
}
