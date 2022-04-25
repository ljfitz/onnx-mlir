//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s


module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x16x8x8xf32> attributes {input_names = ["input"], output_names = ["5"]} {
    %0 = "onnx.Constant"() {value = dense<"0xD36F043E2FDB5B3D192451BDA2D24BBDE38D2DBE6833C83D28D2FEBDF642B8BC7DBB853D0877C13CD29B06BE4B2A08BCB4D023BD443F00BEF17DB0BDAA408DBDC49EFBBD27FA083EA60919BE0CB249BDEBFDBEBC013B07BEFD22343E401901BE9758883DBF125B3D23E79EBCB0D5413ED0CB82BC53AB333ED8D52BBE88EF833D4F22813C75421FBD533C79BC7AF48C3D17D10E3DABB9933DB33D29BE92F294BD331F3D3DDCA30C3E4F17943D47099C3D398F81BD32583DBE7FF0203E0E4343BE3711C53D64E6DE3D7F32133D5EB03E3E23ECCFBCE7ACDD3D2C88B13D9E9A0ABE7BF8333E97F3E63DC0A8893D7E24683C65A5B23D45B4F7BDC924433E86EDC5BD065E963DAD10453DB3FEAA3A9CFA3ABE4DC13D3E99BA033D8E5D68BA18D1A9BDC8FA19BC78D3BC3CAD6C17BE914EFABC507681BD2E19DE3D6E54C03B675C4CBCCF82D53D11008E3D752A44BE9EF029BDAF133EBD04120DBE8825003E4CEA403E9288F03DAC710FBE6EEE5BBBF1C4D73DE49910BE27C5373E3AAFBABC02BDD43D009C97BDAE884E3D80AEFD3DFD2332BE919042BE45ABEE3DCF6139BB7778DE3DD1A9333D355230BDF344013CD1C2213EFD2CF93D82A1A93D372F6EBD67EC3ABD74EEDDBD9A72CFBB65D4C5BBFAA1A9BDAD5226BEA60FADBC6A5611BE9240D3BD04BEDB3D780AD93DD5AFBE3C28AA213E1961C23CBC10E7BD1055D6BAB691DC3D6D548F3DE85BC63DB7A802BD8F779BBDD9D0AE3D20238A3DF517FDBDC49BECBDDA45B3BD525C72BDE920BABBF3AA18BE2E8F443D79F3BD3D09D595BBF0D4C7BD2FF72E3EEB10293D2E97C83DFEE325BE39463A3ED80A2E3D5FC8043E5D39C13DFFAC3B3E42FE7BBD529463BC9289153E877AD4BCD700263EDB12C2BD4297CE3C55480EBE58CD0F3DF8C4893D01B9D8BC4330693DD766733D54F78CBD274A063D029EED3C37C233BE0CE10D3EDA8441BEF2E23EBE2E2D9F3D7B07B4BD9917053E0210FF3D7A86003E1B52EA3D49EF2ABEA056F83D77AC07BE92821EBE29A4303E4BC40FBE82A663BD94E829BE2C991C3E0A6C2ABEB4F83F3E2CC0373EA44618BDE7E17CBB9C5782BDC42E413E100E443E092D93BDDA3C55BC3B5B29BD96FCEB3BE969433D57E947BD1869793D7D7D3F3D02D212BD003C2DBEBEC0A5BC6C47BFBD5E1FAB3C0410E2BDCBCA3CBEFD3F4FBDFDE29CBC31919D3DF2EBEC3D23EC38BE2D98EFBDDCD6D83D68A5A93D450A68BD0B40A03DA71B183E6B6E41BDBC160EBEAAE9D93D7FE8353E660CB6BD6A933EBE456F713C2881D8BD74D1423E95A7C7BDCC773D3EB078923C57A603BE080CEEBDF6FDBF3D3FC5C43D67D8163E666ACB3DE5F8C4BDBF0C09BE8E0339BE335A1FBD125B03BD0E023C3E493631BDB552AF3DAD09F03DD980B33C3A340FBD9145C2BD4835DD3D495713BE37BFC5BD0C91B13CFDF70BBE4FACB5BD18B029BE2CBA6CBDC44C08BD88CD793DECB2B3BD0C6E34BE05471D3EC9243EBEE726A3BDA128C4BD73AE063E954823BE5D8FCFBDC77A8C3D222B413E4175153EB116B9BDEA20E73D2006BC3DE1F5DFBDFC6C083E655393BDE7D163BD706213BE9B4F12BE3994F1BAD04336BC13F3F8BDE29613BED3E244BD33C1FE3D1784DB3D0BB5B2BBDD86453D3AC235BEC15B3E3D0FE6D43C8174EFBD86E2BB3C4A54E43CE794153E42BB163E4369443E7F8C27BE9B20803D83F40B3EBA77053EB1C4A03DC8D092BD83F0BF3D6C74DEBDAF81233E36B9ACBD997A173E317D10BED1FD1E3EEEC03EBE161A00BE0D88633D55F510BDB15386BD50EF153E2E3006BE38169CBBACAF0DBDBCC401BE4B2B37BEB0412ABD9C745F3D4D224BBD1CCC473DFB046C3D43D627BE7A1A00BE12B116BEB867063E4D2C003E5BA6303EB698AC3CBF0D78BDCD38813D3AB444BE5EF333BE02AA933D6FB9EBBD9BAAAFBD118D223E3708513DF359113EBC79F8BD9657BDBCEF28EA3D637225BE11A3913D4EE526BE0EE9843DA6E4A7BCD8F4403E9C5C23BD3C4CF33DCDE8133E4253E83CFD6312BE383AD53D9DB52F3E6492443E409124BE79059FBD6DAAAF3D073506BE670E83BB584543BD1BE535BD0E13983DBD10DDBC248938BEBF210DBE19CFDFBDDADC23BD734781BDC09E033E70EFD03D64E691BDAEF138BE84EC40BD8F32F7BDCAEDC5BD3CEB113D4B43433EE5AC1ABDE1CEC2BDCDABE5BD530173BDE917AB3B6252E73D2F51EB3D3BB5A53D21BB99BDFAEC44BCB45428BE2D5EED3DEAED25BE98B8D4BD32A63E3E1C0C9B3C343B413D7735E1BDCF2E003DA69DABBDA477B23DCE8C0D3E3EEE9F3DC5DC053E05C0883D6532193D254E443E4F98243E702873BD1244B5BD430483BD11C8223ECBA4823C1A64573D955920BDB53939BD968BAD3CBC48BFBCEF57903DE950663DD6BF403E0DCADA3C32B6CE3C2CF34ABD91AAF2BD"> : tensor<16x3x3x3xf32>} : () -> tensor<16x3x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<[0.084744744, 0.158327296, -0.0163249075, 0.042918928, 0.00730454362, -0.100426935, -0.140009776, -0.0904820635, -0.101498894, 0.00596288219, 0.152941301, 0.186120018, 0.0249227975, -0.101020187, -0.158874407, -0.129497498]> : tensor<16xf32>} : () -> tensor<16xf32>
//CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
    //CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
    //CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int1, %int1{{_*[0-9]*}} :
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x3x16x16xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x16x16xf32>
//CHECK: torch.aten.conv2d %arg0, %{{[0-9]}}, %{{[0-9]}}, [[STRIDE]], [[PAD]], [[DILATION]], %int1{{_*[0-9]*}} : !torch.vtensor<[1,3,16,16],f32>, !torch.vtensor<[16,3,3,3],f32>, !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,16,16,16],f32>
    %3 = "onnx.MaxPoolSingleOut"(%2) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_1", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x16x16x16xf32>) -> tensor<1x16x8x8xf32>
    %4 = "onnx.LeakyRelu"(%3) {alpha = 0.00999999977 : f32, onnx_node_name = "LeakyRelu_2"} : (tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
    return %4 : tensor<1x16x8x8xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 16 , 16] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 16 , 8 , 8] , \22name\22 : \225\22 }\0A\0A]\00"} : () -> ()
}
