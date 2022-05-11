//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x4x12x12xf32>) -> tensor<1x12x12x24xf32> attributes {input_names = ["input"], output_names = ["4"]} {
    %0 = "onnx.Constant"() {value = dense<"0xF828F13DCE69D03D2B5C903C96E5FD3C7E23073EB88690BD641526BEDE0920BE803581BC1BD053BDABB3D7BDD479263EF8ED0FBE164846BD9BA03F3D00D292BCD0EE3A3DBEC7183E5CF5253E7BC4183DABB0013B787F0ABEF6956DBD00D78EBC1B90263E80A83FBC0BEE05BE5B31ED3D10543E3DAB059F3D36707FBDD79A0C3EB665C4BCF070D6BDD60CF13D53E4823D7B8664BD7338B6BD40CDF0BCB6DD4EBD5BE2B13D6E6B00BE9A07293E6BFC38BCF372A6BD403A0ABEB851943DAB5C82BC70688ABD00CCFB3C5B819BBDB64D8A3D8E9CC2BDF318A4BDB6617A3DD6E1DBBD509E333D8EEDB83D064C2A3E061A873D2ECCFF3D0097323B8B0D083DF024213DE723253E163C033CE8B123BEB04DF1BDA687B2BDB8B71C3EF3D0993D40015E3CABF8D93B10B70DBEB0DC323DD86CDABD9BAF603D0096DA3BA65F0F3D6B5635BCEB74143CABFF52BB6B8EAABC605E093EF6E75CBD94450B3E2BA825BC6F5607BEC0D6593D10A627BE1EFF143E2866B53D1696CF3C2027943C96D4E0BD563B5BBD909BBFBD80C180BDFB57EBBD86AF1EBEB6360ABE7645B6BC46D4BE3D4B59FB3D9B3A76BDF08505BD406107BED63CD0BD3070B33D4BD906BE7EFDC33D209B203EA0EDDFBC000A043DE37DEA3D800FE8BB166CADBCB01D70BD1BCD4EBD90EC053E8B3EBCBC46E846BD3020AE3DF68DB93D03DB1DBEA007F53D00D5793B6B0EA3BD68BFB43D805D033C6C7A043E0C9023BE005B59BB530A9F3DC326C03D061A2DBD5B18103E086AD6BDC0B2ECBCF078CCBDEBD970BC8B02F93C4BAB8F3D30DB74BD8B60C83D03EA22BEFBA7CE3D0EE6E4BD606C603D1E32AFBD56B0A63D2B3586BCBC18213E137EA93D7BBD1B3E4C9401BE800256BC80D1BD3BDB6F74BD722018BE200D3BBD9E79EE3DE671C73D8364C8BD6C5C01BE2B68DB3DF353E03D6BA3143DBB15193EF672C3BC2A281EBEAB386DBC1800043ECB85813C83B097BD3C3104BEC0B23ABC8B85F8BC2B139ABCAA791F3EA6E85E3D301498BD3BC1ABBD764C753D905B52BDC315B6BD835FEA3DF00FB73D6BE4B33D2873033E764A20BDD6E8F1BB80AF6B3DE041ECBCC0B62FBD0033243BB69C00BDD8F7F83D705D6E3D56881BBB0378F1BD169C0CBC66DE64BD00C51FBB56627C3BE242103E5678AEBC10A604BDBBD5AE3D68BBBBBD3B1CDA3D60009C3D286B183E0B1E87BDEB012F3DF6103C3DE8D3A6BDE8B4BE3DFB5539BDAB4D673D33E0F0BD8A830A3E76FA92BC8695163E7BCFCCBDA050C13C401C433C56E3ADBDFEAED3BD93EAD8BD7B0545BDB05289BD005AB03CD6EF4C3C8BF7353D18F68FBD8B1DA33CE3A0DABD963D4EBC8E10D8BD468EBDBDE4DE0FBEC0F1933C73752A3EAB37133E760872BD28E9B33D1B41413DCB69003DBEF9223EC0F76EBC6400153EEBE5373C880005BED600A2BD2A27133E3E9B8DBD9364B0BDE839873D4FA0143EAC5E03BE909D7A3DAB2E68BBEB94683CC0D29A3DCB46203D76D4733D2FB409BE0BEAE63CB6B2C53DC62208BE8EC5ADBD5624373D209BD9BDFB396A3D1BE0FBBDD8249DBDD0D2603D8BBB943D4EFE843D72E61A3EBBC64EBD107E11BDC0682A3D7622F7BC3823A0BDD6B38BBD567C233C96EF1F3D1606653D566BD2BC36A7FDBD6B4A173EC7F7253E069D31BDA43F0A3EEB3F98BCA0C3BDBC33EE90BD20CCA8BC80890D3EA68A35BDACED0EBE680EEABD66312C3D589A973D1050003E3647E33CFB74DABDC380A13D80D8F83D001B48BD00C09F3B07F5213E68E7B73DE4F11F3EBB641DBE8B451EBE88538ABDB00605BED6C2283C230683BD36F702BE20383FBDE6E82FBD160CE1BD78040F3EC09C1F3C8B9CAABC53DDCCBDA8A4ED3DDE6806BEE6A739BD563D0DBD8B6079BDC06BFC3CF37BB23DFEB1D6BD58C790BD4870BF3DAE40FC3DF66FE13C5623C03C0B66CCBD56178B3AD677B53C06326D3D84B6053ECE08A8BD033E8FBDA83B833D4016CFBC70731CBE84890C3E36278C3D5877833DFBA9573DD6FEF83C205EBD3CAB3B25BEC68318BEBEC4E83D801EDFBC6352063E5809F3BDDB7F00BD38AF873D0E0E1CBE4B80023D9331D2BDEB54E6BC57DA0ABE137FCFBD1024223D5BE812BEF89021BE58A4B63D1E1CDABDBBF91E3E5B5C46BD80E78EBB3677193E239A1ABE83D1913D673E17BEEBF7E1BCC8799ABD5690F8BD2849F2BD84E11B3ED8FCC4BD8BC1EABCAB03653B8882A33D7659BCBDAB48CD3CC0A571BDAF4407BEF7A80ABE6B01163EAB79F73C13EEEA3DC36726BE93FA8FBDC6404DBDDF841E3ECFE725BE6BA771BC1B5A23BECAAF043E6012ABBD10D2EABD3A62043E2041333DF3F899BDFB03F3BD77FC21BE5824B0BD9B3E553D8C7A0ABEC628333DC8DACE3D3B69D2BDFE2F92BD004069BC98B11A3E12AF0FBE3EBEED3D1608ACBDDFA813BE503DA5BDC305F9BD"> : tensor<12x4x3x3xf32>} : () -> tensor<12x4x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.0787997097, 0.0934591516, -0.106071398, 0.105203271, 0.0158753395, -0.0936289653, -0.15529643, 0.165667564, 0.0695708394, -0.0372162871, 0.166498899, -0.0204498172]> : tensor<12xf32>} : () -> tensor<12xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x4x12x12xf32>, tensor<12x4x3x3xf32>, tensor<12xf32>) -> tensor<1x12x12x12xf32>
//CHECK: %int[[DIM:[^ ]*]] = torch.constant.int 3
//CHECK: [[INPUT:%.]] = torch.prim.ListConstruct %8, %8 : (!torch.vtensor<[1,12,12,12],f32>, !torch.vtensor<[1,12,12,12],f32>) -> !torch.list<vtensor<[1,12,12,12],f32>>
    %3 = "onnx.Concat"(%2, %2) {axis = 3 : si64, onnx_node_name = "Concat_1"} : (tensor<1x12x12x12xf32>, tensor<1x12x12x12xf32>) -> tensor<1x12x12x24xf32>
//CHECK: torch.aten.cat [[INPUT:%.]],  %int[[DIM:[^ ]*]] : !torch.list<vtensor<[1,12,12,12],f32>>, !torch.int -> !torch.vtensor<[1,12,12,24],f32>
    return %3 : tensor<1x12x12x24xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 4 , 12 , 12] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 12 , 12 , 24] , \22name\22 : \224\22 }\0A\0A]\00"} : () -> ()
}