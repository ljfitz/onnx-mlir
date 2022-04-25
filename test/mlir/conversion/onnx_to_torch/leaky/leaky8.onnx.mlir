//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x3x55x55xf32>) -> tensor<1x26x13x13xf32> attributes {input_names = ["input"], output_names = ["5"]} {
    %0 = "onnx.Constant"() {value = dense<"0x9F0626BCEBE528BD231C683C5FCCADBC98F659BC565BB43D4F9FAD3BEC541D3D84079F3DF414DDB9B1370FBDDC85863DE9935E3DB95018BD78DA473C14A238BD9325923D448E54BD813F0DBDB4C33A3D37C9A13DD19FC63B8C64D1BDB7DDC83D969266BDBBB5BDBDDAE8C5BD82D9D4BB570F2A3DF725B73DE94D8DBD46D20BBDDF0C30BD5FE8C23DD6DED6BDD2F07BBD7A64AD3C0A77AD3DF6E466BDBBD5A1BB5B87B5BD99B9863DC9B490BC50AE17BDF788E93D8FF5C33D3EC49F3D06BC93BDCCFD5D3CD1E38DBC9B20393ADA1084BC9AB1763DBF50A3BDF2A2B73DA3C1933D1E0CC0BDD10DD5BCB7ED3B3D2AF42E3D8BB96B3D16BFE43D849638BDC2146EBD13DCABBDCCF181BD0DA7A23C648BB93D32BA98BD82E5C63D626F1C3C4EF4A8BD3763AB3CCFDFDCBD8B5E97BD7B1EADBD19A0073DC816973DD8D3663D8018183CDB1EBFBCB7F7A63DFF9AE33B4254B83DE7CC8CBDDD0865BDC90B86BD6326CFBD251495BDCCA84FBCE336C4BD725700BD3F5DC4BD44E2653D16DBC4BDB59C9BBC3C4CD0BD3ACB6FBCCD92E23CF94D883D9BA0CF3D0681B9BC65AFA4BDF61E20BD7EB24BBDAF85B93D1BC9C23C2F78B1BD3E1FDEBDC5F9AC3D0D3B063D6DDDA6BD522DC1BD7475CDBD170859BD6317C63DDB63AEBDBDA6A23D26728F3DBE384CBC1EF6D63D5DC686BD604572BD0A7D6F3D3A4AC7BD9935E9BCD680823B84082ABCBD51383C1D251E3DD524213C6B9FB53D56B8953D6A2B96BDC470CC3DD7A2BE3D20BD5B3DAF0487BD47A5CEBCB702BDBC3762CABD9115643C6F18B9BDEC241BBDAFAB183DC820CC3BC11F163D7773E2BD23DBD0BD4D3BD9BDBA13123C75C546BCF37037BD67FBA6BD65D8DC3D2E22E73D09BF5F3C862F633D12FFE63C3D5D923D6F30B5BBD2AE943D7404BBBC8C29BE3D72C4603D8C95BCBD36171A3DCADD61BD6726583B77F147BDC26B2B3D4B9608BCFCA4EABD8009C1BD73DA13BD38B344BDBA320F3D8B7D2A3CE63EE73C3492A83D37CEE13CF05C11BDF93865BD9DEBD8BDD272113C41DF963DF4F5E23DE19C8FBD1DB91ABC0B54B33D31CABFBD1635B2BD14F2DD3DC745E4BD19403E3D109B87BC51C9273D206DBB3CC71A3DBDF889F6BC5B0299BCF6E71CBD39E520BD530E8FBD1FDC99BD93EE4D3D0E95E43DE29383BB562BAF3DA07FABBD9C1448BCC415253DAC2D1E3D09F14BBDAA7ADC3D694BA63D4F75733CC97D8D3DA775A1BDBF213EBD7859ECBDAD12E93D5D47943DF23CB4BD570F9C3C7375A7BD276D833D1E4A743CE137143D3AB2A23D64CB8FBA041D05BDDB35A43DEEA8A03CAAFA77BDA62E07BD8F76293A9E4B89BD3DA475BC780BA6BD5E5E963DBDBDDC3DD25D56BD67E7C7BAA3BC653D753CCEBDFA139F3B6B3B3B3DE76A36BDB41C623DCD4247BD7A9ED4BD3397AD3C6944AC3DA39A53BD4AEBF3BC7BFD9DBDF65D823C998E0E3CA463CBBBDCE1473C69EA8A3C23BE7E3DA059DEBCC4B51ABD7FA51BBDE238D6BDBB33B4BD96C395BD0760AABD5C66BF3CEAE8E13D66E5003D7EA5AE3D8662AA3C7450513D68BF0BBDFAA9EB3D74B4DFBD6122E6BCDEC0103D53301F3D458595BDC3D6A4BD3A578A3D24F7D0BC9C20103B593EC93C4ACD613D8F69323DDFDE653D9F0CD93C7BECDE3DD30A073D4F80CABC32FA823B5BE860BDCC3CA93CCBB14DBD573981BD0BF5983DBC4D133D82B5C6BDE99A13BD8E9930BDEB3FB63DA343A03C4B5780BD105A5DBD3586D83D7E64BC3C7DAC923DB35DF1BCE768A33DCA2866BCC1D40CBD11202CBB945502BD8A2319BD681F8F3DB411633D5A01A1BD57558BBD288B1F3DD3AF1D3D3425823D5548A83D1732D2BD9958A33D3CD6D8BDE821EA3DD034B83DBBFCA63D639F9D3D04F5A4BBB319D9BC0D24C93D515286BC221C1ABC622102BDB1CF883D78BBBD3C65507A3CB0154FBC519CF93BB1A7ACBDDF2A0ABDA8FAC23D26DE9D3D10869FBB43118A3D218DCB3DC660C4BDB373F03C0C34B23D2908CABCD6CD753DC7AA833CCE2AA63DD73A2C3D098D503D9282B7BD2374E53DC6A480BDAE34BC3D3D0561BBBDAA9FBD3E0C88BD0A193F3D4095373D49BEF2BBEE48583C160C55BD26668BBD70FC5A3D7A09D13D585AA63D7CCAB63DD09402BC5FDCAD3D3456B03D16495D3C3715F03C002A3EBDE66DC53D1A4F7C3D2013DB3D4801CB3DF59F8B3CC5AEB2BA422869BCFA25863D0E4096BC26A1A4BD921BE53C6615633D210E9B3C18E475BC56F60D3D2F4F8CBC4BDA523D024ED9BC975C9BBC5E89EA3CFD024CBC5CB2CC3D48E8753D64D8433DE7CAB83D5E1A62BC42129EBD8925E0BD27F6A2BD1F16893D152AE5BD1E31E13B84B2D23D4399E93DA365BF3B2B0B21BDA8CC11BDDF47DF3C373F5D3B51C9353D252CD4BD697E52BCC5AC4DBD67E635BDE13FC6BD60B8523D86181A3DEDBA043D9F3A0E3D8822313DE36C8A3D1DFE7CBC8A0ACCBD41EA40BC58B7283A3E5D4CBDE3E6B23B90140D3DB6AF9C3DD32118BD53CA9BBD38076F3C9435243DBEA862BD7B87143DDEE2033CB306473BFB18B2BD33F7E0BD97DC99BDFD2D6EBDA309C13DB5CD623D73A48A3DF2A7AB3D376EA03CBD32B03D232B1D3C30FEA73CB1E62B3DC2D8683D653878BD82A3DF3DF3F4DDBDCBE6E13D21BF2A3DAB76D83DCD7FA5BDA50FC03D83A1663C2200103DCAEEBF3D2DBFFEBC41FA01BC86A692BD86CFB53D870C93BDA6EE9FBC467CDE3DB9522DBD1AFBBCBD56D3963A1D19CBBDC62AF2BC1A4E74BCDED3CC3D6AE3A9BD07CE6FBD4A7299BD5E9E2FBD1C73CF3D538C79BCE4039A3D72FE183DD0BBB1BD434E0BBD11EFD6BD2E199A3D0EB21FBD512F7DBD9C50CE3DC4659FBD7291D7BB9E6510BD313362BCE0CBA2BDC206813C0581A63D26AD843D2E0C703DA8748F3D445AD73D0DB1E03D1A2290BD52B73ABD2950DB3DF9D05B3D1643573D7664C7BC5731BB3D2BEB25BD205943BDBA9DA63DEEFB463D56481D3D69B9493D99DB1EBDA588B83DD400DFBCBF11CCBD91420C3DF9C3E93AFF8B99BCC046CC3D1F21E9BDD0DFA3BDE5BCCD3C3AABA33D468E1A3D50E90ABBCD17343D195D24BD406476BDC74580BD398B923DD5E2553D3150B43C7ABBC3BCE101EC3D3B0B98BDD2BD333D23B5683C38675ABD7705CABDA43BC03DCFC8093CB67743BD2943D7BB3115BD3D1277A23DC41AA73CD83757BBC748383D759D253DC345953D796BC83C801480BC501CC0BBA7E5DA3DD864A1BDC0F88EBD153024BDF81C60BD166A803DC542E63C91B16ABDC29E983D0A77803D605F70BB25476A3AD272B4BDE17287BD65699DBDC33B633DEC2BC3BDAF7F86BD69FBA4BD5BF05ABD46E9C7BD5B01E13DA6E8CF3BB246C0BDC733903BBEA5DBBD0FB47A3D557891BD2C67AABD770450BD3B35C53D81DDBCBC2F3CFB3CF75DDEBD5EEC81BDB7DDC8BDE9BFF0BC81CD303DAB4315BD6B0A9CBDFA9D80BD247A79BC1D2F813D2AC1AA3D3B9CFA3CADAFC43D128B81BD0771A6BD9A6A54BD2D7997BD4916963D104FA13CC828893DF570AA3DEF283DBD106CF63A306AAF3C99B28EBDCCC4AF3D4D1F303CB19826BCACEEA13B3B01B7BC498FD93D1CA78DBCCD7177BDCA8F343D0B6784BDFB234DBD38CDA43DBFC85BBD27E619BB1802B8BC603359BD6A6BCFBD682FA0BDFC91793C8DEFB6BDFAEECEBD0C3C563DCE0D0ABD01B240BD209FD9BDB561373D303891BD62BDAF3DBAB84BBD1671AABD2407A8BD276E9ABD37F6503D26D6763DF4F2C4BDD06BD7BD6FFD963D3C5DD9BD3C69EBBDFC7DEBBD2762F8BC9422E5BD6DF24E3D87D9E43DF591B53D88DA8DBD65A8C93DEC4DCA3DE0DDA0BCBE1114BDC865093D37D596BDD343DBBD6B9D8D3DD6A8B43DE4BCD03D26E4C2BDA9AE16BD9C74E13D55BBD3BDF084F9BC2C6075BDFED8BABDD770C33DE52C143D1210DC3DC14C67BD54DB98BC897AACBC0A4402BDF13269BC7CDEDF3D4EFC46BC1B9984BC6021463DF74DA9BD94ACBA3C92054D3D951AA63DAB6E1EBD6EDEBBBBF8ED60BC1203583BA1B271BD61EC6B3B5CF740BC75D1D43C47EE8A3C20B8DD3DF9AF713DF29145BD5B4F3EBC0B1BA23CB5AA4D3C9664083CD456843D42D7133D3933AFBC7A200CBD6A2B87BDCA5751BD7B09263DED3F11BD2ABF55BD842CBA3DD6808CBDC479C63D3AB4DCBD33D9EFBC72F3BCBD1322CEBC008E603D1F9A663D235FA3BD0F8E93BD8AD4F33C315A3ABD779753BB53C0D83C1A1D2D3D046AAE3DBE54953DEEBB503DD02C9CBA282E163D25273E3D6789A73D6B8A8FBD92A68A39219D8DBD93376EBD4C1208BD25ADC13DFD61B5BD4922A03D7ECC9A3D0F32E03D802194BD4A33CDBD9B2AE43DC1C6F93C1D36C73D9DFFC8BDFB427BBD615F15BD29B2ADBC8CC87D3D50181DBD3AE8D33D5D5B3E3C11BB64BDB072A1BCC3C585BC36777D3D7FD0E8BD7AF7C7BD63790CBD314CD1BD1DD3C6BC04381B3D1662903D5E1F19BC0D4D29BDF24593BDEB36D7B98EC4A1BCB2F0E33DE028E63D193E5A3D9E9A65BD8951D73D85CCCC3DE392313D9D2CE3BDB55D053DBFF4AE3DB18D62BDA81C7C3C6B5E84BB0AEE303D539C5E3D2F6767BDD283AF3DC9DC543D111CAF3D7370BABDE83E783C99288ABD2881B53DD8B6B43C930573BD190CD1BDF971C63D7D1F263D1A9F34BD5431E73A31ACE7BA1BDB423D26E4D33D628BDD3D0215DBBD5CAE9F3D8558703DC8E1B13C8AF2543D788C703D07C1A83DD684E4BDA07E883BA143EBBDBF40B2BD3ECFE4BD69D37ABDC160A8BD891F1BBC9D3C5EBD019985BDCBAB1FBD98D026BD1C32213C2348B03D9B98B9BC835707BDB964D3BD4CEC98BDF265E3BD3197753D741BA93B0EEAEB3D647A45BD73D19ABC0AB8223D00ADA83D1EE5263DE8F6CBBDC612A9BD2CF9D1BD4645C3BDEB74EB3D599C473D3DCE67BD269070BD2CF7D93C53B6A13D460AD2BCA947CD3CB5B696BDADBD653DAFA6C13D51B5E5BCCFEB1E3D5036433D0AD803BDFB6E2A3D9110BFBCBB73D8BDD395EABD73C5173D4FDEB33D98C6F5BCADD298BD32ED1D3DAA96A33DF428973DC62FE63C790DDCBDD73FE3BD6E450BBD7A25593D6FD0443C1111A73B4619B4BD7B4351BDA75CB53DD291F5BCF24ADA3D24709C3D8B3C9DBC7F34933DA7BFD13B99B34DBC00F5C93D87D7D3BD022384BCD6B37CBD83B99CBCD20946BA9BB5523DDEB5B33D771BBA3C863BA73D8C80B83D994022BCAD2AAD3D837D803D372684BD85F444BC8D23A53C71DB37BC79E5CB3DF374A4BDE858263DB06E303DB0D90D3DA012F8BCFFE9B73D842515BDDC83C0BD9F4D7FBDDAB0713DD594D5BCFF83793D0F11DE3D0DAED3BDC385883DAF2A503DAD9853BD77381FBB960942BD42132EBD71C8B4BDEE0A1FBD5F19DD3C2034A6BD9D4FADBD4AB4A0BDBE75723DB52EE13D9859EBBD887040BD2C1A5FBDD753A93D90E2DA3D0A2BB23DEDACCBBDE56192BDDE7A633D1454BD3DEECD923D6852323D0F6CE03D63A19ABC9018813DFB03A03C9B555CBD7A4BD63B5B9A60BDB49B84BC07689ABC4246033D94C0A83BE381BF3DB9C93ABD080E4CBDADB0363D8A1AD5BD72D20BBDDBAF2139A26ECF3D23E96F3D73DAB1BD019DF23C450D503DAA246EBCB9BB923DDAC2BFBD9970423D9288663DFB65F23CC742BB3C9E42D43C439FDE3D89602EBD71E0613D369F683D7361C0BC2C75A23D1B69B2BDBD0FC13D5F2F2ABD00CC133DA2AB3D3D8279B23D0C1317BD1E87FD3C642912BD8BD780BD074EE1BD3C6854BC2F96873C68C1BF3C7A310CBDB6E409BD82E1A13C3A14853DBABF4BBD7552AF3C67AA343D2548B03D00F5A2BD9421E8BDF7741DBD5089733D450B9CBC7906003D84E3CEBD9D541F3D4D3BE6BD2F5A79BD3016553D1FC57A3DF686D7BD8E0247BDCB1265BD5548C13CE64859BD8698E43D8EF9213D0E9EDA3C0FFCE7BD9DE219BDABEB4ABB7ADAE8BD8F7DA0BDB4C5DF3D97AAC63DF9A718BDAD304BBDD4ED2EBD35E4353D50A8AE3AFDCBC43DA39AB6BC6C7C51BD1F10E83D192C473D73BDD4BB9479C2BD75FFA1BD8DF5DCBD7CD9DC3CD0E9E53DDC638EBDFC2DF03C7F760DBD87D96BBD5E474FBDCE686F3D327F37BD292E81BC7DAAA1BCD200033D0D86D7BDC904A2BC382485BD173F9F3DB5A7933D2E5D3FBDBF15263D19F7A0BD27815BBD3C608FBAA703CFBD5F1A93BDB30C7FBD66FBE93A57F485BDD1B8773DB793BDBB16EDA03D56864CBDE60F043C37040BBD5D4ECB3C4ACC983DCD488CBBE703543D5080973C5A078A3DA4118B3C9504853BA7C517BC5E4E36BD869681BDA876443D666CA9B9ACE8BDBCEBF2CB3D36DD343D97E50A3D191D253DF4EE37BD5D446D3CFB0150BD75EAD53D6807243DEF27083D9B1AD8BDFC43513CA33E0D3CE2A82CBD95D71B3D6E1C713DCF49C23D3FE22BBD1BA5A0BDF8FF65BD0A281B3DDF4112BDA73CDBBCC4575DBD04E36F3CAE5CEA3D242E1C3D8EBBD33D6C8D793DA02F5DBDF5060C3DDAD1E13D4A2399BD7B2916BDF66FB7BD852C9E3C688B773C3F48D5BA0410D4BCBC47BE3CDF21AE3DC5B1693D312CB2BD0BD29F3C22CBD93D25C6F03B8CC6D33D119140BD6AD8CFBDDFC1BF3D04AE6EBD5E81D23DC3649B3CCC76F7BC92068FBD2C4D9CBC23F8B93D015BA5BD6F7E8BBD05514CBDB71391BD9051B9BA577F3C3CF7B1E6BB864F623DF78ED7BC2113C33D9372A53D7B0E00BD551BA23C2A59C5BDD358E8BDA290CD3D1DEF013C43A3CABDD851A53D23C9B03DE159823D999EB43C1475133D2AB58E3CD6F709BDF54F5EBAB4108DBD58E9BBBC07B579BCD73B8CBCA08DDBBBE404CD3CC55D963D1A964B3CBD91C23D143C0F3DF5A970BD36D27ABDD78B9EBC74952D3DA214B1BD03E846BD27300B3D06EEBA3AE1DF913DE3BE4CBDB28F0EBDA1A874BD7B8A0F3DDA98A5BDB7348DBDA057993D8FCF103D09691A3D40DC403D0D385F3BC8C3AE3DF459EC3DA375A23DCA2396BD5B6494BDDCD57B3DEB0FB9BC176B933DB912CC3DBE57613DDE4B69BD542AD8BD973DF4BC5299643D9C4503BDBB69E7BD5B62A1BDF4D65ABD5D41AF3D0165FBBC17C8A63D8B8BB4BD620D14BDA39367BD00F3C73DEC30D0BDED09A1BB90D5953D3ACF93BDEC6A973C79016ABDD481523DB088C33D6E1F043CDFBAB03D472AB83C537348BD00C258BD4E01B7BD1D2AFB3C12FEBBBA01FB0B3DA45C28BCD075C8BD7978EC3DB841A4BCF8DCD5BB0D43E83C491A973D05F495BDF12C93BDDCD482BD338E5C3C77AC8DBDE874A73DEE40F7BC8FB287BD2894443B6C1D32BD56AA2E3D57D728BDACC13EBA2FB6CFBDE103A33DE9FD313D4D8EADBDD7672D3D96905FBD730A6B3DF35498BC2E12AC3DF166DC3DC1F8D0BCF72E83BBD761ACBDF196E8BCBD258D3D069F2CBDE01571BC0009DE3C963E843D7D8CBFBD5478CA3D5DE4C23D58E7D1BD53E5E2BD738BBC3D477FC73DA82DE0BB47BD543C50B0A5BD5F6A76BD4EBA903D96E89F3CD738CFBDF1530EBC1F47723DBFAC78BD284578BDCD01FFBB76B3593DF17E3A3D5E89E53D03F59DBD875BDD3D4F6EBC3A23F8853DDA3DD6BB5B5C4E3DE5EBD73D2D46963D9FE414BD02687A3CF292A4BDB0EFAC3D1D48E03D9981433DA774C6BDF670C6BD0F7B35BD837CA73CC38B4DBDB1E912BD884C153CA6B9EA3D0B98033DA846F33CEE09D43D2B4EE5BDCFDC01BD482A913D5B0D053D807B46BCB766D8BDDF63C93D88F6863D3A84AFBCAD23C9BCD28987BDF1DFD13D97729ABC4124D33D6B74A8BD2E3204BDECC2D83C084C843CF4B50B3D039BA23D3003113DD8B5823D9D6FB03DAABCD63C38EB223D66BFE8BD19AFD3BD58359EBDA806BFBD2607693D8D9FC0BD25DEA7BB930A863DE2FF243D5060D8BCCF91A73D0ED0A7BC320F21BDBA7B1FBCE0D080BDC323673C992EC6BDF141CF3D4A73593DB599103DE3F9CBBDCC07D5BD1F89AEBDD6E1283D4021ACBCC1FD533D9C1EC4BB57257A3D5671333DAD93FDBC3C3EC1BD3C58EC3DD6CD8ABDBACBA13D8A12ADBDD506993BB9B4EB3CFE1FB83D401FCFBDE9BAAF3D186F923DAB768E3C7CD6343DE4308BBD41EF073D5114BDBDDCEACABD8E0C2A3D577A42BD996CAABC9373CB3C3D49E53B90756ABCF728CC3DE9FF4DBDB2B03FBB726FE03D930C38BD18663CBDEA748BBCE93BDB3DDC87923D10F1AF3DDCBF323C8BFF90BD275A2ABDE6248A3B561480BDBA74E8BD4BB847BC2A7F47BD76AD7A3D040F883D30B5873D4A59BFBD734CB8BD2509E73BE02EB83DB613ABBCAEB62C3DB5310C3CF312E1BCD68786BC787BBC3D880F42BDCF35483CED95653D335A4CBDBA35E9BD7CAF9D3DA65655BDA2CCDABD7917E4BC33352BBD1FDDC6BDD088D03D28877ABD6AEED2BACF9A72BD815E983D2AC2E53C37FFA6BD662955BD95CFC23D558E333D88F8D33D95A005BDA1BC27BD98E6A2BDC8DB91BBF728C7BC52C5CEBDF79F6CBD8F442FBDB3B3E3BD6C7452BD6B70A43DB9D359BA0EEDAB3CD97DB9BC622C7F3D09916D3C3690F3BB5EB128BD61D2BD3D925DDF3DFF10DEBCD5709C3D5AE05E3C0455D13CBDA1403DC96CEB3BF08FD03C7F5A3A3D0FB7DDBD5704E9BDAB1026BDEC76C43DF022E2BD6516F13BBBB6C5BD9CD92DBD3D2488BC8E91C83D3182C8BDD7329CBD5D9FCDBDF2911EBDB10AA6BDD2EE34BD6BBA37BBF01CB43D129633BD640E1B3D6999D43DEA3EBCBD21C3BB3DF9A224BCB1AE34BDE713B73DAA9CBE3CDBF5D73D7CE9E23D3272C23D80D586BD9CA4A73D521DE93C412B58BD413FBC3C89C8A2BD3D9BAB3D0525AF3B224C583D8754863D597ACDBD95E18F3D6835C9BD716EBBBDB8C02FBDD6FD923D230F8BBD012907BDA865913CF590A03DFC5D68BD322A6EBD4BA9C23BAE62DABD8E74A6BDEC9CA8BD9C4E283C59D1DBBC726D9D3C5923073C5A2B0EBD1B8C49BD6FEC36BDAF67F2BBD1ADB7BD9AADDB3DB8329B3DE3DBE23DD38C5DBD0BF28FBC21B3BBBB6BD2503DDA4790BDEA2AB3BD56422FBD3EBC88BD41472C3DCF9F993D41A5013DC19CE93DEAC162BDA00CE23B4DFEACBD82C3C13B80A5C73DE0D3E7BD20DA7E3D43F6CA3DD3F3B8BD9988A83D493485BDB187B83DB94414BDEE5BA6BDE2CDB93D37182BBC922E89BC1C27B5BDD9E5C4BD9268883B9C17A6BDCD48E6BC6C9FBBBD5AE5D13DB8E9453DFE9E73BB71E2923D2A3D183C1C8BC3BDD8FFB23D66DD61BCBE5F3D3DB14F1EBDFC39C1BC51713ABCE39BA4BDD13CE1BD3BC0A2BD2D06AA3D27BA6A3CA59B28BDB8B95B3DDAB8A53D2CDDA13D496C50BDCAC8A4BC3C05933C16BDD23D503985BD16559E3DF33EDD3D51E42EBDD89D953DCED49B3D0A31FCBC8161D73DC0C9923D733F113DF7B1533D8860C5BD22695C3C99FEDD3DB638D9BD264AEC3D22F7A03C52276ABDA97943BCEF38D1BD56260CBCE26EA9BC2302813D9A75CEBD86406EBDD2AEA7BD2888ADBC22F21DBD06EE533DBE23DD3D73E778BD8A68CEBD5866ECBD427FD8BD76C8AF3DDAD20FBDFE8645BC5C76E03D569CC1BD3A0FBBBD22EA9ABDF688EBBC458DACBD164C63BD0E618B3CFAE284BD797E14BDC2433CBCC90C08BDDD3DB9BD583E55BDDC0F3ABD6A25783DFCA2DA3DD02CAEBDB3D2A8BD14D9133DC02697BD73FDAA3D5DF9643D804FEB3DD00B8ABDCABDC23D0F559DBD339000BB8214A8BD4ADA983CD7FC283D11502EBDDEF10BBDE8F1943DFCE096BD2E13EB3C444A143DB52DDDBDFBCCE03D0E1CB7BD466A12BC3645E8BDD598D9BBC6AD5E3C13BA833C5514893C5B9ED6BD1672983DA98C5F3DA85ED2BB63FDD3BD8A90383D21077A3C8F60713DF04ED23DAC33963C83F28D3DB1F98BBD8F3EA83BB9BDB7BD8E31853CB0A7BCBD3EB1F1BC1BFBAE3BD862CF3DB650703C6B5D973DE01FE8BDD162A83CA5D3D63CF9CD8FBDB8EEB9BC2BB769BD5EC6EBBDCC72503DEB0499BD589ED93CFB11953DE2CA5FBD59AA8E3DB12C9E3D006639BDFCB4EA3D0D5DCD381F8915BD4A99B2BDACC3D83CE782863D2D3C563D624826BC1C5CC03CB0129F3DC87684BA85E3EABC7A2231BBA9A160BDD3947FBD6ED0C0BD5F2376BDD2098A3D189B4BBDFA20D83B03CA91BD1C53B8BD071AFDBCD6E290BD6365C33D465EE3BD0A2FD93DA6EA8E3DC91DCABD8C8A113DF315C93D149E393DC19CB23D6229653DC8517EBD52D3083CF87017BD7F86B73D8F1CBD3DFDAA273DA6787ABD2A5782BD34ABD5BBDA81E33C488AF73CF5723BBDA4AF583D5C95F03CA7C30A3DDD1ABDBD1D48813DA2B871BB5F20EC3C2064483D1F8121BD6CA8BA3DD060373C299D0ABC5656363D0F729E3D404DCCBDC30C473D89B4B23D7DD6C93D340ACDBDA866B53DD133F9BC7DA85FBD3268CABD8E6DC2BD9AE4023D88A690BDDF84E83D3EC689BC1E248ABC20B49BBD038F983DFA4A453D13BA81BD8F709C3D8F4D2CBBDD07C4BCC01409BDCC3F75BD57D1453D7C70643D83D081BC5F58A73C7081AE3DE4B56B3D3AE9A83CA037AB3D7C60BDBD8DCDB83D1B87C5BC6BFC8B3D3FE9803DB5ADCBBDF5D2D13D0F7D233DB897223A4362253DA29089BC3E41B83DCCDFDE3DFCD8E03DAB97483D445E113D68C145BD3F7694BCB02E6C3D4D429BBD5811ACBCA3ABE83C8FB3E43DFDCC653BFE25F83C45D6EABDF44AADBD543E803DFD36D1BD23B687BD4A7762BD240FD23DE6F169BDD156B0BC137D3CBD8DED563DE5D7C03C038B383D29871FBCC8E3FDBBCB1651BD99BFDCBDCEBA23BDE43A1F3CD1C7A63DF2DA7EBC19EABC3DB09FEA3D570417BD341EB1BB7198F23C42E01A3BBF9526BDA02E883D86B890BD9AAEC33A1B0192BD9D8E093C3CF66C3D0818A7BDBE538D3D4646EABD5DEF71BC48911DBD808A5E3C45A636BD5FF754BDE6031FBD0578373DAD3BB9BDA23979BD5073BD3DF6B4053CA81A67BD7D3B9B3D78C792BD82776B3D2925933D3370AD3BEDF95EBD346859BDA32249BBB48898BDFBB3A33CCE681ABC73EBEB3D68B129BDD61C16BCC26A45BCCD8ADCBD1FB28E3D6FDEE7BD308FF8BCB9F898BD"> : tensor<26x3x5x5xf32>} : () -> tensor<26x3x5x5xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.0808858945, -0.0512616299, 0.0775271505, -0.0734893903, 0.102354176, -0.0399804935, -0.054235097, -0.102388181, 0.105638646, 0.0340362117, 0.101155154, 0.0976797193, -0.0893388242, -0.0812426805, -0.007843039, 8.669120e-02, -0.0671022758, -0.0834114403, 0.107561201, -0.0890404731, -0.0617868528, -0.00462248689, -0.0916779413, 0.0764199123, 0.100574143, -0.11029733]> : tensor<26xf32>} : () -> tensor<26xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x3x55x55xf32>, tensor<26x3x5x5xf32>, tensor<26xf32>) -> tensor<1x26x27x27xf32>
    %3 = "onnx.MaxPoolSingleOut"(%2) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_1", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x26x27x27xf32>) -> tensor<1x26x13x13xf32>
//CHECK: [[ALPHA:%[^ ]*]] = torch.constant.float 0.0099999997764825821
    //CHECK: torch.aten.leaky_relu %13, [[ALPHA]] : !torch.vtensor<[1,26,13,13],f32>, !torch.float -> !torch.vtensor<[1,26,13,13],f32>
    %4 = "onnx.LeakyRelu"(%3) {alpha = 0.00999999977 : f32, onnx_node_name = "LeakyRelu_2"} : (tensor<1x26x13x13xf32>) -> tensor<1x26x13x13xf32>
    return %4 : tensor<1x26x13x13xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 55 , 55] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 26 , 13 , 13] , \22name\22 : \225\22 }\0A\0A]\00"} : () -> ()
}
