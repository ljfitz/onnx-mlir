//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x3x50x50xf32>) -> tensor<1x25x6x6xf32> attributes {input_names = ["input"], output_names = ["5"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDC79E53DA6512FBDC0ED91BDBB12BFBBFFAF463DD9919C3D8AA063BC66F78ABC4BEE8539169F82BC6EE1A1BD87C15CBCB2E75ABB3E154EBD8C60ACBD3CC6E2BD23F8933D2C3C91BDF1DC94BD7DE67BBDB0AF20BCE5F1823D26B8A8BD411BD1BD518B9D3D3D53D43D75C49DBD9EB583BD2E907E3D0A371EBDEBD5D8BCCEDD973D68237E3DAC36573DEE3EE33D1C77243D7C0789BD5D9CCC3DCA7E0FBD605DF33C744DC03DC45492BD837675BC5C8FB2BD1BFFB2BDE3C4BA3D0099DB3CC287623D35384C3DBC3FB83DA10F613CC8C585BD3F0DA93D4A7B713DB1F870BDA5E8E5BC041EB33D81C58E3CFE97A13D1ECDB63D4DB38F3DC9F1C0BCFB08A83DA78A9A3D0A76BE3C969428BD1655EB3D5DCD963CB5D0E03DC11905BDE47B40BD1A8A343D8915333CA15D9E3D6A82B9BD5242B7BD9BB7953D6826D4BD04D80B3D28F4E5BDCF00BE3C1F8B133C2D2A51BD721E943D40BC0FBD1332C1BC284EE9BD484B71BC3BCD52BDCC05713D1B8B5EBD7CC923BC993FBCBDD3594BBB9498113C1268E4BD3BB51B3D5ACFAB3D7FA7E0BDADC0CFBDD892833CBCA72A3D259D823DD7CB0ABD8A0198BD12B2D13D7851B2BCA9AD043D4FB7343D5028C33DE48DB3BD04A7793D04E54E3DACE8C5BD5DB7F13C0C89AF3DD923BF3DB446C0BDC436E7BD832539BC077F2BBD337FA23D2D4B9EBC078DBC3DDC598DBD6D8E7CBD8BE5DFBD6652C93DA66CC33D1B8144BD6CC1DDBD4102783D6F86C8BC814B863D2E40E33DBF04B03D16C5BBBC1EA6A7BDE5034DBD4A947A3C513939BB2C1B313D6710CB3D74FE01BD3F26343DF26387BD1942A03DA9F284BD2AA7B8BD01CA883D72C2223BCB7A8CBD1139A1BC250CACBCB4932EBD1735E0BDE8D3DD3DD31721BDB13A463DD8301BBDE5EA1BBD104DC7BDB553C8BD586E4D3D3166A93C6FF4A5BD0AB8453D913BB9BC7CB5983C6F8D1B3D9DD8363D05FFB23B86E0453DC16477BD12B4BFBC35DE6BBD641035BC61198D3D03396B3D77AE143D82588BBC9F4EAE3DA8CBBDBDD6ADBC3DC3489DBDDA20AABC1D50D93C01BD683D46CEEA3DFCBE083DB97F8A3D4907A13D643680BC8547D6BC32648DBD712DAA3DB95B21BD2506BCBD2ED8B9BD656183BDEF7E233D029FBBBCB2E99BBD55AD82BD29AC44BDF65A2F3D803D943D706CA63D1C6CD9BDF3F243BD24D996BD4FBC8A3C6B8CE8BD6CEBA0BD32AAAE3D6CED9A3DCB113ABDDB35BFBDE9F0D8BC3BF4DA3DC2221F3C7B297A3C0221523D6348FE3CAB41AC3BE7BE97BDAA1DB7BDB68BAC3D2981B13ADA938E3DC554D4BDBD1A82BDABDD84BBA887273C4035923D80B1B53CEA44B1BC87CE9D3D73CAE5BB884390BDFE55D53DA89BDABD66B4BBBB5763CA3DC33A9CBDDD4E7FBDC9C3A53D538DA33DFCF9983CCAFF983DEE05E6BD4F6ED03CD22B2FBD725DC7BC5736523D21EB2D3D0944ACBC12F04ABD9E5188BD3F65E3BDC172F7BB13B569BD05A5993D78C1E6BCE2F6ECBA72D680BDD3136DBDAB94E5BCEEA6BF3D80ACA6BD884EA13DBF8DC8BDB831B0BD1E04EBBD267667BD4C0C5CBB1B61CCBD4D29D43B5CD9C23CBE40ACBDD3C1B3BD6DA28BBD6414D53DDE41853D6B718C3DF6A12F3CC728743DDBB1993D2788B9BCB372A73CE0D6B1BD998F303AAC10A4BDB91A64BD6DF7DABD4D8D90BD57F0E93D7A688A3CC6E8B63DACA49CBB8D58F43B3718A13D7AF292BD13FC923DF3FE93BD173EBFBCF21873BD33C18BBD0349143BDE90ADBDF437B1BB88C559BDEE34FE3CF3CA883D790797BC622ACFBDB62A60BC931AC9BDAE6A343DC6EEE73D297A683DE12989BDDA6DDCBD1C10A6BD0894BBBD6B57BBBC49CAABBD0C7C41BD3C0EDB3D3458CB3DFC0CB53CFC6639BC14F9493D94B1383C138E603B9A7A923D2B088CBCE7ECE53DF9444F3C3868A93D379ED13D9BE5B6BD7520943D9F00E03B95981A3D4377E43DCCE902BCF6A0D13DDCCB163DDEA1363D44AC9D3DBEC79D3CFC048FBDFC398DBDA957CB3DF3C0C8BBADB772BDFABCA33D4E76593C7D13F9BAA66F46BD44496FBC131C51BDAF4CBEBDBADCEB3DFB6CBC3DEF45AE3B1C706E3DB8620EBDE0DCE43DBA0D603D00884B3D9287B83D1C9E6A3D4E94A83D2E891ABD14F1B13D446C913D55CD523D463ED93C2F5DEF3BF78D1C3D180AA33D2E464C3D8E61A3BDBE72AC3CF60F66BC0C5BC3BDBA36B03D9D97A1BDDF1C053D7247873DBD2877BCEDD2A33D60B7623D4BD27CBC7D96DD3DEFDA1A3DB7BC493DF806B5BDD1D131BC604FA1BD6E4884BDCD4F443BD9CE96BD8AE4303D5A45BC3DFF7B83BDB4D9523D2F0FB4BA5350903C59E7DF3DB99E903D38B7B5BD841BADBD2C6AE1BDBC6D18BD99B0CC3D9D8DC33D5AD08CBDF3F225BDC926CCBD9CA5BFBBD836713CDAF421BDC886E63BE20497BD6FB9DD3DF1725FBA4A3E8BBDEE66233DF893C93C5E738A3D373956BDAC44B33D242E123D77648ABCE6D3EBBD0D1BBFBD7BBDB53CC331C73D0D7320BDA35F57BDB46EB5BC8E1F1A3D18D3CC3D4678923D9D56D83DA16EC5BB5B30C7BC9B34763D12FFAB3C026C423D3435A9BC4FCB9D3D5752673DA20BD13DC9CD323D9E3647BDBB9BA8BDD1B00A3D92D7E9BD63918A3D7671883CDA0A913D826D743C0F45093C5C6375BDDC4F35BD0B2CDFBDC823CDB94DF8ABBD6B2F7D3C9ABD073D7692E0BCCD1DCEBD2CB07C3CC68280BC8DFE83BC0A326B3D54C20FBD131B8ABD0DC9E93CDB96B53C58BDAE3D34E700BC2C01BF3DECE4BC3D65FC00BD83B872BDFCE4D9BC2898E73C398CD73DEFF5D3BDFA31A23DC8C781BD9F14A93DF36E3EBD0806343DA4C5E03DF7DDA83DDF47373DA2B641BDCB70ECBD36C26E3D9BB18A3DDEFD223D7A4E7E3B4C138FBD8AEEF7BCB0148FBC06CAB13B81987F3D02D3D13DB2C0EEBC7EA4A53D8BE57D3D68DD4A3D119935BD3439A4BDA18589BD7D8BBDBD5B9EA93DF30D0E3D5459483D22041EBD677A803C827DE6BDE99B763DF413AA3DE24F8C3DEABA2BBB27C01CBDA094023DD3B0EBBDA408E43D464509BD9EE8E93D1125EF3C2DEFD3BD4E44BC3CF528B03D01E1BF3DFB85363D9EA8F83C4AB1913C768D493D17729EBB7D7454BD2E2B8EBCFC2D9D3D84F0CBBD50D90CBC6014A1BC797CE73CD8E7B03C3C5AB33DEE4899BD9E5FECBD5F69A5BDAE95D13D2941B13C51CCD6BD0B7A1EBD203F183DD9EEC63DB226BE3DDE3816BDB5E18ABDECE446BC531FCCBD5FF5ABBD8AD4BF3DED09AB3C17661A3C7B7B753D4C05113D00A0FD3C20A9C83D16358F3D12CD863D531E75BD4A0D8F3DE39016BDB9CCA9BD7F6CE43D612FD73D5D1FA53CCB4EDBBDDAC9ABBD2AB958BD28B260BC956B983C387613BDD8665ABD1091B23CE1D8523C10509EBD01F68E3C7DB087BDBF6D22BC3DC1483D30C7CE3DB833733D1AC5223DCDD5953D9E065EBD5C53E93C0B39A9BC95E0CF3DF142D23B21FCA0BDD5B57BBAE5129A3CF61DD8BD5A7339BD23DFB73AB949BFBD6639D8BDC06ED73CD84DBFBD68BACCBDCED8A33D2812E63D4E65DDBDC74A883D7A9BCEBD33D6C6BD41949B3D26CA00BD632A6CBC717E3D3DC03F6CBDC9FC84BC87F87CBD2BF7AB3DDFB4E73D31598F3C454F1BBD7E69B03DE46F95BDF02271BDCCFEB73DA02E80BB59E0DFBC41A2943DDF422EBC58F0E83DE0D8D3BD111BA83D5DD7303CE10FA2BD4CBD8F3DF1C4A0BDFE1B853DC2F5443D86DCCB3DE3BCCD3D2175A2BD0DDEAC3D1829D33D8D9D8EBB89D72B3D7D7520BD8F46A2BDAEAE37BD0D6CBEBC7511D6BD7DFD943BCB8C893C8F6BA8BD19113F3D815E87BD4D8D54BD65B752BDF0813C3B1E608FBC862FEA3DB05CD43DB7DBA6BDCDF0553D93E9273CA0C3E2BD7C98BB3C0E70B4BD3BBDC43DDFC8E1BDCFF8BDBD267686BD6C96C23D5C453CBD9911CBBCD2D8D3BC7CAF82BD1BC9AC3DECA388BC3A11B93DE5F5B3BC93A1AD3DE8FFE7BB3034813D2A712DBDE7A3B63C9579FCBC14F13C3D4896BE3D9934C4BD90EAB3BD27720DBDA2AEB4BD9004D1BD25FF6A3C0EBFDA3DB68CB0BD368618BD271CBB3C1E5DFC3C180EBB3D3F0B78BD59A80E3C674984BD593B35BCC18DC23C83AADABDE9395B3B38DC853CF732C83D3B757E3D120EDCBDE90366BDD2D484BDDEABB9BD83A19E3D92F1363D9C272EBD1BF5983D108C993DB10BA03C633575BDF3D996BD056185BDF356153D12DC343DB5A62FBD9E407ABDDB5C0F3DC4DCAEBC585098BDB29BE8BD84D7D4BCFF6786BC1CE4DC3CC1E48BBDEADF41BDDDBDC0BD76CCA7BD5B4928BC0E9718BDC81A823D8592283D752BC33DF747DCBCD1866D3C1F2BC2BD4394C0BD7C3E4CBDB4C9723D5621F13C7E8B8A3D6A1701BCEE1ADC3C8F381D3D8DB272BDF966AB3D0B1F2BBD70B02ABDA777B5BCBF931FBC6DD3E5BC6AC9D0BDC8AFE03D9E70003C4F29B43C6361063D9DF6B43DE5E9D83C6F7C74BD517FD7BD73D7533D562E9E3C1272AEBD60DCEABD2C5A97BD5EC51B3D68F2BE3D7B3CA83CA3BA49BD7A95D83D2D3D9B3D937409BC97EE96BD177D543D0DCC7EBCDE7709BCDD8DE2BD875B1F3D35A2D1BBFD89DB3DAA6F1EBC8FBD56BC62BAA4BC7E186B3C20C752BD8798A03D5EEDBABD7D008FBD8660853C00921EBD0490CC3DA89BBABDD1A8B9BDE8466CBCCB3EC33D202421BDA30C90BDD98E3EBDBB5C86BDEE05BA3B0A1AB1BD45CCB8BDA49D67BD6B22CC3DDCAE1C3D5DFC9EBD21E4AD3DA155B1BDD37044BD9707D2BD7502A63D137FC03D4059873DEF23183C7345363CC20DE7BDCA09DA3C2408153D0DE8F5BC107119BAC844EBBDD358C3BD18BCD13D4235DABD4F5568BDE44618BD802EAB3B22D1E2BD835BDBBCF2C4FFBB14DADA3D4DFB89BCA4D7B33C9B38A1BCD8C7FBBBAC0A4B3CB6DDA23D6D5433BD900BACBD83CCC8BD6A1C13BD1BA894BD7ECFE63D8EB6763D73BAE8BDE63891BDD6E971BD3F1F85BD20C72B3D2F43C03D49D27E3C108463BD6E263B3CD3B92DBDA04AA4BD20DECD3DEEBF8A3D54A9043DA6B796BD6A101F3D23AF02BD881342398BC8AA3B48C5E53CBE6D79BDE495A43D28C9F7BCDA37543D300BAEBCB1A4E43B5BD27FBDA7EF13BD394777BCF28B883D2E78D03C485850BD597CA4BD993DA73D11028BBD5665573C2B1F9EBD4D68E9BD573BE9BDB5E0553DBBEEAD3D9B32A4BD6F49C2BD2D8A103DE6D8DBBD6055DBBD185F83BDB1B350BD0791E6BC297FC8BC17CE3CBD34C620BD81D56B3DCFE0C23C009E223DE13C993DBF94AABDBE49A9BD4876BFBDF39BE03DBDF8C73D53E4CDBD47A13ABD163EE63D32F799BDE0AFD4BD7AEA3DBD52D169BDDC2C6B3D6204893DA577B23D846BA83D69DD303DEB23A5BDA3AC933C97A5D9BCAA79FEBC2557773DB253D93CBBDA95BB6E5E883D96D39F3DD6B9E73D7CD08D3D4DEB33BDD459723C886B6F3DA4A655BC7BA8E03D0870173C359CC13DE1A4FDBCE2CD2ABDE4E2873D6F65FC3CCC9ED03D9525DC3B6B45A7BD20C1A13D14BCAA3DE4B5F13CBEBB7DBD2392E4BD766DB7BD06097E3D44C2C73D9740FFBCB8CAC33C1515363DF6EACFBC2963BEBC9C2788BD3C8DD1BD4C7FD4BC9BB27DBD09E91A3C112F613D9FA145BDC033BCBDC924A7BCF6B113BD625EA23DBFFDA4BDCC01C5BCBAB0E03DF39C523D0FD0D8BDC08B78BC196725BD9991DC3D539276BD643C48BD4251CEBD738691BCB6F8D3BD55E365BC2A094DBCFE2E8FBD5D9DF7BA3C33C0BD3435BE3C05DE823B442385BCBDADE83D16D8D0BD4BCEDF3D87DF9B3D251D73BCD883A8BBAEDD1DBD12FA91BDAB5E763B4B0B283D3A3B823DC8A948BD60DF573DC741423D74959E3C215ECABCD2EE663B9709C6BB7748E33CE96086BDDDE5E4BDAAFBB43A4D025E3D08ACC43D0B32243DD306C43DDDCF99BD0107FDBAF46D973DEB30863D590127BD3AAC993DBC37553C0828C83D81A0B3BC1875AEBD12524DBD41EF11BD3B8B653DA5A066BCC68BC93C21FA7C3CDF4364BD2779A63D51A09A3CC83E2C3C3C31CCBC76A410BD8CEF463D2C2B8CBD09DD9CBD6BEDC4BDF71EDA3D0007C7BD3102A13B7069A83D24DC4EBD5C71A13C3491BD3D112898BDBC1FB03C2383ACBC0A5FE83D9317D5BC17363B3D96DD023DAF03953D4294B33D0CF890BC681B67BDA9C6933D0703623DD03C42BD5FC8953C678536BD66E280BD25B2D0BAC7CE42BDBE2D86BD5CBDAB3D599AACBDE9F7F13BB54A82BCB7AF983DD243B0BC9FC84FBDEEC1D03D920345BD0F3CFFBC6263723DED8AE0BD30B3C53D8FDCF6BC1357C4BBCD5DBCBD82EFB53C4D7C5C3C1898AA3DFB965FBD1AD0FBBC5218D73B359CC6BD4C20033B0C9DBB3C9900C8BDC0A352BD6E4FD63DBD42F03B234D1FBCC5A86CBCE15463BD87CCE5BC393CAD3DC86EDFBC342FA83CFD40913D4D67E1BD9972A9BDF75561BD3FA10E3C1A84D03DC4F2CEBD396FE7BAC221CEBCAB7D13BC816791BC5BBFB9BD13C5D33C3E922A3D1D3E73BC9CBE30BD1F2ABF3DAFDD4E3DA19292BDB48B93BD4425D03DAF7F3C3DE7D8BEBD06ECC9BDF4770EBD9AAC503CD7AEBFBD3F45B63C726DBABCB0E0CD3B46B9B03D2B22C6BDCCDAF6BC28A83D3CFFE09FBC92DB973D93EF1ABD8F2D633D5A36EABD3EA0073D0A409CBCEDCE94BC6028C9BD0893C4BDB7BED83DD02845BD821851BCF9FE1ABDA802AABD37DFCDBCC3B5D5BD7A7CE5BC9DD8E63DBEC8613DB16F88BD64087DBC2AD904BD04B129BD9C01C23DF0B2D33D7ACFDCBDAFE492BD96982CBDE7E618BCFA43E33D1F6A373D039D2EBD8FC940BD94E64D3D13039F3DAE3BD8BC1AD412BDA8D7FE3BA69022BD85A9C43D113BB23B28D3233D43BCC6BD2CCFE43D05CCB93DA87790BDF82790BDA416B6BD6065AABDB4048C3D0A32DCBD1AD3DEBDCA8721BC9DD4DA3C3F8FE6BD07BCBE3D65179E3DDFC4183D44189A3BA71FA23D0B19A03DA960B4BDA5BD753DCFA0D6BD1749C93BCFD69D3C6A0DCA3CEDFDEA3D542A583C5F1DE9BDD9BF8D3C811ABCBD309CA93D55CD253DC57FA73DF22183BC0543C6BDDFA28C3CCD6B04BD3A0F11BDE3FB733C9F7AE8BCF25CCABAB076D23DD3C1763D86169DBD4D173CBC1ABC813C77BB5C3D901DC7BD4387C2BDDC30BFBD9973C3BCB80A273A28E695BDCD135CBCC821313DF4EAC5BD8A098F3D85ADFA3C15AED23DE886D13D370D273D4737B0BDE5CDC4B9E1F70EBD53A29F3C90A38EBD0729093D8965493D5DFFC2BD5B96A33D30FE263DD11A2ABD13EF8E3A48C327BDBF8E123D0995D9BDB4C898BD0A9360BC32DE553D0E01F43CD50E9ABDD750E2BDCEB6EABDEB2FBEBDCBC2CCBD35527CBDA8D9213D69B17C3D8AC413BDC6C0DD3A76DFBE3DEF41903D1984E6BC8789813CB7E8963DF92F8D3CBC5AA3BD1D27D03D89F781BDA845BBBD9144A03DF7BD84BD9744963DB0D9163CE129C2BD0339873DD177FDBB798E8FBD56A0943C71BCF13C9FA2A43D3CBC693C66EB9BBD67BA633D83D1503DC24AC03CDED93C3D3A87F6BC15F234BCC2091ABD6D39E5BDA463DF3C147522BBFF0C45BD18DDB5BC2587A43C492677BD0AC7C8BD9F544E3A26D5553D6F144EBD0FB91ABC2471C93D5004D0BDF479E23D6FA9C93D58971EBDDCDF093D717710BDCCD9DCBD37C4A93DB41E1EBDFF9368BD8544553D0BE0A93D3C801E3C51A0CCBCAF947EBBD6CC54BD657BDE3DC4BEA6BCAD8BA4BCD0C11CBD92687ABDA69B08BD273E3D3B21FDE73D1F92DBBD46D4A6BC5C57ED3CB494E3BD330B80BD8BA07ABDCA1A063D71D5E03D8A8ED23D75FBCABD713A433DCA22BF3D4F75F2B8A541E0BD856C22BD411DE9BD8A4FBA3C13D4AFBC1FB2E7BD801D9BBDE4A8D8BD0AEE213C956FA13DF73F5D3C139CEA3C1F6CC13D22A7373C8FE5C13D9EE2DEBDD78AE0BDF2E0D8BCB9C8A93D40A8D6BD39E3C3BDC720613CC169B93C2EC5AEBC3680D03DB85C8FBDB221BBBD55C307BD038D77BD22C2E6BCD4B4E03DDAF34FBD57BCE33DD2338B3DB5BF163CCFCAC7BD719C903D007CC7BDB3BF6EBDD1AC933DDD39EBBB0E78C43D1714E63CD371EB3D718C8B3D40AD8EBD7E656DBDB22EB03D37CDB73D45D8C7BD1928863DDB6122BC7749803D28EE79BD2DAC8DBDAE5A18BDC4BC293D5AD429BD1B2FBABD07A391BC41141A3D50CAA13D441B793DFE2AA1BDC36523BDADD7A53D3B14B53C693F56BD717B5ABD4F1E573D3176D6BDAA1098BD90638FBD7E61223DC905973DE47BD5BC2D86D7BD5EEC2F3D249CBA3D282EDDBC495EBC3C561C5F3C83B7EF3CE51207BBE7852EBD73DCD5BD17BACEBD99C0633DFDE2643C7270D23DA59CDD3DC8E3CC3D5BF732BDE60F89BDABE093BDD1F17C3DBA3FB33D46F0A2BDBD8B3C3DDC578EBDD366D2BD3503EE3C5A617BBD17278BBD0CDBBB3DD9CAA33DEBB3013D0413D4BC93AC4B3DC4EEDB3C21CB74BDD213383DC6BB72BDD9B0BEBD49ADD1BD3D825ABD3D81E8BDF326933D9AF0193C6A7B72BD58F8C33D99CB8DBD4DE3B63D1684DF3DF601523DE493BC3C2A21C63D7C9029BD0BC4C6BDC3180E3B804BB0BDCFB3DEBD47486D3DE350ACBD4294A23C965BB2BDE6D3C43C4F9A3A3D7C9A423D990E19BD7D6183BDE2790F3DCA0475BDBB7F45BD61181FBD6A4ECEBDF354873D4986093D81A3D03DDD700E3D559299BD2D4DE1BDEB9104BD1D145E3D99B6A63D0FE2E43D59F2593D99CAD0BDAB6570BD186DC6BD0CE2903DF2F1093D1C0CD63D034994BC67646FBD2D203D3D60B6C1BD9A624B3CC98BE23D4C150E3CF5C229BD75FBA23DEC9FAD3DAC498E3DCBBF8CBCF27B88BDC413A3BCEFDE46BDD11930BCDFCE31BCED92D53CA7A695BDBB780E3D6E41E73D0E27B93DD6C17CBC16EE51BDF657A93BE34F0DBDD2A259BD7C04A6BC5EA88D3D7BB3C93D3D96443DBEA1B5BC36B8D9BCC177A8BD2A6E143D926C9BBDE27D4FBD27CB2FBDB395D23D35CD90BD25EF9EBD1C7266BCB8B229BC877B373D715BC13CF3D2653DDE86BC3DA85E1FBDB74C25BC5A978FBD6CB5A3BCC5CDB8BD4616CBBD2704923DD5DDA9BDF0B63CBC130B863C93862BBD755E15BD0EEEA3BD702EC1BBBA4E7C3D4947303D3D5BA43C1D75DC3CDE322D3D594CB1BDE563A53CFF10813DD6749C3CDFF7B3BD2CA921BDDF13B03C181963BC2853D33DCAEFE33DDF0059BC8A925EBD76C3653CB1E4CC3D0F3B973C62308BBDE25F3DBDC6DFCA3BD581B1BD3449CB3D79F453BC427AD03BC501E23BF01337BDED8EBC3CE0997B3D46FC81BCAB1A4ABC3BF66E3D3043113DBDB5D3BD6C48B1BD093CF3BCEC1C7BBD00F8B4BD8B1DA0BD4727AD3D7F22AABCE563CD3C5C578EBD176AE03DBE9949BC1C512CBD4814DB3D160D18BD1FC486BDB06C153B5F654BBB986D163DD680E5BC824E24BA87CC813D46F677BCB3E7B33DBBF4313D49D6883D446CB9BD2101A13D914CBD3D25552EBDFBE1CDBD6B4CC6BD032CBC3D2939973D49D9E13D51A56DBDCAFCD7BD53970B3C7BD4BF3DF264C1BDBBB71E3DB252963DBB97693D8E61F53C3F24603D2BE46F3DC8A081BD95A1EBBCFB47963D476C53BB858C3BBDCAA9DFBD6042D33DDB64B8BD9AC1B5BD5D02AE3D10DEC03DAC5F94BC9BA29F3D05D8E23D82AACEBDEADD76BDA62BEFBBEA07B5BDED24E9BDCA333DBDFC45DB3D762096BD280BEABD88ABB8BD860B4FBB84B6953D20C7B43C9FE0BC3BF2F3AC3D0F49E2BD2BB4B1BC93ED0D3D0D8B943D4473DABD99FFE9BD712FF5BB07DF96BD16E894BD120EADBDAAF9193CDCA9DABCD1A9AD3D5C75EB3DA82704BD8D81873DAEBD2DBD55A0423D3E7D2E3D9780AA3D670D9D3B0DAB263C330D823D4DF7D4BD7564E33D5A05A1BDCA79E73C40565E3CB071093D6EB92A3D31EE13BDE13EE43CE0F2D73D09F701BD3251993CDA0D7B3D583F173DC869D5BD5ADDD6BDC7B0C3BD5EE180BD0711B5BDE01D11BD1FCB9DBD79C84C3D9BFEC9BD4693363C8C2985BD36AAB63D5392043D7CAE983D7F52C53D788DE13D86BF9ABC3C65493D5FBECEBD1D42B1BC39A35ABC7CCFF5BC6F8D2ABD58ABD1BD166BF43C2C235D3D5505683CB8990C3D1DD486BDBB53CC3CB59331BD1CE4A83D9037E03D8BF4ACBD3249E2BD3B8A2F3D64EF54BC0ADEAD3DCA70643D9677CCBC84089DBDD086BABDA948073D833EE83D5EE0A2BD7529BA3DD1560C3D9EF915BD77EC7B3DEAD7CC3DB1949DBDA00AEABDCFD682BD719C8EBD6CAC67BCF70ADD3C6946CC3D1C8CD1BDC87E3F3D8140203DBEBE113DF4CBA1BDC564D13D23DB943CB563BEBD8F731ABD0E94983B261D5E3CBD25B5BD1684373D45820ABD6A8F903D025389BDC3E61DBDF38BE33D67B3D93DAE4FAABD23ACAA3DC73812BDB305CDBC68B847BD3DCC3DBD1ACC89BCCF3AC03D2BF5E6BC37BD7DBB8F44E3BD3764C53D1F1A213D69C1CBBDAA2D9F3D77AC83BDB28A153D7120DD3CB707A9BD7D88833D153C923CE1ADB03DABFC72BD3D9D583D8B59353C61EC9B3D574097BBA5F58F3DD01748BD9A012F3D4D27EBBD04575DBD71B02EBD872DA3BD250F95BD2FFDB5BD2301B03C5C2BACBD070881BC2367093DAF9F3CBD3FF9283D7169D5BCA3BBDCBC3889E1BC4382E13D7825A33D00D1B43C590BA1BD77287CBD07E9C43B"> : tensor<25x3x5x5xf32>} : () -> tensor<25x3x5x5xf32>
    %1 = "onnx.Constant"() {value = dense<[0.0233115871, 0.108500093, -0.0508011431, -0.0942372456, -0.0305622444, -0.042483665, -0.0267022345, 0.02920462, -0.0689887702, -3.319730e-04, 0.0733448043, -0.107868604, -0.022100918, -0.0510402173, 0.0831331685, -0.0151331201, 0.0026565271, -0.0054282546, -0.0614622571, -0.100244321, -0.0912019535, 0.0994764491, 0.11011605, -0.0039803586, 0.0902173519]> : tensor<25xf32>} : () -> tensor<25xf32>
    //CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int2{{_*[0-9]*}}, %int2{{_*[0-9]*}} :
    //CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int1, %int1{{_*[0-9]*}} :
    //CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int2, %int2{{_*[0-9]*}} :
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Conv_0", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x3x50x50xf32>, tensor<25x3x5x5xf32>, tensor<25xf32>) -> tensor<1x25x25x25xf32>
//CHECK: torch.aten.conv2d %arg0, %{{[0-9]}}, %{{[0-9]}}, [[STRIDE]], [[PAD]], [[DILATION]], %int1{{_*[0-9]*}} : !torch.vtensor<[1,3,50,50],f32>, !torch.vtensor<[25,3,5,5],f32>, !torch.vtensor<[25],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,25,25,25],f32>
    %3 = "onnx.MaxPoolSingleOut"(%2) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_1", pads = [0, 0, 0, 0], strides = [4, 4]} : (tensor<1x25x25x25xf32>) -> tensor<1x25x6x6xf32>
    %4 = "onnx.LeakyRelu"(%3) {alpha = 0.00999999977 : f32, onnx_node_name = "LeakyRelu_2"} : (tensor<1x25x6x6xf32>) -> tensor<1x25x6x6xf32>
    return %4 : tensor<1x25x6x6xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 50 , 50] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 25 , 6 , 6] , \22name\22 : \225\22 }\0A\0A]\00"} : () -> ()
}
