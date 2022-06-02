
python tools/test.py configs\mattors\indexnet\indexnet_dimaug_mobv2_1x16_78k_comp1k.py checkpoints\indexnet_dimaug_mobv2_1x16_78k_comp1k_SAD-50.1_20200626_231857-af359436.pth
# Eval-SAD: 529.6451088702147
# Eval-MSE: 0.2270029882950443
# Eval-GRAD: 204.3321555059524
# Eval-CONN: 580.2440952380952
# Fix
# Eval-SAD: 190.0655320261438
# Eval-MSE: 0.04878358784632651
# Eval-GRAD: 49.91380915178571
# Eval-CONN: 210.36305952380948
python tools/test.py configs\mattors\indexnet\indexnet_mobv2_1x16_78k_comp1k.py checkpoints\indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth
# Eval-SAD: 156.0859434173669
# Eval-MSE: 0.03000798682279871
# Eval-GRAD: 48.59182096354166
# Eval-CONN: 177.7921688988095

python tools/test.py configs\mattors\gca\baseline_dimaug_r34_4x10_200k_comp1k.py checkpoints\baseline_dimaug_r34_4x10_200k_comp1k_SAD-49.95_20200626_231612-535c9a11.pth
# Eval-SAD: 133.5571045751634
# Eval-MSE: 0.02563480355999673
# Eval-GRAD: 38.67630226934524
# Eval-CONN: 145.26864657738093
python tools/test.py configs\mattors\gca\baseline_r34_4x10_200k_comp1k.py checkpoints\baseline_r34_4x10_200k_comp1k_SAD-36.50_20200614_105701-95be1750.pth
# Eval-SAD: 117.74601027077499
# Eval-MSE: 0.019846623887001806
# Eval-GRAD: 32.05425883556548
# Eval-CONN: 126.46961904761905
python tools/test.py configs\mattors\gca\gca_dimaug_r34_4x10_200k_comp1k.py checkpoints\gca_dimaug_r34_4x10_200k_comp1k_SAD-49.42_20200626_231422-8e9cc127.pth
# Eval-SAD: 137.2705803921568
# Eval-MSE: 0.027311727840526795
# Eval-GRAD: 39.63902873883928
# Eval-CONN: 151.5926097470238
python tools/test.py configs\mattors\gca\gca_r34_4x10_200k_comp1k.py checkpoints\gca_r34_4x10_200k_comp1k_SAD-34.77_20200604_213848-4369bea0.pth
# Eval-SAD: 111.98605751633984
# Eval-MSE: 0.019084558225849264
# Eval-GRAD: 31.007079520089285
# Eval-CONN: 115.11171614583331



python tools/test.py .\configs\mattors\dim\dim_stage1_v16_1x1_1000k_comp1k.py .\checkpoints\dim_stage1_v16_1x1_1000k_comp1k_SAD-53.8_20200605_140257-979a420f.pth
# Eval-SAD: 169.33473109243698
# Eval-MSE: 0.03904411720440254
# Eval-GRAD: 52.53457189360118
# Eval-CONN: 184.98380431547622
python tools/test.py .\configs\mattors\dim\dim_stage2_v16_pln_1x1_1000k_comp1k.py .\checkpoints\dim_stage2_v16_pln_1x1_1000k_comp1k_SAD-52.3_20200607_171909-d83c4775.pth
python tools/test.py .\configs\mattors\dim\dim_stage3_v16_pln_1x1_1000k_comp1k.py .\checkpoints\dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth
