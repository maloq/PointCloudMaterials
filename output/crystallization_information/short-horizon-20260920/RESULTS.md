# What information is missing for crystallization within 12 ps?

Frozen encoders; same natural at-risk windows and historical source splits. One seed. All predictors have matched input slots and hidden width within a readout family. No encoder training or test-based checkpoint selection. Positive NLL gain means an added block helps.

| Encoder | Input | Readout | Event NLL | 3 ps AP | 6 ps AP | 9 ps AP | 12 ps AP | Paired NLL gain [95% source interval] |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline | all | linear | 0.20646 | 0.3045 | 0.3454 | 0.3517 | 0.3471 | — |
| baseline | all | mlp | 0.16745 | 0.3808 | 0.4604 | 0.4770 | 0.4914 | — |
| baseline | conditions | linear | 0.24446 | 0.0130 | 0.0272 | 0.0421 | 0.0560 | — |
| baseline | conditions | mlp | 0.24237 | 0.0237 | 0.0503 | 0.0749 | 0.0969 | — |
| baseline | current | linear | 0.22436 | 0.0908 | 0.1127 | 0.1334 | 0.1497 | — |
| baseline | current | mlp | 0.20164 | 0.1976 | 0.2364 | 0.2601 | 0.2709 | — |
| baseline | geometry | linear | 0.22446 | 0.0878 | 0.1128 | 0.1335 | 0.1493 | — |
| baseline | geometry | mlp | 0.20035 | 0.2017 | 0.2460 | 0.2706 | 0.2818 | — |
| mace-epi-direct | z | linear | 0.23348 | 0.0605 | 0.0819 | 0.0934 | 0.1040 | — |
| mace-epi-direct | z | mlp | 0.21705 | 0.1405 | 0.1720 | 0.1889 | 0.1939 | — |
| mace-epi-direct | z | strong | 0.21465 | 0.1481 | 0.1842 | 0.2083 | 0.2113 | — |
| mace-epi-direct | z+all | linear | 0.20536 | 0.3184 | 0.3560 | 0.3579 | 0.3529 | 0.02812 [0.02243, 0.03379] |
| mace-epi-direct | z+all | mlp | 0.16886 | 0.3921 | 0.4636 | 0.4718 | 0.4839 | 0.04819 [0.04078, 0.05667] |
| mace-epi-direct | z+all | strong | 0.16728 | 0.3874 | 0.4701 | 0.4843 | 0.5021 | 0.04737 [0.03980, 0.05615] |
| mace-epi-direct | z+angular | linear | 0.23030 | 0.0857 | 0.1040 | 0.1175 | 0.1281 | 0.00318 [0.00253, 0.00388] |
| mace-epi-direct | z+angular | mlp | 0.21202 | 0.1781 | 0.2026 | 0.2222 | 0.2246 | 0.00503 [0.00335, 0.00694] |
| mace-epi-direct | z+current | linear | 0.22228 | 0.1302 | 0.1489 | 0.1640 | 0.1782 | 0.01120 [0.00905, 0.01341] |
| mace-epi-direct | z+current | mlp | 0.20343 | 0.1859 | 0.2225 | 0.2473 | 0.2599 | 0.01362 [0.00978, 0.01732] |
| mace-epi-direct | z+density | linear | 0.23312 | 0.0627 | 0.0830 | 0.0954 | 0.1063 | 0.00036 [0.00009, 0.00062] |
| mace-epi-direct | z+density | mlp | 0.21725 | 0.1405 | 0.1698 | 0.1877 | 0.1919 | -0.00020 [-0.00073, 0.00026] |
| mace-epi-direct | z+geometry | linear | 0.22239 | 0.1272 | 0.1484 | 0.1632 | 0.1775 | 0.01109 [0.00903, 0.01318] |
| mace-epi-direct | z+geometry | mlp | 0.20190 | 0.1926 | 0.2318 | 0.2569 | 0.2700 | 0.01515 [0.01151, 0.01884] |
| mace-epi-direct | z+history | linear | 0.21365 | 0.3059 | 0.3469 | 0.3368 | 0.3184 | 0.01982 [0.01507, 0.02487] |
| mace-epi-direct | z+history | mlp | 0.19616 | 0.3631 | 0.3886 | 0.3898 | 0.3796 | 0.02089 [0.01729, 0.02476] |
| mace-epi-direct | z+motion | linear | 0.23317 | 0.0652 | 0.0829 | 0.0937 | 0.1043 | 0.00031 [-0.00002, 0.00063] |
| mace-epi-direct | z+motion | mlp | 0.21879 | 0.1272 | 0.1576 | 0.1752 | 0.1821 | -0.00174 [-0.00272, -0.00076] |
| mace-epi-direct | z+order | linear | 0.22818 | 0.1084 | 0.1214 | 0.1319 | 0.1428 | 0.00529 [0.00416, 0.00642] |
| mace-epi-direct | z+order | mlp | 0.20435 | 0.1835 | 0.2267 | 0.2539 | 0.2601 | 0.01270 [0.00819, 0.01683] |
| mace-epi-direct | z+outer_geometry | linear | 0.22939 | 0.0603 | 0.0810 | 0.0966 | 0.1142 | 0.00408 [0.00313, 0.00509] |
| mace-epi-direct | z+outer_geometry | mlp | 0.16584 | 0.3091 | 0.4327 | 0.4608 | 0.4843 | 0.05121 [0.04195, 0.06142] |
| mace-epi-direct | z+outer_motion | linear | 0.23334 | 0.0624 | 0.0813 | 0.0934 | 0.1038 | 0.00014 [-0.00008, 0.00036] |
| mace-epi-direct | z+outer_motion | mlp | 0.21708 | 0.1429 | 0.1721 | 0.1872 | 0.1920 | -0.00003 [-0.00028, 0.00022] |
| mace-epi-direct | z+radial_pair | linear | 0.22670 | 0.0873 | 0.1117 | 0.1251 | 0.1417 | 0.00678 [0.00534, 0.00843] |
| mace-epi-direct | z+radial_pair | mlp | 0.20557 | 0.1746 | 0.2113 | 0.2331 | 0.2484 | 0.01148 [0.00892, 0.01416] |
| mace-epi-direct | z+shuffled | linear | 0.23472 | 0.0587 | 0.0799 | 0.0893 | 0.0996 | -0.00124 [-0.00178, -0.00070] |
| mace-epi-direct | z+shuffled | mlp | 0.22423 | 0.1043 | 0.1345 | 0.1448 | 0.1533 | -0.00718 [-0.00979, -0.00463] |
| mace-sigreg-direct | z | linear | 0.23419 | 0.0513 | 0.0740 | 0.0863 | 0.0966 | — |
| mace-sigreg-direct | z | mlp | 0.21788 | 0.1427 | 0.1678 | 0.1815 | 0.1875 | — |
| mace-sigreg-direct | z | strong | 0.21515 | 0.1514 | 0.1842 | 0.2001 | 0.2048 | — |
| mace-sigreg-direct | z+all | linear | 0.20543 | 0.3165 | 0.3544 | 0.3554 | 0.3507 | 0.02876 [0.02271, 0.03468] |
| mace-sigreg-direct | z+all | mlp | 0.16849 | 0.3993 | 0.4684 | 0.4771 | 0.4898 | 0.04939 [0.04210, 0.05773] |
| mace-sigreg-direct | z+all | strong | 0.16609 | 0.3846 | 0.4735 | 0.4873 | 0.5077 | 0.04907 [0.04166, 0.05764] |
| mace-sigreg-direct | z+angular | linear | 0.23114 | 0.0672 | 0.0901 | 0.1055 | 0.1169 | 0.00305 [0.00241, 0.00372] |
| mace-sigreg-direct | z+angular | mlp | 0.21228 | 0.1880 | 0.2053 | 0.2208 | 0.2239 | 0.00559 [0.00381, 0.00760] |
| mace-sigreg-direct | z+current | linear | 0.22284 | 0.1198 | 0.1417 | 0.1570 | 0.1719 | 0.01136 [0.00901, 0.01365] |
| mace-sigreg-direct | z+current | mlp | 0.20366 | 0.1886 | 0.2195 | 0.2429 | 0.2558 | 0.01422 [0.01051, 0.01778] |
| mace-sigreg-direct | z+density | linear | 0.23382 | 0.0534 | 0.0751 | 0.0883 | 0.0994 | 0.00037 [0.00007, 0.00067] |
| mace-sigreg-direct | z+density | mlp | 0.21832 | 0.1375 | 0.1620 | 0.1784 | 0.1838 | -0.00044 [-0.00098, 0.00006] |
| mace-sigreg-direct | z+geometry | linear | 0.22294 | 0.1163 | 0.1406 | 0.1566 | 0.1712 | 0.01125 [0.00897, 0.01354] |
| mace-sigreg-direct | z+geometry | mlp | 0.20223 | 0.1933 | 0.2279 | 0.2519 | 0.2635 | 0.01565 [0.01193, 0.01934] |
| mace-sigreg-direct | z+history | linear | 0.21394 | 0.3034 | 0.3456 | 0.3346 | 0.3165 | 0.02025 [0.01514, 0.02554] |
| mace-sigreg-direct | z+history | mlp | 0.19667 | 0.3578 | 0.3876 | 0.3883 | 0.3783 | 0.02120 [0.01784, 0.02457] |
| mace-sigreg-direct | z+motion | linear | 0.23388 | 0.0547 | 0.0749 | 0.0869 | 0.0968 | 0.00031 [-0.00001, 0.00061] |
| mace-sigreg-direct | z+motion | mlp | 0.21974 | 0.1256 | 0.1536 | 0.1676 | 0.1758 | -0.00187 [-0.00286, -0.00088] |
| mace-sigreg-direct | z+order | linear | 0.22896 | 0.0888 | 0.1068 | 0.1206 | 0.1321 | 0.00523 [0.00409, 0.00636] |
| mace-sigreg-direct | z+order | mlp | 0.20396 | 0.1984 | 0.2336 | 0.2564 | 0.2617 | 0.01391 [0.00930, 0.01809] |
| mace-sigreg-direct | z+outer_geometry | linear | 0.23012 | 0.0513 | 0.0737 | 0.0901 | 0.1079 | 0.00407 [0.00309, 0.00514] |
| mace-sigreg-direct | z+outer_geometry | mlp | 0.16574 | 0.2970 | 0.4272 | 0.4581 | 0.4821 | 0.05214 [0.04273, 0.06229] |
| mace-sigreg-direct | z+outer_motion | linear | 0.23406 | 0.0522 | 0.0741 | 0.0866 | 0.0967 | 0.00013 [-0.00009, 0.00035] |
| mace-sigreg-direct | z+outer_motion | mlp | 0.21796 | 0.1401 | 0.1668 | 0.1805 | 0.1869 | -0.00008 [-0.00032, 0.00017] |
| mace-sigreg-direct | z+radial_pair | linear | 0.22738 | 0.0745 | 0.1004 | 0.1150 | 0.1326 | 0.00681 [0.00526, 0.00847] |
| mace-sigreg-direct | z+radial_pair | mlp | 0.20659 | 0.1662 | 0.1990 | 0.2194 | 0.2346 | 0.01129 [0.00870, 0.01389] |
| mace-sigreg-direct | z+shuffled | linear | 0.23547 | 0.0496 | 0.0721 | 0.0825 | 0.0925 | -0.00128 [-0.00182, -0.00073] |
| mace-sigreg-direct | z+shuffled | mlp | 0.22511 | 0.0923 | 0.1241 | 0.1339 | 0.1442 | -0.00724 [-0.00977, -0.00462] |
| mace-vicreg-direct | z | linear | 0.22901 | 0.1076 | 0.1291 | 0.1430 | 0.1556 | — |
| mace-vicreg-direct | z | mlp | 0.21620 | 0.1414 | 0.1713 | 0.1921 | 0.1972 | — |
| mace-vicreg-direct | z | strong | 0.21620 | 0.1403 | 0.1718 | 0.1922 | 0.1978 | — |
| mace-vicreg-direct | z+all | linear | 0.20469 | 0.3289 | 0.3658 | 0.3675 | 0.3609 | 0.02432 [0.01953, 0.02947] |
| mace-vicreg-direct | z+all | mlp | 0.16864 | 0.3830 | 0.4576 | 0.4720 | 0.4854 | 0.04756 [0.04034, 0.05555] |
| mace-vicreg-direct | z+all | strong | 0.16708 | 0.3860 | 0.4727 | 0.4892 | 0.5046 | 0.04912 [0.04139, 0.05797] |
| mace-vicreg-direct | z+angular | linear | 0.22610 | 0.1348 | 0.1525 | 0.1670 | 0.1782 | 0.00291 [0.00219, 0.00370] |
| mace-vicreg-direct | z+angular | mlp | 0.21115 | 0.1787 | 0.2038 | 0.2270 | 0.2301 | 0.00506 [0.00334, 0.00697] |
| mace-vicreg-direct | z+current | linear | 0.21959 | 0.1513 | 0.1713 | 0.1886 | 0.2043 | 0.00942 [0.00760, 0.01124] |
| mace-vicreg-direct | z+current | mlp | 0.20223 | 0.1959 | 0.2295 | 0.2559 | 0.2665 | 0.01397 [0.01095, 0.01713] |
| mace-vicreg-direct | z+density | linear | 0.22882 | 0.1086 | 0.1296 | 0.1442 | 0.1569 | 0.00019 [0.00002, 0.00036] |
| mace-vicreg-direct | z+density | mlp | 0.21663 | 0.1396 | 0.1707 | 0.1885 | 0.1939 | -0.00043 [-0.00096, 0.00004] |
| mace-vicreg-direct | z+geometry | linear | 0.21975 | 0.1473 | 0.1693 | 0.1871 | 0.2025 | 0.00926 [0.00746, 0.01106] |
| mace-vicreg-direct | z+geometry | mlp | 0.20115 | 0.1996 | 0.2364 | 0.2634 | 0.2743 | 0.01505 [0.01202, 0.01814] |
| mace-vicreg-direct | z+history | linear | 0.21181 | 0.3251 | 0.3640 | 0.3563 | 0.3381 | 0.01720 [0.01310, 0.02173] |
| mace-vicreg-direct | z+history | mlp | 0.19617 | 0.3618 | 0.3918 | 0.3915 | 0.3822 | 0.02003 [0.01672, 0.02378] |
| mace-vicreg-direct | z+motion | linear | 0.22862 | 0.1147 | 0.1321 | 0.1452 | 0.1572 | 0.00039 [0.00003, 0.00075] |
| mace-vicreg-direct | z+motion | mlp | 0.21748 | 0.1298 | 0.1632 | 0.1799 | 0.1878 | -0.00128 [-0.00213, -0.00048] |
| mace-vicreg-direct | z+order | linear | 0.22419 | 0.1427 | 0.1635 | 0.1787 | 0.1907 | 0.00482 [0.00377, 0.00581] |
| mace-vicreg-direct | z+order | mlp | 0.20365 | 0.1838 | 0.2271 | 0.2511 | 0.2588 | 0.01255 [0.00906, 0.01588] |
| mace-vicreg-direct | z+outer_geometry | linear | 0.22543 | 0.1061 | 0.1231 | 0.1396 | 0.1577 | 0.00358 [0.00266, 0.00455] |
| mace-vicreg-direct | z+outer_geometry | mlp | 0.16618 | 0.3079 | 0.4117 | 0.4479 | 0.4713 | 0.05002 [0.04066, 0.06048] |
| mace-vicreg-direct | z+outer_motion | linear | 0.22884 | 0.1115 | 0.1303 | 0.1436 | 0.1558 | 0.00017 [-0.00006, 0.00040] |
| mace-vicreg-direct | z+outer_motion | mlp | 0.21610 | 0.1411 | 0.1720 | 0.1927 | 0.1978 | 0.00010 [-0.00017, 0.00038] |
| mace-vicreg-direct | z+radial_pair | linear | 0.22304 | 0.1197 | 0.1451 | 0.1618 | 0.1791 | 0.00596 [0.00469, 0.00736] |
| mace-vicreg-direct | z+radial_pair | mlp | 0.20533 | 0.1834 | 0.2189 | 0.2395 | 0.2509 | 0.01087 [0.00881, 0.01319] |
| mace-vicreg-direct | z+shuffled | linear | 0.23013 | 0.1058 | 0.1271 | 0.1384 | 0.1511 | -0.00112 [-0.00169, -0.00056] |
| mace-vicreg-direct | z+shuffled | mlp | 0.22260 | 0.1087 | 0.1428 | 0.1562 | 0.1665 | -0.00640 [-0.00867, -0.00419] |
| old-vicreg-gatr | z | linear | 0.23912 | 0.0335 | 0.0500 | 0.0620 | 0.0723 | — |
| old-vicreg-gatr | z | mlp | 0.23154 | 0.0781 | 0.0879 | 0.0995 | 0.1135 | — |
| old-vicreg-gatr | z | strong | 0.23173 | 0.0732 | 0.0812 | 0.0904 | 0.1032 | — |
| old-vicreg-gatr | z+all | linear | 0.20641 | 0.3090 | 0.3466 | 0.3511 | 0.3458 | 0.03271 [0.02685, 0.03849] |
| old-vicreg-gatr | z+all | mlp | 0.16761 | 0.3882 | 0.4687 | 0.4799 | 0.4959 | 0.06394 [0.05369, 0.07449] |
| old-vicreg-gatr | z+all | strong | 0.16624 | 0.3788 | 0.4777 | 0.4944 | 0.5120 | 0.06549 [0.05511, 0.07647] |
| old-vicreg-gatr | z+angular | linear | 0.23415 | 0.0459 | 0.0683 | 0.0841 | 0.0946 | 0.00497 [0.00401, 0.00599] |
| old-vicreg-gatr | z+angular | mlp | 0.21866 | 0.1510 | 0.1673 | 0.1861 | 0.1927 | 0.01288 [0.00846, 0.01732] |
| old-vicreg-gatr | z+current | linear | 0.22469 | 0.0956 | 0.1139 | 0.1323 | 0.1471 | 0.01443 [0.01174, 0.01742] |
| old-vicreg-gatr | z+current | mlp | 0.20192 | 0.1935 | 0.2377 | 0.2569 | 0.2689 | 0.02962 [0.02365, 0.03545] |
| old-vicreg-gatr | z+density | linear | 0.23833 | 0.0346 | 0.0505 | 0.0630 | 0.0744 | 0.00079 [0.00038, 0.00121] |
| old-vicreg-gatr | z+density | mlp | 0.23301 | 0.0733 | 0.0882 | 0.1008 | 0.1139 | -0.00146 [-0.00371, 0.00087] |
| old-vicreg-gatr | z+geometry | linear | 0.22471 | 0.0929 | 0.1138 | 0.1316 | 0.1462 | 0.01441 [0.01162, 0.01740] |
| old-vicreg-gatr | z+geometry | mlp | 0.20057 | 0.1999 | 0.2488 | 0.2721 | 0.2816 | 0.03097 [0.02470, 0.03732] |
| old-vicreg-gatr | z+history | linear | 0.21564 | 0.2952 | 0.3344 | 0.3271 | 0.3098 | 0.02348 [0.01805, 0.02863] |
| old-vicreg-gatr | z+history | mlp | 0.19972 | 0.3379 | 0.3658 | 0.3702 | 0.3635 | 0.03182 [0.02585, 0.03770] |
| old-vicreg-gatr | z+motion | linear | 0.23890 | 0.0338 | 0.0500 | 0.0623 | 0.0730 | 0.00022 [-0.00007, 0.00050] |
| old-vicreg-gatr | z+motion | mlp | 0.23415 | 0.0595 | 0.0769 | 0.0909 | 0.1046 | -0.00261 [-0.00399, -0.00123] |
| old-vicreg-gatr | z+order | linear | 0.23178 | 0.0548 | 0.0836 | 0.1017 | 0.1135 | 0.00734 [0.00508, 0.00931] |
| old-vicreg-gatr | z+order | mlp | 0.20687 | 0.1787 | 0.2100 | 0.2250 | 0.2392 | 0.02467 [0.01938, 0.02966] |
| old-vicreg-gatr | z+outer_geometry | linear | 0.23386 | 0.0371 | 0.0612 | 0.0840 | 0.1020 | 0.00526 [0.00403, 0.00654] |
| old-vicreg-gatr | z+outer_geometry | mlp | 0.17032 | 0.2876 | 0.3855 | 0.4294 | 0.4634 | 0.06123 [0.04926, 0.07441] |
| old-vicreg-gatr | z+outer_motion | linear | 0.23903 | 0.0337 | 0.0497 | 0.0621 | 0.0725 | 0.00009 [-0.00012, 0.00030] |
| old-vicreg-gatr | z+outer_motion | mlp | 0.23135 | 0.0786 | 0.0881 | 0.0998 | 0.1142 | 0.00019 [-0.00028, 0.00067] |
| old-vicreg-gatr | z+radial_pair | linear | 0.23046 | 0.0510 | 0.0748 | 0.0905 | 0.1052 | 0.00866 [0.00679, 0.01055] |
| old-vicreg-gatr | z+radial_pair | mlp | 0.20617 | 0.1761 | 0.2148 | 0.2286 | 0.2430 | 0.02537 [0.01940, 0.03126] |
| old-vicreg-gatr | z+shuffled | linear | 0.24047 | 0.0299 | 0.0472 | 0.0573 | 0.0683 | -0.00135 [-0.00182, -0.00080] |
| old-vicreg-gatr | z+shuffled | mlp | 0.23873 | 0.0310 | 0.0494 | 0.0602 | 0.0737 | -0.00718 [-0.00987, -0.00462] |
| old-vicreg-mace | z | linear | 0.23832 | 0.0408 | 0.0650 | 0.0812 | 0.0928 | — |
| old-vicreg-mace | z | mlp | 0.21366 | 0.1797 | 0.2113 | 0.2293 | 0.2297 | — |
| old-vicreg-mace | z | strong | 0.20971 | 0.2064 | 0.2439 | 0.2650 | 0.2625 | — |
| old-vicreg-mace | z+all | linear | 0.20596 | 0.3093 | 0.3497 | 0.3534 | 0.3491 | 0.03236 [0.02679, 0.03801] |
| old-vicreg-mace | z+all | mlp | 0.16874 | 0.3827 | 0.4626 | 0.4739 | 0.4858 | 0.04492 [0.03769, 0.05328] |
| old-vicreg-mace | z+all | strong | 0.16568 | 0.3754 | 0.4745 | 0.4923 | 0.5140 | 0.04404 [0.03637, 0.05298] |
| old-vicreg-mace | z+angular | linear | 0.23360 | 0.0494 | 0.0752 | 0.0917 | 0.1037 | 0.00472 [0.00359, 0.00586] |
| old-vicreg-mace | z+angular | mlp | 0.21102 | 0.2057 | 0.2192 | 0.2322 | 0.2363 | 0.00264 [0.00102, 0.00432] |
| old-vicreg-mace | z+current | linear | 0.22385 | 0.1059 | 0.1234 | 0.1420 | 0.1573 | 0.01447 [0.01168, 0.01735] |
| old-vicreg-mace | z+current | mlp | 0.20242 | 0.1895 | 0.2305 | 0.2529 | 0.2655 | 0.01125 [0.00835, 0.01405] |
| old-vicreg-mace | z+density | linear | 0.23753 | 0.0406 | 0.0617 | 0.0765 | 0.0897 | 0.00080 [0.00044, 0.00115] |
| old-vicreg-mace | z+density | mlp | 0.21402 | 0.1707 | 0.2067 | 0.2238 | 0.2248 | -0.00036 [-0.00084, 0.00011] |
| old-vicreg-mace | z+geometry | linear | 0.22395 | 0.1036 | 0.1242 | 0.1420 | 0.1571 | 0.01437 [0.01163, 0.01719] |
| old-vicreg-mace | z+geometry | mlp | 0.20066 | 0.2063 | 0.2483 | 0.2680 | 0.2789 | 0.01301 [0.00999, 0.01592] |
| old-vicreg-mace | z+history | linear | 0.21505 | 0.2948 | 0.3374 | 0.3305 | 0.3137 | 0.02327 [0.01812, 0.02870] |
| old-vicreg-mace | z+history | mlp | 0.19811 | 0.3352 | 0.3690 | 0.3742 | 0.3701 | 0.01555 [0.01249, 0.01887] |
| old-vicreg-mace | z+motion | linear | 0.23806 | 0.0398 | 0.0597 | 0.0734 | 0.0841 | 0.00026 [-0.00011, 0.00061] |
| old-vicreg-mace | z+motion | mlp | 0.21675 | 0.1500 | 0.1812 | 0.1933 | 0.1980 | -0.00309 [-0.00459, -0.00143] |
| old-vicreg-mace | z+order | linear | 0.23081 | 0.0648 | 0.0881 | 0.1048 | 0.1178 | 0.00751 [0.00537, 0.00943] |
| old-vicreg-mace | z+order | mlp | 0.20191 | 0.2154 | 0.2605 | 0.2810 | 0.2873 | 0.01176 [0.00823, 0.01494] |
| old-vicreg-mace | z+outer_geometry | linear | 0.23324 | 0.0394 | 0.0682 | 0.0928 | 0.1128 | 0.00508 [0.00384, 0.00637] |
| old-vicreg-mace | z+outer_geometry | mlp | 0.16599 | 0.3004 | 0.4179 | 0.4542 | 0.4783 | 0.04767 [0.03796, 0.05817] |
| old-vicreg-mace | z+outer_motion | linear | 0.23821 | 0.0406 | 0.0620 | 0.0777 | 0.0881 | 0.00012 [-0.00012, 0.00036] |
| old-vicreg-mace | z+outer_motion | mlp | 0.21345 | 0.1834 | 0.2169 | 0.2296 | 0.2294 | 0.00022 [-0.00010, 0.00052] |
| old-vicreg-mace | z+radial_pair | linear | 0.22935 | 0.0546 | 0.0795 | 0.0960 | 0.1138 | 0.00898 [0.00693, 0.01122] |
| old-vicreg-mace | z+radial_pair | mlp | 0.20561 | 0.1686 | 0.2113 | 0.2282 | 0.2421 | 0.00805 [0.00570, 0.01036] |
| old-vicreg-mace | z+shuffled | linear | 0.23977 | 0.0332 | 0.0559 | 0.0666 | 0.0762 | -0.00145 [-0.00200, -0.00088] |
| old-vicreg-mace | z+shuffled | mlp | 0.22713 | 0.0753 | 0.1033 | 0.1177 | 0.1303 | -0.01347 [-0.01760, -0.00922] |

## Physical information retained in the export

| Encoder | Decoder | Feature block | Test standardized MSE | Test R² |
|---|---|---|---:|---:|
| baseline | strong | radial_pair | 0.7246 | 0.0201 |
| baseline | strong | angular | 0.8576 | 0.0080 |
| baseline | strong | order | 0.9872 | 0.0144 |
| baseline | strong | density | 0.9887 | 0.0131 |
| baseline | strong | motion | 0.9823 | 0.0264 |
| baseline | strong | outer_geometry | 0.9056 | 0.1001 |
| baseline | strong | outer_motion | 0.9737 | 0.0295 |
| mace-epi-direct | linear | radial_pair | 0.3331 | 0.5495 |
| mace-epi-direct | linear | angular | 0.6574 | 0.2395 |
| mace-epi-direct | linear | order | 0.7280 | 0.2732 |
| mace-epi-direct | linear | density | 0.3826 | 0.6181 |
| mace-epi-direct | linear | motion | 0.9852 | 0.0235 |
| mace-epi-direct | linear | outer_geometry | 0.6957 | 0.3087 |
| mace-epi-direct | linear | outer_motion | 0.9771 | 0.0261 |
| mace-epi-direct | strong | radial_pair | 0.2748 | 0.6284 |
| mace-epi-direct | strong | angular | 0.5985 | 0.3077 |
| mace-epi-direct | strong | order | 0.6616 | 0.3395 |
| mace-epi-direct | strong | density | 0.2090 | 0.7914 |
| mace-epi-direct | strong | motion | 0.9842 | 0.0245 |
| mace-epi-direct | strong | outer_geometry | 0.6210 | 0.3829 |
| mace-epi-direct | strong | outer_motion | 0.9753 | 0.0279 |
| mace-sigreg-direct | linear | radial_pair | 0.3452 | 0.5331 |
| mace-sigreg-direct | linear | angular | 0.6674 | 0.2279 |
| mace-sigreg-direct | linear | order | 0.7306 | 0.2706 |
| mace-sigreg-direct | linear | density | 0.4242 | 0.5765 |
| mace-sigreg-direct | linear | motion | 0.9851 | 0.0237 |
| mace-sigreg-direct | linear | outer_geometry | 0.7003 | 0.3041 |
| mace-sigreg-direct | linear | outer_motion | 0.9770 | 0.0262 |
| mace-sigreg-direct | strong | radial_pair | 0.2791 | 0.6225 |
| mace-sigreg-direct | strong | angular | 0.6002 | 0.3057 |
| mace-sigreg-direct | strong | order | 0.6647 | 0.3364 |
| mace-sigreg-direct | strong | density | 0.2245 | 0.7760 |
| mace-sigreg-direct | strong | motion | 0.9839 | 0.0248 |
| mace-sigreg-direct | strong | outer_geometry | 0.6229 | 0.3810 |
| mace-sigreg-direct | strong | outer_motion | 0.9752 | 0.0280 |
| mace-vicreg-direct | linear | radial_pair | 0.3529 | 0.5227 |
| mace-vicreg-direct | linear | angular | 0.6580 | 0.2388 |
| mace-vicreg-direct | linear | order | 0.7272 | 0.2740 |
| mace-vicreg-direct | linear | density | 0.4121 | 0.5887 |
| mace-vicreg-direct | linear | motion | 0.9855 | 0.0232 |
| mace-vicreg-direct | linear | outer_geometry | 0.6962 | 0.3082 |
| mace-vicreg-direct | linear | outer_motion | 0.9774 | 0.0258 |
| mace-vicreg-direct | strong | radial_pair | 0.2935 | 0.6031 |
| mace-vicreg-direct | strong | angular | 0.6104 | 0.2939 |
| mace-vicreg-direct | strong | order | 0.6802 | 0.3209 |
| mace-vicreg-direct | strong | density | 0.2806 | 0.7199 |
| mace-vicreg-direct | strong | motion | 0.9841 | 0.0247 |
| mace-vicreg-direct | strong | outer_geometry | 0.6297 | 0.3742 |
| mace-vicreg-direct | strong | outer_motion | 0.9752 | 0.0279 |
| old-vicreg-gatr | linear | radial_pair | 0.2579 | 0.6512 |
| old-vicreg-gatr | linear | angular | 0.6862 | 0.2063 |
| old-vicreg-gatr | linear | order | 0.8191 | 0.1823 |
| old-vicreg-gatr | linear | density | 0.3353 | 0.6653 |
| old-vicreg-gatr | linear | motion | 0.9852 | 0.0236 |
| old-vicreg-gatr | linear | outer_geometry | 0.7701 | 0.2347 |
| old-vicreg-gatr | linear | outer_motion | 0.9772 | 0.0260 |
| old-vicreg-gatr | strong | radial_pair | 0.2226 | 0.6989 |
| old-vicreg-gatr | strong | angular | 0.6289 | 0.2725 |
| old-vicreg-gatr | strong | order | 0.7487 | 0.2525 |
| old-vicreg-gatr | strong | density | 0.2265 | 0.7739 |
| old-vicreg-gatr | strong | motion | 0.9837 | 0.0250 |
| old-vicreg-gatr | strong | outer_geometry | 0.6746 | 0.3297 |
| old-vicreg-gatr | strong | outer_motion | 0.9751 | 0.0281 |
| old-vicreg-mace | linear | radial_pair | 0.3229 | 0.5633 |
| old-vicreg-mace | linear | angular | 0.6668 | 0.2286 |
| old-vicreg-mace | linear | order | 0.7206 | 0.2806 |
| old-vicreg-mace | linear | density | 0.3242 | 0.6764 |
| old-vicreg-mace | linear | motion | 0.9853 | 0.0234 |
| old-vicreg-mace | linear | outer_geometry | 0.6833 | 0.3210 |
| old-vicreg-mace | linear | outer_motion | 0.9773 | 0.0258 |
| old-vicreg-mace | strong | radial_pair | 0.2182 | 0.7049 |
| old-vicreg-mace | strong | angular | 0.5818 | 0.3269 |
| old-vicreg-mace | strong | order | 0.6496 | 0.3514 |
| old-vicreg-mace | strong | density | 0.1034 | 0.8968 |
| old-vicreg-mace | strong | motion | 0.9837 | 0.0251 |
| old-vicreg-mace | strong | outer_geometry | 0.6131 | 0.3907 |
| old-vicreg-mace | strong | outer_motion | 0.9747 | 0.0284 |

## Interpretation

- An add-back gain measures information inaccessible to the tested frozen readout; it is not proof of information-theoretic absence.
- Compare linear, MLP and stronger embedding-only readouts: recovery with a stronger head suggests accessibility/optimization rather than missing input.
- All additive comparisons use the same input dimensionality, initial weights, update count and source-balanced batches; absent groups are zeroed. The shuffled full-block negative control preserves role and temperature, but is deliberately unpaired with the target observation.
- Weak physical decoding plus a positive corresponding add-back gain identifies a candidate representation deficiency. Good decoding plus a gain points toward readout organization/optimization.
- Velocity, observed history and 7–25 Å outer shells are additional inputs beyond a position-only local snapshot. Their gains identify missing observation context, not necessarily compression failure.
- Current packet support and raw bond-order neighborhoods differ from the encoder crop; even geometry gains can reflect observation support.
- Confidence intervals use paired temperature-stratified whole-source resampling. They exclude training-seed uncertainty, are exploratory and are not corrected for the many comparisons. AP gains are point estimates.
- Timing metrics include missed event windows; MAE alone excludes misses. The 3 ps origin spacing limits timing resolution.
- No future observations, PTM labels or sustained-event confirmation enter predictors. Those frames define labels only. No new simulations or TDA targets were generated.

Completed readouts: 159. Missing rows remain pending.
