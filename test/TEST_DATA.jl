# -*- encoding: utf-8 -*-
#
# This file is part of GaPSE
# Copyright (C) 2022 Matteo Foglieni
#
# GaPSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GaPSE is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GaPSE. If not, see <http://www.gnu.org/licenses/>.
#


##########################################################################################92



const ZS = [0.0, 0.006559072275194, 0.01364313038009, 0.02077687583214, 0.02796065459573,
     0.03519481503942, 0.04247970795382, 0.04981568656961, 0.05720310657579,
     0.06464232613797, 0.07213370591695, 0.07967760908732, 0.08727440135637,
     0.09492445098303, 0.102628128797, 0.1103858082182, 0.1181978652761,
     0.1260646786292, 0.1339866295853, 0.1419641021209, 0.1499974829017,
     0.1580871613026, 0.1662335294283, 0.1744369821336, 0.1826979170446,
     0.1910167345789, 0.1993938379675, 0.2078296332749, 0.2163245294215,
     0.2248789382045, 0.2334932743195, 0.2421679553825, 0.2509034019519,
     0.2597000375503, 0.2685582886868, 0.2774785848796, 0.2864613586783,
     0.2955070456865, 0.3046160845849, 0.3137889171538, 0.3230259882968,
     0.3323277460632, 0.3416946416721, 0.3511271295355, 0.3606256672822,
     0.3701907157812, 0.3798227391661, 0.3895222048585, 0.3992895835929,
     0.4091253494405, 0.4190299798338, 0.4290039555914, 0.4390477609422,
     0.4491618835508]

const CONF_TIME = [9726.894892455999, 9707.260856601, 9686.120254803998, 9664.900383866,
     9643.601962505, 9622.225727519999, 9600.77243363, 9579.242853425, 9557.637777233,
     9535.958012965999, 9514.204385993999, 9492.377738984, 9470.478931732,
     9448.508840974, 9426.468360169, 9404.358399338, 9382.17988477, 9359.933758846999,
     9337.62097975, 9315.242521228, 9292.799372296999, 9270.292536953, 9247.723033885,
     9225.091896146, 9202.400170845, 9179.648918789999, 9156.839214152, 9133.972144094,
     9111.048808421, 9088.070319180999, 9065.037800279999, 9041.952387068999,
     9018.815225966, 8995.627474001, 8972.390298403, 8949.104876173, 8925.772393629,
     8902.394045964998, 8878.971036781999, 8855.50457764, 8831.995887567999, 8808.446192623,
     8784.856725372, 8761.228724437, 8737.563434012, 8713.862103352, 8690.125986289999,
     8666.356340761, 8642.554428276999, 8618.721513465, 8594.858863528, 8570.967747797,
     8547.049437205998, 8523.105203795]

const COM_DIST = [0.0, 19.634035861439997, 40.774637652938, 61.994508593793995,
     83.29292995072, 104.6691649423, 126.12245883194998, 147.65203903268, 169.2571152265,
     190.93687949574, 212.69050646823, 234.51715347585, 256.41596072631, 278.38605148773996,
     300.42653228769996, 322.53649312415996, 344.71500769082996, 366.96113361466996,
     389.27391270725997, 411.65237122806997, 434.09552016158995, 456.60235550608,
     479.17185857534, 501.80299631216997, 524.49472161387, 547.2459736699899, 570.05567831037,
     592.9227483653599, 615.84608403605, 638.8245732754899, 661.85709217943, 684.9425053874199,
     708.0796664928, 731.2674184612999, 754.5045940593, 777.7900162885999, 801.1224988304999,
     824.5008464937999, 847.9238556754, 871.3903148195, 894.8990048894, 918.4486998378999,
     942.0381670902999, 965.6661680217999, 989.3314584468, 1013.0327891082, 1036.7689061673998,
     1060.5385516971, 1084.3404641789998, 1108.1733789958998, 1132.0360289315, 1155.9271446625,
     1179.8454552556, 1203.7896886602998]

const ANG_DIST = [0.0, 19.506093981206, 40.225831390626, 60.732673380022, 81.02735214455,
     101.11059621019, 120.98313076947998, 140.64567801913998, 160.09895749812998, 179.34368642694,
     198.38058004743996, 217.21035196252, 235.83371447572998, 254.25137893105997, 272.46405605076,
     290.4724562733, 308.27729008916, 325.87926837507996, 343.27910272593, 360.47750578455,
     377.47519156844004, 394.2728757933, 410.87127619306995, 427.27111283612, 443.47310843717,
     459.47798866445, 475.28648244217, 490.89932224774, 506.31724440268994, 521.54098935841,
     536.57130197531, 551.4089317949799, 566.0546333057, 580.5091662006699, 594.77329562868,
     608.8477924363799, 622.7334334032, 636.43100146703, 649.9412859417799, 663.2650827254199,
     676.4031944991999, 689.3564309172799, 702.1256087867, 714.7115522378999, 727.1150928845999,
     739.3370699722, 751.3783305196, 763.2397294473001, 774.9221296956999, 786.4264023328999,
     797.7534266498999, 808.9040902508999, 819.8792891232999, 830.6799277049]

const LUM_DIST = [0.0, 19.762816921707998, 41.330931350639, 63.28256080112399, 85.62185479533,
     108.35297684281, 131.48010404956, 155.00742673052, 178.93914802756998, 203.27948353187998,
     228.03266091311997, 253.20291955482998, 278.79451019686, 304.81169458655995, 331.25874513732,
     358.13994459753997, 385.45958572849, 413.22197099323, 441.43141225638, 470.09223049536996,
     499.20875552470994, 528.78532573216, 558.82628782904, 589.33599661447, 620.31881475361,
     651.77911257193, 683.72126786383, 716.1496657181999, 749.0686983612, 782.4827650129,
     816.3962717637999, 850.813631472, 885.7392636686999, 921.1775944952, 957.1330566462,
     993.6100893418, 1030.6131383128, 1068.146655807, 1106.2151006172, 1144.8229381253,
     1183.9746403694999, 1223.6746861304, 1263.9275610353, 1304.7377576884, 1346.1097758129,
     1388.0481224183, 1430.5573119900998, 1473.6418666921, 1517.3063165939, 1561.5551999186998,
     1606.3930633057998, 1651.8244620983, 1697.8539606434, 1744.4861326186]

const GROWTH_FACTOR_D = [1.0, 0.9966409009584, 0.993021170975, 0.9893850225296, 0.9857326997999,
     0.9820644522585, 0.978380534605, 0.9746812066927, 0.9709667334511, 0.9672373848039,
     0.9634934355829, 0.9597351654362, 0.9559628587335, 0.9521768044659, 0.9483772961421,
     0.94456463168, 0.9407391132946, 0.9369010473811, 0.9330507443948, 0.929188518727,
     0.9253146885768, 0.9214295758196, 0.917533505872, 0.9136268075535, 0.9097098129447,
     0.9057828572426, 0.9018462786131, 0.8979004180403, 0.8939456191736, 0.8899822281723,
     0.8860105935478, 0.8820310660035, 0.8780439982736, 0.8740497449591, 0.8700486623628,
     0.8660411083234, 0.8620274420474, 0.8580080239409, 0.8539832154403, 0.8499533788425,
     0.8459188771346, 0.8418800738236, 0.8378373327661, 0.8337910179982, 0.8297414935658,
     0.8256891233555, 0.8216342709265, 0.8175772993433, 0.8135185710098, 0.8094584475046,
     0.8053972894179, 0.8013354561904, 0.7972733059537, 0.7932111953733]

const GROWTH_FACTOR_F = [0.5126998572951, 0.5166752007981, 0.5209456198548, 0.5252214068795,
     0.5295019302879, 0.5337865518372, 0.5380746269755, 0.5423655052041, 0.5466585304504,
     0.5509530414522, 0.5552483721526, 0.5595438521047, 0.563838806887, 0.568132558527,
     0.5724244259342, 0.5767137253412, 0.5809997707523, 0.5852818743989, 0.5895593472022,
     0.5938314992407, 0.5980976402232, 0.6023570799664, 0.6066091288758, 0.61085309843,
     0.6150883016669, 0.6193140536721, 0.6235296720668, 0.6277344774972, 0.6319277941217,
     0.6361089500979, 0.6402772780658, 0.6444321156292, 0.6485728058323, 0.6526986976317,
     0.6568091463639, 0.6609035142046, 0.6649811706234, 0.6690414928287, 0.6730838662055,
     0.6771076847435, 0.6811123514552, 0.6850972787839, 0.6890618890005, 0.6930056145879,
     0.696927898614, 0.7008281950909, 0.7047059693213, 0.7085606982302, 0.7123918706828,
     0.7161989877874, 0.7199815631826, 0.7237391233091, 0.7274712076658, 0.7311773690493]

const COM_H = [0.0003335640951981429, 0.00033237369077582837, 0.0003311164019378049,
     0.0003298794803941373, 0.00032866285640142793, 0.00032746645996064303,
     0.0003262902207941897, 0.00032513406832466497, 0.00032399793165289537,
     0.00032288173953739976, 0.0003217854203739545, 0.0003207089021754946,
     0.0003196521125537181, 0.0003186149786999385, 0.00031759742736752936,
     0.0003165993848549706, 0.0003156207769894872, 0.00031466152911081526,
     0.000313721566057328, 0.0003128008121510082, 0.00031189919118453015,
     0.00031101662640907766, 0.00031015304052184887, 0.00030930835565644126,
     0.0003084824933720576, 0.00030767537464485695, 0.0003068869198595986,
     0.0003061170488029462, 0.000305365680656388, 0.00030463273399186197,
     0.0003039181267665217, 0.0003032217763196493, 0.0003025435993701211,
     0.0003018835120149112, 0.0003012414297285388, 0.0003006172673631339,
     0.00030001093915001857, 0.00029942235870193597, 0.00029885143901640313,
     0.00029829809247963, 0.00029776223087197684, 0.0002972437653737255,
     0.0002967426065720797, 0.00029625866446930183, 0.0002957918484909039,
     0.0002953420674961558, 0.00029490922978798957, 0.00029449324312429457,
     0.00029409401473072066, 0.0002937114513130251, 0.0002933454590714021,
     0.000292995943714621, 0.0002926628104756253, 0.0002923459641273642]


const COM_H_P = [7e-8, 6.006930707806362e-8, 5.8879595111785904e-8, 
     5.77048057642471e-8, 5.65436909241771e-8, 5.539631661048028e-8, 5.426239960437481e-8, 
     5.3141754079780996e-8, 5.2034171500634614e-8, 5.0939452842564076e-8, 
     4.9857399978773165e-8, 4.878781787537354e-8, 4.7730513975328607e-8, 
     4.668529827370413e-8, 4.56519833359993e-8, 4.463038416347233e-8, 4.362031821837858e-8, 
     4.262160540338953e-8, 4.163406795686277e-8, 4.065753052670967e-8, 3.969181998463066e-8, 
     3.873676555154331e-8, 3.7792198667344174e-8, 3.6857952979455275e-8, 
     3.593386433810119e-8, 3.501977072693157e-8, 3.4115512241977185e-8, 
     3.322093109019046e-8, 3.233587153470907e-8, 3.1460179846468915e-8, 
     3.059370431066888e-8, 2.973629518746432e-8, 2.8887804683527015e-8, 2.8048086884506395e-8, 
     2.721699780540519e-8, 2.6394395309839293e-8, 2.5580139088670515e-8, 
     2.477409063351769e-8, 2.397611322631209e-8, 2.3186071892382766e-8, 2.240383339138877e-8, 
     2.1629266196943872e-8, 2.086224042848283e-8, 2.0102627906195153e-8, 
     1.9350302052093087e-8, 1.860513787398953e-8, 1.7867012008005382e-8, 
     1.7135802638399202e-8, 1.6411389389191127e-8, 1.5693653807629034e-8, 1.498247739361126e-8, 
     1.4277748859281751e-8, 1.3579336976668888e-8, 1.288719172040533e-8]

