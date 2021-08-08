import ../stat
from std/fenv import epsilon
import math
let sample = [-1.8289045372471258, 0.6366135106906864, 2.016118225303785, 0.7196873323392996, 1.715192425771572, -1.1811517644900131, 0.9137202980575135, -0.6169227519769487, -0.14403112323056258, 2.1619645145355824, 0.550277670728083, -0.28040392658922103, -1.0193586542677717, -0.6007076447836636, -0.49736946960645845, 0.7183969888433221, -1.7742668788785527, -2.0788190295251687, 1.0123696571341303, -0.4072763339219371, 0.2610202893567192, 1.1090510634133217, -0.5124613529597237, -0.1313020402713572, 1.8090825255907166, 1.1561989276643803, 1.2992567592012887, 0.9357280748890917, -0.5304663677724003, -0.5078147333285467, -0.536127588815893, -0.15837851332479463, 1.4345596415648891, -0.6822414667191297, 1.909517202778697, -1.7315831905448176, -0.9278224328577886, 1.2399086322426272, 0.8990142501601955, -1.8010340305874042, 0.12847346881611238, 0.4498788877419787, -0.46934827127576073, 0.5473531822820835, -1.854741888937016, -0.20599036318047348, -1.1483855151502245, 0.7554741839307241, -0.8208052723115216, -1.5944593864884438, -0.9353030810735327, 0.08834233764924913, -0.08568732478703946, -0.18607976598145654, 1.038826158667134, 1.250259012789156, -0.2739792542863674, -0.45171935893474474, 0.9307171396032611, 0.6829847700261658, 0.8789243551282011, -0.3835953671837799, 1.4885468588498252, -2.4775595319462664, -1.0904929139665471, -0.8316513212643589, -1.4922926190326211, 1.043834732375302, -1.5033295362386474, 0.8103139197642216, 0.41886069696445577, -0.46942472421939296, -0.9755663592964534, -0.5393931586241599, 0.3418927753739093, -0.6764702295063617, -0.4432058078556312, 0.5056496475698162, 0.046522674389286936, 0.8504288687912052, -0.831979604648102, -0.60599780717075, 0.33935681524403116, -0.20534627819607632, -0.40263897982612956, -2.057807489261788, -0.5437578717831507, 0.5844805994761679, 0.7301945712161301, -0.9848114769473572, -1.4963105482083219, -0.5817290626258592, 1.2547236874565701, 1.2620087096944923, -1.2303223208033212, 0.44323517820795383, -0.08062616903527871, 0.5575845920010052, -0.6367474292275008, 1.5596648962947397] 
let weight = [0.5906925497996864, 0.7783384778765643, 0.5136239515178409, 0.6711842302359649, 0.5136482925491235, 0.10525229211534026, 0.3518794500558682, 0.9959059034631299, 0.9327473672289002, 0.6789806405885435, 0.057881204025142075, 0.489341060675513, 0.3041221566380824, 0.41466435240205657, 0.5282665039804311, 0.8001301509099783, 0.5017369166408513, 0.2631070758704478, 0.6719134693898368, 0.7219821370855124, 0.0016031026281101424, 0.36334889002639503, 0.7126667565502857, 0.08850520858549948, 0.717858172376264, 0.9046206913143263, 0.023573309855305924, 0.6764004947718654, 0.9845679097300735, 0.5014819586622332, 0.03515396706032281, 0.43481606538096884, 0.24727362072428827, 0.9477437803589889, 0.5539549913536602, 0.8893557442728443, 0.8166145160657124, 0.6665478318342714, 0.7359248296682717, 0.5279228679228388, 0.01767544064808746, 0.31727440080671854, 0.09973434943660098, 0.9880658450766584, 0.7503454802245689, 0.9981131284306756, 0.4446185834445098, 0.18413711258508003, 0.5129569095849522, 0.5426295780994044, 0.5287778529708693, 0.09951207769514936, 0.4659929939074996, 0.5156417986160133, 0.3845094593315045, 0.044440637009341866, 0.951690624506324, 0.8840935427427398, 0.1619060913574819, 0.24710843798958249, 0.9225449377322374, 0.9175734882767228, 0.6475405904191918, 0.005327799333914585, 0.06578552968332241, 0.19400525270531177, 0.9376216051916326, 0.39673416673658235, 0.039436837852378726, 0.9779716654321182, 0.33552372791703033, 0.6140454528437026, 0.22006544796907856, 0.950767879873117, 0.7463873279823349, 0.8502736105583368, 0.012316117232022639, 0.47686684922656075, 0.9001978604883224, 0.09539486305378853, 0.4379912647046823, 0.5263895652108053, 0.6530298061270874, 0.8891654906088607, 0.6144435342651748, 0.9390515057207409, 0.7861719302771472, 0.19368679687689783, 0.6189375314242419, 0.9146363842580826, 0.059226373090987994, 0.20206587225839512, 0.42056425440846956, 0.8670889124628608, 0.7228106271890884, 0.8197618272283145, 0.9602340885065446, 0.40669735139415764, 0.5587492619183383, 0.48441592863817373] 
assert abs(mean(sample) - -0.050297892104046366) < epsilon(float), "[STAT TEST] Failed at mean" 
assert abs(std(sample) - 1.0411575298061162) < epsilon(float), "[STAT TEST] Failed at std" 
assert abs(median(sample) - -0.17222913965312558) < epsilon(float), "[STAT TEST] Failed at median" 
assert abs(mad(sample) - 0.7832055805318674) < epsilon(float), "[STAT TEST] Failed at mad" 
assert abs(weighted_mean(sample, weight) - -0.05034658741485623) < epsilon(float), "[STAT TEST] Failed at weighted_mean" 
echo "[STAT TEST] passed"
