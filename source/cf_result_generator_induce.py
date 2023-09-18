# This file generates explanation quality based on metrics.

import torch
import argparse
from gnn_trainer import GNNTrainer
import data_utils
import cf_metrics as metrics
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Mutagenicity',
                        choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'syn1', 'syn4', 'syn5'],
                        help="Dataset name")
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN layer type to use.')
    parser.add_argument('--device', type=int, default=0, help='Index of cuda device to use. Default is 0.')
    parser.add_argument('--gnn_run', type=int, default=1, help='random seed for gnn run')
    parser.add_argument('--explainer_run', type=int, default=1, help='random seed for explainer run')
    parser.add_argument('--explainer_name', type=str, choices=['cff_0.0','rcexplainer_0.0', 'clear'], help='Name of explainer to use.')
    parser.add_argument('--explanation_metric','--list', nargs='+', help='<Required> Set flag', required=True, type=str, choices=['sufficiency', 'size', 'sparsity', 'stability_noise', 'stability_seed', 'stability_base', 'feasibility', 'accuracy'])
    parser.add_argument('--verbose', type=int, default=1, help='Default: 1 (print computed metric), else 0')
    # parser.add_argument('--explanation_metric', type=str, choices=['sufficiency', 'size', 'sparsity', 'stability_noise', 'stability_seed', 'stability_base', 'feasibility'],
    #                     help='Explanation metric to use.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
    model = trainer.load(args.gnn_run)
    model.eval()

    result_folder = f'data/{args.dataset}/{args.explainer_name}/'

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    dataset = data_utils.load_dataset(args.dataset)
    
    if args.dataset in ['syn1', 'syn4', 'syn5']:
        if args.dataset == 'syn1':
            train_idx = [333, 97, 380, 311, 281, 10, 56, 680, 67, 488, 467, 492, 465, 78, 186, 696, 478, 216, 65, 116, 676, 236, 613, 37, 554, 382, 325, 668, 440, 201, 543, 495, 107, 124, 328, 221, 587, 237, 266, 61, 239, 66, 356, 322, 567, 515, 46, 87, 424, 572, 273, 123, 570, 342, 521, 673, 410, 577, 282, 252, 42, 117, 559, 590, 686, 57, 511, 243, 291, 26, 598, 213, 489, 413, 47, 277, 421, 62, 561, 157, 386, 594, 507, 188, 436, 651, 553, 55, 564, 363, 296, 376, 187, 430, 43, 115, 137, 295, 257, 354, 161, 475, 102, 408, 180, 160, 442, 22, 58, 659, 271, 60, 72, 32, 75, 381, 288, 194, 591, 81, 460, 175, 601, 150, 48, 120, 109, 523, 334, 510, 29, 370, 628, 637, 513, 389, 650, 634, 428, 130, 17, 474, 302, 390, 101, 119, 191, 133, 165, 398, 387, 606, 480, 362, 401, 660, 687, 212, 621, 466, 485, 625, 9, 304, 623, 336, 666, 575, 379, 287, 337, 77, 493, 153, 443, 438, 422, 481, 626, 453, 426, 64, 263, 491, 192, 284, 85, 524, 396, 324, 656, 234, 350, 233, 459, 528, 532, 518, 642, 317, 534, 299, 285, 557, 307, 198, 612, 638, 602, 536, 603, 79, 458, 18, 73, 383, 114, 624, 355, 135, 499, 373, 405, 12, 406, 279, 441, 588, 586, 599, 174, 195, 461, 500, 301, 689, 618, 654, 217, 238, 449, 583, 3, 627, 95, 448, 399, 139, 679, 596, 551, 140, 463, 223, 698, 88, 529, 321, 368, 147, 256, 372, 403, 156, 222, 502, 178, 584, 446, 416, 141, 199, 486, 275, 340, 595, 435, 400, 509, 685, 108, 59, 255, 359, 592, 241, 264, 437, 313, 620, 83, 661, 270, 691, 525, 278, 8, 112, 34, 231, 662, 329, 580, 104, 189, 93, 579, 482, 267, 699, 24, 21, 118, 138, 341, 30, 268, 326, 397, 695, 531, 166, 323, 514, 290, 692, 395, 505, 468, 197, 182, 283, 375, 615, 300, 427, 203, 522, 314, 411, 417, 566, 53, 357, 52, 96, 578, 27, 316, 286, 503, 89, 655, 549, 196, 319, 533, 539, 504, 432, 353, 556, 129, 439, 645, 261, 207, 169, 608, 128, 494, 360, 470, 106, 548, 303, 84, 80, 565, 348, 418, 339, 297, 204, 517, 550, 149, 352, 351, 152, 94, 526, 452, 132, 269, 684, 682, 391, 7, 541, 113, 6, 163, 483, 392, 501, 210, 409, 555, 450, 472, 484, 552, 248, 347, 346, 11, 469, 644, 244, 71, 126, 431, 445, 154, 366, 657, 20, 576, 677, 50, 614, 69, 568, 309, 669, 343, 168, 220, 51, 633, 159, 540, 653, 635, 183, 569, 312, 260, 111, 652, 76, 122, 393, 344, 45, 639, 506, 318, 694, 289, 155, 631, 664, 39, 49, 306, 68, 476, 259, 177, 98, 434, 641, 636, 251, 479, 678, 315, 193, 358, 527, 394, 4, 648, 227, 293, 361, 407, 593, 229, 148, 622, 31, 535, 611, 90, 378, 690, 173, 158, 185, 520, 170, 629, 23, 249, 674, 640, 327, 308, 144, 681, 171, 19, 457, 127, 179, 546, 125, 136, 365, 38, 544, 384, 369, 423, 538, 190, 335, 225, 54, 162, 181, 332, 224, 146, 331, 36, 547, 420, 530, 258, 338, 131, 145, 247, 91, 675, 658, 240, 693, 571, 388, 16, 134, 121, 433, 40, 0]
            val_idx = ... # todo
            test_idx = [563, 415, 605, 320, 597, 345, 367, 310, 496, 671, 451, 464, 607, 562, 498, 456, 688, 419, 537, 425, 616, 672, 487, 560, 414, 604, 643, 617, 385, 508, 542, 471, 667, 558, 462, 412, 364, 349, 429, 609, 581, 402, 589, 444, 574, 630, 512, 697, 600, 490, 582, 670, 649, 610, 497, 573, 330, 647, 447, 516, 619, 305, 545, 404, 477, 377, 585, 455, 371, 646, 374, 519, 454, 665, 663, 632]
        elif args.dataset == 'syn4':
            train_idx = [498, 852, 868, 518, 364, 248, 474, 70, 281, 709, 635, 770, 578, 842, 627, 278, 106, 72, 720, 643, 513, 393, 421, 261, 344, 30, 79, 451, 254, 110, 172, 858, 2, 114, 538, 660, 503, 177, 150, 744, 628, 88, 229, 238, 507, 687, 415, 33, 558, 4, 319, 683, 382, 120, 164, 445, 402, 803, 828, 559, 159, 593, 259, 511, 201, 240, 138, 493, 707, 394, 657, 436, 640, 819, 556, 755, 649, 681, 191, 599, 133, 768, 472, 270, 384, 619, 747, 775, 815, 28, 298, 466, 809, 678, 35, 708, 497, 173, 704, 588, 607, 306, 795, 52, 129, 772, 833, 475, 267, 3, 645, 804, 134, 369, 48, 365, 540, 241, 244, 455, 349, 340, 44, 176, 165, 322, 847, 308, 405, 321, 427, 420, 145, 680, 668, 111, 526, 253, 36, 287, 594, 605, 163, 50, 243, 41, 728, 454, 840, 274, 12, 846, 477, 633, 827, 509, 548, 419, 142, 311, 59, 80, 676, 284, 746, 699, 808, 603, 855, 128, 156, 608, 67, 401, 318, 802, 85, 275, 260, 663, 171, 135, 658, 403, 773, 555, 84, 831, 859, 216, 692, 564, 702, 552, 207, 348, 265, 136, 355, 73, 613, 62, 345, 435, 865, 697, 740, 31, 81, 379, 1, 89, 395, 143, 257, 357, 583, 60, 481, 199, 362, 63, 677, 476, 473, 691, 330, 609, 331, 309, 586, 655, 832, 396, 124, 777, 443, 651, 397, 272, 222, 152, 154, 366, 478, 587, 96, 354, 653, 727, 569, 456, 807, 537, 524, 202, 766, 54, 82, 665, 359, 838, 452, 620, 329, 251, 122, 53, 198, 438, 857, 718, 437, 754, 313, 656, 302, 752, 400, 292, 561, 568, 418, 581, 410, 622, 508, 263, 24, 843, 192, 579, 21, 285, 324, 14, 812, 458, 168, 213, 332, 601, 500, 64, 650, 204, 286, 457, 210, 350, 236, 104, 716, 612, 836, 793, 688, 482, 205, 724, 667, 532, 787, 304, 801, 834, 584, 824, 76, 375, 148, 43, 282, 519, 661, 299, 512, 671, 196, 220, 517, 20, 376, 69, 103, 730, 669, 446, 504, 167, 378, 217, 798, 841, 460, 468, 341, 116, 750, 522, 845, 738, 269, 849, 469, 139, 197, 294, 611, 751, 245, 471, 830, 496, 247, 703, 484, 447, 520, 367, 679, 141, 490, 215, 39, 638, 398, 866, 485, 6, 615, 544, 483, 462, 835, 784, 550, 283, 742, 131, 580, 823, 487, 16, 774, 351, 563, 221, 333, 93, 320, 585, 105, 121, 146, 225, 792, 779, 682, 790, 572, 174, 183, 237, 184, 40, 448, 256, 296, 444, 597, 346, 233, 736, 630, 685, 799, 108, 713, 212, 523, 853, 27, 557, 534, 7, 61, 118, 211, 576, 190, 489, 441, 531, 317, 117, 337, 268, 741, 753, 499, 670, 411, 719, 684, 767, 625, 158, 312, 98, 810, 56, 390, 214, 623, 761, 450, 170, 352, 264, 629, 541, 74, 543, 75, 547, 646, 363, 442, 227, 310, 822, 361, 87, 342, 125, 130, 796, 38, 266, 271, 675, 461, 637, 90, 565, 491, 195, 814, 632, 618, 813, 386, 19, 711, 326, 97, 336, 567, 368, 389, 495, 763, 465, 854, 314, 101, 745, 749, 188, 510, 295, 430, 358, 280, 765, 654, 107, 723, 186, 157, 560, 778, 206, 829, 600, 689, 480, 353, 47, 325, 123, 505, 412, 26, 760, 279, 717, 178, 756, 467, 343, 757, 575, 242, 648, 494, 200, 839, 46, 722, 102, 381, 479, 759, 439, 166, 506, 781, 223, 86, 149, 464, 732, 820, 126, 160, 739, 219, 339, 372, 276, 373, 582, 743, 327, 856, 185, 539, 463, 252, 848, 502, 95, 11, 169, 226, 182, 300, 806, 100, 864, 786, 297, 338, 224, 112, 155, 844, 262, 551, 232, 119, 783, 514, 380, 83, 694, 17, 610, 385, 218, 470, 115, 209, 789, 370, 659, 180, 595, 725, 769, 571, 15, 29, 32, 715, 521, 710, 666, 246, 797, 706, 193, 673, 528, 42, 288, 631, 371, 573, 335, 591, 714, 234, 179, 289, 258, 672, 194, 696, 429, 825, 616, 652, 862, 782, 58, 94, 18, 162, 453, 255, 189, 700, 323, 592, 698, 91, 729, 794, 598, 536, 290, 870, 695]
            val_idx = [498, 852, 868, 518, 364, 248, 474, 70, 281, 709, 635, 770, 578, 842, 627, 278, 106, 72, 720, 643, 513, 393, 421, 261, 344, 30, 79, 451, 254, 110, 172, 858, 2, 114, 538, 660, 503, 177, 150, 744, 628, 88, 229, 238, 507, 687, 415, 33, 558, 4, 319, 683, 382, 120, 164, 445, 402, 803, 828, 559, 159, 593, 259, 511, 201, 707, 657, 640, 819, 556, 755, 649, 681, 599, 768, 619, 747, 775, 815, 809]
            test_idx = [811, 589, 686, 604, 861, 726, 771, 624, 762, 731, 577, 764, 542, 817, 674, 574, 758, 785, 867, 553, 621, 530, 602, 529, 606, 821, 590, 818, 837, 863, 642, 735, 748, 525, 690, 705, 644, 712, 533, 549, 535, 737, 516, 566, 515, 545, 701, 734, 546, 791, 721, 554, 851, 634, 780, 664, 626, 869, 641, 816, 636, 614, 860, 693, 826, 570, 527, 617, 647, 562, 596, 805]
            test_idx_mapped = np.arange(0, len(test_idx))
        elif args.dataset == 'syn5':
            train_idx = [1073, 669, 664, 837, 662, 1178, 538, 237, 298, 886, 552, 630, 361, 822, 624, 333, 493, 682, 686, 876, 36, 481, 752, 243, 1168, 730, 850, 824, 459, 577, 677, 78, 388, 282, 1056, 378, 865, 875, 642, 386, 477, 592, 1107, 962, 666, 435, 128, 328, 487, 890, 90, 168, 484, 561, 408, 659, 1176, 139, 845, 573, 185, 934, 678, 456, 205, 739, 519, 1122, 47, 747, 448, 794, 1033, 1219, 723, 993, 496, 564, 390, 582, 781, 949, 929, 1079, 1110, 578, 737, 1091, 587, 534, 306, 812, 922, 688, 423, 1099, 562, 395, 75, 802, 387, 902, 429, 479, 777, 1230, 923, 990, 1163, 1031, 268, 364, 422, 461, 1133, 1124, 644, 1203, 648, 41, 848, 202, 379, 228, 1032, 458, 175, 410, 546, 421, 240, 701, 725, 330, 1070, 869, 1197, 368, 191, 986, 676, 277, 1067, 1172, 1027, 212, 1095, 1209, 1045, 898, 560, 672, 936, 46, 544, 311, 199, 1224, 1150, 58, 871, 543, 518, 74, 383, 978, 532, 823, 303, 1167, 773, 892, 211, 604, 265, 841, 971, 1223, 344, 417, 1135, 1013, 358, 258, 256, 650, 914, 1025, 259, 1085, 655, 591, 391, 454, 753, 698, 278, 1114, 1191, 542, 918, 609, 105, 1038, 770, 472, 431, 501, 1165, 899, 893, 693, 248, 374, 436, 252, 627, 61, 1141, 541, 734, 178, 879, 1204, 1202, 766, 957, 1097, 870, 1103, 489, 1156, 85, 130, 239, 708, 226, 121, 728, 377, 568, 339, 106, 673, 1094, 549, 166, 29, 1083, 19, 1096, 511, 940, 1059, 293, 49, 530, 603, 1154, 104, 571, 910, 230, 599, 1158, 785, 409, 778, 1187, 943, 88, 528, 87, 131, 1052, 186, 1222, 499, 371, 263, 699, 647, 687, 1123, 281, 1002, 1060, 1130, 229, 476, 334, 83, 101, 1225, 411, 357, 56, 985, 796, 815, 948, 1020, 376, 1216, 710, 679, 792, 288, 533, 896, 1101, 192, 581, 322, 315, 235, 1064, 917, 1169, 290, 646, 631, 1072, 343, 757, 193, 503, 222, 983, 1062, 295, 329, 665, 103, 365, 814, 466, 67, 768, 805, 52, 1131, 1161, 1200, 1186, 855, 526, 570, 113, 760, 291, 1065, 181, 654, 969, 966, 138, 853, 30, 21, 878, 1057, 470, 989, 789, 478, 523, 825, 551, 23, 34, 208, 832, 920, 342, 1183, 1142, 626, 490, 207, 8, 774, 625, 1174, 930, 196, 513, 690, 1036, 11, 784, 231, 309, 1, 909, 177, 392, 639, 132, 313, 1108, 658, 1035, 559, 731, 441, 656, 931, 958, 279, 556, 1051, 751, 671, 1210, 187, 215, 37, 700, 804, 786, 583, 1005, 645, 1152, 537, 705, 527, 416, 809, 394, 420, 1115, 606, 572, 76, 398, 1043, 469, 1129, 588, 576, 953, 621, 1047, 6, 954, 232, 1082, 349, 553, 612, 1028, 204, 405, 286, 124, 206, 336, 244, 520, 321, 475, 684, 704, 352, 1098, 1220, 369, 586, 404, 439, 903, 147, 1088, 53, 1034, 1026, 535, 1113, 82, 453, 380, 1120, 1117, 1037, 251, 795, 133, 510, 924, 66, 262, 742, 1119, 995, 938, 901, 50, 312, 720, 253, 1017, 1206, 91, 1055, 782, 967, 129, 857, 521, 27, 1068, 162, 941, 457, 594, 430, 126, 724, 874, 44, 793, 1078, 10, 426, 937, 1014, 1151, 987, 155, 213, 447, 157, 38, 916, 445, 540, 136, 835, 238, 641, 24, 712, 114, 350, 446, 820, 77, 643, 9, 788, 1145, 158, 683, 506, 146, 999, 1058, 150, 505, 1166, 736, 276, 1006, 255, 622, 885, 942, 727, 632, 882, 702, 732, 1127, 492, 135, 486, 325, 932, 670, 319, 242, 60, 1159, 830, 109, 468, 660, 451, 94, 585, 507, 1049, 696, 233, 102, 689, 434, 26, 326, 1121, 1198, 182, 980, 372, 595, 566, 810, 858, 1184, 860, 1066, 141, 674, 864, 264, 43, 1016, 866, 1132, 783, 601, 267, 210, 485, 711, 254, 1162, 64, 1157, 611, 3, 960, 926, 915, 54, 190, 1226, 1181, 842, 0, 1100, 250, 415, 596, 1010, 80, 1177, 852, 977, 1087, 629, 363, 959, 300, 840, 1089, 945, 973, 1139, 968, 1171, 184, 767, 144, 1144, 667, 502, 801, 933, 1179, 1048, 509, 1081, 833, 18, 302, 455, 1193, 651, 719, 500, 653, 661, 976, 715, 593, 907, 800, 970, 89, 755, 703, 714, 1039, 42, 84, 273, 1102, 341, 14, 418, 25, 1207, 863, 717, 531, 975, 1023, 1011, 362, 462, 982, 63, 442, 597, 623, 1041, 397, 292, 691, 316, 1112, 432, 557, 271, 200, 1044, 952, 201, 1084, 1021, 119, 7, 721, 310, 198, 1148, 834, 209, 474, 1086, 28, 838, 1208, 620, 494, 685, 716, 1199, 550, 1008, 189, 165, 638, 827, 283, 984, 1170, 862, 347, 389, 602, 366, 880, 413, 183, 57, 176, 297, 844, 1093, 1007, 122, 463, 2, 367, 234, 839, 460, 956, 92, 1061, 1146, 979, 887, 895, 1029, 127, 1182, 947, 1054, 1018, 799, 296, 813, 1195, 904, 163, 713, 274, 575, 574, 1175, 738, 1050, 790, 706, 928, 524, 680, 780, 614, 913, 285, 4, 965, 749, 340, 854, 634, 33, 891, 317, 889, 424, 821, 403, 99, 160, 488, 332, 974, 1063, 51, 861, 65, 170, 981, 908, 1153, 266, 272, 1185, 112, 97, 223, 203, 906, 548, 289, 951, 1140, 98, 598, 224, 817, 360, 668, 1000, 536, 877, 218, 270, 406, 508, 675, 221, 140, 1190, 152, 40, 214, 294, 247, 547, 605, 327, 68, 829, 798, 194, 467, 828, 771, 1116, 911, 756, 164, 414, 1030, 558, 348, 1128, 48, 787, 1217, 517, 217, 22, 71, 707, 761, 859, 62, 628, 775, 1205, 174, 95, 955, 143, 884, 393, 748, 744, 512, 79, 888, 765, 743, 452, 169, 17, 110, 86, 589, 1092, 718, 616, 134, 16, 996, 381, 370, 764, 1009, 925, 482, 188, 988, 304, 991, 1046, 754, 495, 39, 1111, 1149, 399, 619, 1105, 73, 763, 1053, 331, 729, 151, 287, 849, 867, 762, 843, 1001, 1196, 772, 356, 355, 836, 260, 219, 427, 818, 1004, 12, 735, 994, 758, 425, 31, 819, 70, 1218]
            val_idx = [1073, 669, 664, 837, 662, 1178, 538, 237, 298, 886, 552, 630, 361, 822, 624, 333, 493, 682, 686, 876, 36, 481, 752, 243, 1168, 730, 850, 824, 459, 577, 677, 78, 388, 282, 1056, 378, 865, 875, 642, 386, 477, 592, 1107, 962, 666, 435, 128, 328, 487, 890, 90, 168, 484, 561, 408, 659, 1176, 139, 845, 573, 185, 934, 678, 456, 205, 739, 519, 1122, 47, 747, 448, 794, 1033, 1219, 723, 993, 496, 564, 390, 582, 781, 949, 929, 1079, 1110, 578, 737, 1091, 587, 534, 306, 812, 423, 395, 75, 387, 429, 479, 268, 364, 422, 461, 41, 202, 379, 228, 458, 175, 410, 421, 240, 330, 368, 191, 277, 212, 46, 311, 199, 58]
            test_idx = [617, 972, 912, 726, 584, 964, 1194, 516, 1164, 797, 569, 900, 663, 950, 856, 1180, 1022, 608, 811, 515, 740, 759, 692, 1126, 539, 1024, 873, 745, 1040, 709, 1173, 935, 1076, 992, 997, 635, 1118, 633, 750, 769, 1042, 567, 1189, 1213, 921, 529, 563, 746, 883, 1192, 1137, 791, 741, 847, 733, 905, 1019, 1228, 1155, 1134, 831, 1074, 851, 555, 607, 590, 525, 1229, 1106, 776, 961, 963, 868, 1069, 897, 1143, 649, 722, 806, 652, 545, 872, 946, 807, 600, 939, 1075, 1015, 1227, 522, 1136, 554, 1138, 636, 919, 1147, 610, 1003, 846, 881, 1160, 1214, 1109, 514, 580, 565, 613, 1125, 640, 894, 637, 927, 681, 695, 1104, 1071, 808, 694, 1211, 697, 1090, 615, 944, 1201, 579, 657, 618, 1080, 816]
            test_idx_mapped = np.arange(0, len(test_idx))
        # indices = [train_idx, test_idx, test_idx_mapped] #_mapped
        indices = [train_idx, test_idx, test_idx] #_mapped
    else:
        splits, indices = data_utils.split_data(dataset)
  
        
    explanations = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), args.explainer_run)
    print(len(indices[0]), len(indices[1]), len(indices[2]))

    #generate score
    if(args.explainer_name == 'rcexplainer_0.0'):
        explanations_updated = metrics.remove_top_k_incremental(explanations, model, device)
        explanations = explanations_updated
    
    print(f'Started: {args.dataset}, {args.explainer_name}, {args.gnn_type}, {args.explanation_metric}')

    #flag to track non-implemeted metric
    flag = False
    # Generate explanation quality based on metrics
    if 'sufficiency' in args.explanation_metric:
        flag = True
        sufficiency_scores_dict = {}
        sufficiency = metrics.sufficiency(explanations, indices[2])
        sufficiency_scores_dict['sufficiency'] = sufficiency
        torch.save(sufficiency_scores_dict, result_folder + f'cf_sufficiency_{args.gnn_type}_run_{args.explainer_run}.pt')
        if(args.verbose):
            print(f'Sufficiency: {sufficiency}')
        
    if 'size' in args.explanation_metric:
        flag = True
        size_scores_dict = {}
        avg_size, stdev_size = metrics.size(args.explainer_name, explanations, indices[2])
        size_scores_dict['size'] = [avg_size, stdev_size]
        torch.save(size_scores_dict, result_folder + f'cf_size_{args.gnn_type}_run_{args.explainer_run}.pt')
        if(args.verbose):
            print(f'Size: {avg_size} +- {stdev_size}')
        
    
    if 'sparsity' in args.explanation_metric:
        flag = True
        sparsity_scores_dict = {}
        sparsity = metrics.sparsity(args.explainer_name, explanations, indices[2])
        sparsity_scores_dict['sparsity'] = sparsity
        torch.save(sparsity_scores_dict, result_folder + f'cf_sparsity_{args.gnn_type}_run_{args.explainer_run}.pt')
        if(args.verbose):
            print(f'Sparsity: {sparsity}')
        
    if 'stability_noise' in args.explanation_metric:
        flag = True
        ks = [1, 2, 3, 4, 5]
        metric_names = ['jaccard', 'size', 'sufficiency']
        robustness_scores_dict = {metric: [] for metric in metric_names}
        
        for k in ks:
            explanations_noise = data_utils.load_explanations_noisy(args.dataset, args.explainer_name, args.gnn_type, device, args.explainer_run, k=k)
            if(args.explainer_name == 'rcexplainer_0.0'):
                explanations_updated = metrics.remove_top_k_incremental(explanations_noise, model, device)
                explanations_noise = explanations_updated
            
            #generate score
            avg_jaccard, stdev_jaccard = metrics.robustness(args.explainer_name, explanations, explanations_noise, indices[2])
            sufficiency = metrics.sufficiency(explanations_noise, indices[2])
            avg_size, stdev_size = metrics.size(args.explainer_name, explanations_noise, indices[2])
            
            robustness_scores_dict['jaccard'].append(avg_jaccard)
            robustness_scores_dict['size'].append([avg_size, stdev_size])
            robustness_scores_dict['sufficiency'].append(sufficiency)
            if(args.verbose):
                print(f'----------- noise budget:{k} ---------------')
                print(f'Jaccard: {avg_jaccard}')
                print(f'Sufficiency: {sufficiency}')
                print(f'Size: {avg_size} +- {stdev_size}')
            
        torch.save(robustness_scores_dict, result_folder + f'cf_stability_noise_{args.gnn_type}_run_{args.explainer_run}.pt')
        
    if 'stability_seed' in args.explanation_metric:
        flag = True
        seeds = [1, 2, 3]
        metric_names = ['jaccard', 'size', 'sufficiency']
        stability_seed_scores_dict = {metric: {} for metric in metric_names}

        for i in range(len(seeds)):
            explanations_seed_i = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), run=seeds[i])
            if(args.explainer_name == 'rcexplainer_0.0'):
                explanations_updated = metrics.remove_top_k_incremental(explanations_seed_i, model, device)
                explanations_seed_i = explanations_updated

            sufficiency = metrics.sufficiency(explanations_seed_i, indices[2])
            avg_size, stdev_size = metrics.size(args.explainer_name, explanations_seed_i, indices[2])
            
            stability_seed_scores_dict['size'][f'{i}'] = [avg_size, stdev_size]
            stability_seed_scores_dict['sufficiency'][f'{i}'] = sufficiency

            if(args.verbose):
                print(f'----------- seed {i} ---------------')    
                print(f'Sufficiency: {sufficiency}')
                print(f'Size: {avg_size} +- {stdev_size}')

            for j in range(i + 1, len(seeds)):
                explanations_seed_j = data_utils.load_explanations(args.dataset, args.explainer_name, args.gnn_type, torch.device('cpu'), run=seeds[j])
                if(args.explainer_name == 'rcexplainer_0.0'):
                    explanations_updated = metrics.remove_top_k_incremental(explanations_seed_j, model, device)
                    explanations_seed_j = explanations_updated

                avg_jaccard, stdev_jaccard = metrics.robustness(args.explainer_name, explanations_seed_i, explanations_seed_j, indices[2])
                stability_seed_scores_dict['jaccard'][f'{i}_{j}'] = avg_jaccard

                if(args.verbose):
                    print(f'################ seed {j} ###############')
                    print(f'Jaccard: {avg_jaccard}')
                    
        torch.save(stability_seed_scores_dict, result_folder + f'cf_stability_seed_{args.gnn_type}_run_{args.explainer_run}.pt')
        
    if 'stability_base' in args.explanation_metric:
        flag = True
        bases = ['gcn', 'gat', 'gin', 'sage']
        metric_names = ['jaccard', 'size', 'sufficiency']
        stability_base_scores_dict = {metric: {} for metric in metric_names}
   
        for i in range(len(bases)):
            explanations_base1 = data_utils.load_explanations(args.dataset, args.explainer_name, bases[i], torch.device('cpu'), run=1)
            if(args.explainer_name == 'rcexplainer_0.0'):
                explanations_updated = metrics.remove_top_k_incremental(explanations_base1, model, device)
                explanations_base1 = explanations_updated

            sufficiency = metrics.sufficiency(explanations_base1, indices[2])
            avg_size, stdev_size = metrics.size(args.explainer_name, explanations_base1, indices[2])
            
            stability_base_scores_dict['size'][bases[i]] = [avg_size, stdev_size]
            stability_base_scores_dict['sufficiency'][bases[i]] = sufficiency
            
            if(args.verbose):
                print(f'----------- base: {bases[i]} ---------------')    
                print(f'Sufficiency: {sufficiency}')
                print(f'Size: {avg_size} +- {stdev_size}')

            for j in range(i + 1, len(bases)):
                explanations_base2 = data_utils.load_explanations(args.dataset, args.explainer_name, bases[j], torch.device('cpu'), run=1)
                if(args.explainer_name == 'rcexplainer_0.0'):
                    explanations_updated = metrics.remove_top_k_incremental(explanations_base2, model, device)
                    explanations_base2 = explanations_updated

                avg_jaccard, stdev_jaccard = metrics.robustness(args.explainer_name, explanations_base1, explanations_base2, indices[2])
                stability_base_scores_dict['jaccard'][bases[i]+'_'+ bases[j]] = avg_jaccard

                if(args.verbose):
                    print(f'################ base: {bases[j]} ###############')
                    print(f'Jaccard: {avg_jaccard}')
                
        torch.save(stability_base_scores_dict, result_folder + f'cf_stability_base_run_{args.explainer_run}.pt')

    if 'feasibility' in args.explanation_metric:
        flag = True
        assert(args.dataset in ['Mutagenicity', 'Proteins', 'Mutag','AIDS'])
        feasibility_scores_dict = {}
        e_c, o_c, chi_sq = metrics.feasibility(args.explainer_name, explanations, indices[2])
        feasibility_scores_dict['feasibility'] = {'expected_count':e_c, 'observed_count':o_c, 'chi_squared':chi_sq}
        torch.save(feasibility_scores_dict, result_folder + f'cf_feasibility_{args.gnn_type}_run_{args.explainer_run}.pt')

        if(args.verbose):
            print(f'----------- Feasibility ---------------')    
            print(f'Expected_count: {e_c}')
            print(f'Observed_count: {o_c}')
            print(f'Chi_squared: {chi_sq}')
            
    if 'accuracy' in args.explanation_metric:
        flag = True
        assert(args.dataset in ['syn1', 'syn4', 'syn5'])
        accuracy_scores_dict = {}
        accuracy = metrics.accuracy(args.explainer_name, args.dataset, explanations, indices[2]) #indices[2]
        accuracy_scores_dict['accuracy'] = accuracy
        torch.save(feasibility_scores_dict, result_folder + f'cf_accuracy_{args.gnn_type}_run_{args.explainer_run}.pt')

        if(args.verbose):
            print(f'Accuracy: {accuracy}')

    if(flag == False):
        raise NotImplementedError

    print(f'Finished: {args.dataset}, {args.explainer_name}, {args.gnn_type}, {args.explanation_metric}')
