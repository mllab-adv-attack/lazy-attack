import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')
sample = 500
loss = 'C&W loss'
lg_nat = [28.8, 3.8, 0.0, 0, 0]
pgd_nat = [60.2, 7.0, 0.8, 0, 0]
ldg_nat = [28.8, 3.6, 0, 0, 0]
lrg_nat = [29.2, 3.6, 0.2, 0, 0]
ldg_v2_nat = [29.2, 3.6, 0, 0, 0]

pgd = [94.8, 90.2, 84.6, 79, 73.4,
       66.0, 59.4, 53.2, 48.4, 43.4,
       39, 36.2, 33.6, 32.2, 30.2,
       28.6, 27.6, 26.4, 26.2, 25.8]

lg = [94.8, 90.2, 84.6, 79.6, 73.4,
      68.2, 60.8, 55.2, 49.2, 44.8,
      38.4, 35.2, 32.4, 28.4, 26.4,
      24.0, 22.2, 21, 19.6, 17.2]

ldg = [94.8, 90.2, 84.6, 79.8, 73.2,
       67.2, 60.0, 53.8, 48.0, 43.2,
       38.2, 35.0, 32.0, 28.4, 25.4,
       23.2, 21.0, 20.2, 18.2, 15.6]

ldg_xent = [95.0, 90.2, 84.6, 79.2, 73.4,
            67.2, 60.4, 55.0, 49.0, 44.0,
            38.6, 35.4, 31.6, 28.4, 27.0,
            25.0, 23.4, 21.4, 19.4, 18.6]

ldg_xent_v2 = [95.0, 90.2, 84.6, 79.2, 73.4,
               67.2, 60.4, 55.0, 48.6, 44.0,
               39.0, 35.4, 31.6, 28.4, 27.0,
               25.0, 23.4, 21.4, 19.4, 18.6]

avg_queries = 7500

ldg_NES = [94.8, 90.2, 84.6, 79.2, 73.6,
           67.8, 60.8, 55.2, 48.8, 44.2,
           38.6, 34.6, 31.2, 29.0, 26.6,
           23.4, 24.8, 20.4, 20.0, 20.4]

avg_queries = 5000

ldg_resize2cw = [95.4, 92.6, 89.8, 85.2, 81.0,
                 76.2, 71.4, 66.8, 61.6, 57.0,
                 53.2, 48.4, 44.0, 40.6, 37.8,
                 34.6, 32.0, 29.6, 27.0, 26.0]

ldg_resize2xent = [95.4, 92.6, 89.8, 85.4, 81.0,
                   76.4, 72.0, 67.2, 62.6, 57.0,
                   53.6, 49.4, 45.2, 41.4, 38.4,
                   34.8, 32.6, 31.0, 28.2, 27.6]

ldg_resize2xent_nn = [95.0, 91.4, 86.8, 82.2, 77.6,
                      71.6, 65.4, 59.2, 55.2, 50.0,
                      44.8, 41.0, 37.2, 33.2, 29.4,
                      28.6, 26.8, 23.8, 22.6, 20.0]

ldg_resize2xent_v2 = [95.4, 92.6, 89.8, 85.2, 81.0,
                      76.4, 71.8, 67.6, 62.6, 57.6,
                      53.4, 49.0, 45.4, 42.2, 38.8,
                      35.0, 32.0, 30.2, 28.0, 26.6]

avg_queries = 1850

lq_bbox_100 = [97.0, 96.0, 94.6, 92.6, 90.2,
               86.4, 84, 80, 76.4, 73.2,
               68.6, 62.8, 58.4, 55.2, 49.8,
               43.6, 40.2, 36.2, 32.4, 28.0]

avg_queries = 2100

lq_bbox_1500 = (68.2, 9424)
lq_bbox_1200 = (69.2, 7924)
lq_bbox_1000_1e6 = (69.2, 8909)
lq_bbox_1000 = (70.2, 6818)
lq_bbox_500 = (73.4, 4706)
lq_bbox_250 = (76.0, 3550)
lq_bbox_100 = (80, 2100)
lq_bbox_50 = (84.2, 840)

#Autozoom
auto_l2 = (73.8, 2656)
auto_l2_fix = (96.6, 2656)
auto_linf = (73.8, 1496)
auto_linf_fix = (96.2, 1496)

#500 block size 16 1e07 8 parallel
admm_ldg_par4_over4 = (52.8, 18092, 4638, 252.96)
admm_ldg_par4_over2 = (53.0, 15511, 4014, 162.10)
admm_ldg_par4_over1 = (53.2, 14326, 3720, 122.79)
admm_ldg_par4_over0 = (53.0, 13687, 3588, 59.83)
admm_ldg_par2_over1 = (53.0, 13047, 3857, 180.61)
admm_ldg_par1_over1 = (53.0, 12470, 4030, 300.37)

#500 block size 16 1e07 4 parallel xent
basic_ldg_par4_over0 = (52.4, 14930, 3965, 125)
basic_ldg_par4_over0_centered = (52.6, 14042, 3783, 17)

admm_ldg_par4_over0 = (53.2, 14011, 3711, 67)

admm_ldg_par1_over1 = (53.2, 13012, 4150, 274)

admm_ldg_par2_over1 = (53.2, 13599, 4028, 43)

admm_ldg_par4_over1_centered = (52.8, 15256, 4090, 145)

admm_ldg_par4_over1 = (53.2, 14800, 3727, 124, 33)
admm_ldg_par4_over2 = (53.2, 16225, 4214, 147)
admm_ldg_par4_over4 = (52.8, 18752, 4816, 202)
admm_ldg_par4_over8 = (53.0, 22591, 5695, 343, 58)

#500
admm_ldg_1e07_4 = (52.8, 60985)
admm_ldg_1e07_6 = (53.2, 64150)
admm_ldg_1e07_8 = (53.4, 60859)

#250
admm_ldg_1e05_4 = (54.0, 45758)
admm_ldg_1e06_4 = (50.8, 49337)
admm_ldg_1e07_2 = (48.8, 78421)
admm_ldg_1e07_4 = (48.8, 57933)
admm_ldg_1e07_6 = (49.2, 61545)
admm_ldg_1e07_8 = (49.6, 58234)
admm_ldg_1e08_2 = (48.8, 101430)
admm_ldg_1e08_4 = (48.8, 71493)
admm_ldg_1e08_6 = (48.8, 70760)
admm_ldg_1e08_8 = (48.8, 66172)
admm_ldg_1e09_4 = (48.8, 87523)
admm_ldg_1e09_6 = (48.8, 79861)
admm_ldg_1e09_8 = (48.8, 74783)
admm_ldg_1e09_10 = (48.8, 71733)
admm_ldg_1e09_12 = (48.8, 71067)
admm_ldg_1e09_14 = (48.8, 66188)
admm_ldg_1e09_16 = (48.8, 66039)

ldg_dec_2_05_05 = (56.4, 2466)
ldg_dec_4_05_05 = (60.4, 1149)
ldg_dec_8_05_05 = (70.8, 437)
ldg_dec_4_05_075 = (57.4, 1949)
ldg_dec_8_05_075 = (63.8, 1175)

# mask, min size 1
ldg_mask_100_10 = (71.6, 5029)
ldg_mask_200_20 = (69.6, 5493, 1855)
ldg_mask_300_30 = (69.0, 5723, 1915)
ldg_mask_500_50 = (67.0, 5772, 1984)
ldg_mask_500_200 = (62.4, 6238, 2358)
ldg_mask_1000_300 = (62.2, 6722, 2367)

#mask, min size 8?
ldg_mask_min8_top5 = (63.4, 5785) #blocksize = 2209
ldg_mask_4 = (63, 5619)
ldg_mask_3 = (64.4, 5496)
ldg_mask_2 = (68.2, 5137)

lrg = [94.8, 90.2, 84.6, 79.4, 73.2,
       67.6, 60, 53.8, 48.4, 43.6,
       38.6, 35.2, 32.2, 29.0, 26.2,
       23.4, 21.8, 20.6, 18.8, 16.8]

ldg_v2 = [94.8, 90.2, 84.6, 79.6, 73.2,
          67.2, 60.2, 53.8, 48.4, 43.2,
          38.6, 35.4, 32.2, 29.0, 25.6,
          24.0, 21.8, 20.4, 18.4, 16.6]

ratio_lg = [.59, .56, .54, .53, .53,
            .52, .50, .49, .48, .46,
            .45, .44, .42, .40, .39,
            .38, .36, .34, .34, .33]

submodular_xent = [69.9, 71.1, 71.4, 72.5,
                   73.3, 73.3, 73.7, 74.1]

submodular_xent_v2 = [76.0, 75.4, 75.1, 74.9,
                      75.0, 75.3, 75.4, 75.0]

submodular_cw = [68.0, 69.2, 70.3, 71.5,
                 72.7, 74.3, 75.5, 76.8]

# basic, no overlap
# block_size_2 = 2
adv_queries_2 = 17079
success_round_count_2 = [102, 18, 4, 1, 0]
convergence_stat_2 = [3032, 674, 445, 373, 350]
adv_accuracy_2 = 50.0

# block_size_4 = 4
avg_queries_4 = 18804
success_round_count_4 = [102, 19, 3, 2, 0]
convergence_stat_4 = [3034, 663, 427, 339, 314]
adv_accuracy_4 = 49.6

# block_size_8 = 8
avg_queries_8 = 18668
success_round_count_8 = [106, 18, 3, 1, 0]
convergence_stat_8 = [3033, 654, 401, 305, 276]
adv_accuracy_8 = 48.8

# block_size_16 = 16
avg_queries_16 = 16942
success_round_count_16 = [114, 10, 4, 0, 0]
convergence_stat_16 = [3032, 610, 372, 298, 269]
adv_accuracy_16 = 48.8

# block_size_32 = 32
avg_queries_32 = 19579
success_round_count_32 = [123, 0, 0, 0, 0]
congergence_stat_32 = [3037]
adv_accuracy_32 = 50.8


# basic, overlap
# block size 8, overlap 1
avg_queries_8_1 = 19152
success_round_count_8_1 = [114, 10, 4, 0, 0]
convergence_stat_8_1 = [2977, 623, 379, 299, 274]
adv_accuracy_8_1 = 48.8

# block size 8, overlap 2
avg_queries_8_2 = 21371
success_round_count_8_2 = [115, 10, 3, 0, 0]
convergence_stat_8_2 = [2931, 637, 387, 311, 286]
adv_accuracy_8_2 = 48.8

# block size 8, overlap 3
avg_queries_8_3 = 24036
success_round_count_8_3 = [116, 8, 4, 0, 0]
convergence_stat_8_3 = [2894, 648, 401, 324, 302]
adv_accuracy_8_3 = 48.8

# block size 8, overlap 4
avg_queries_8_4 = 26414
success_round_count_8_4 = [118, 8, 2, 0, 0]
convergence_stat_8_4 = [2870, 658, 415, 349, 322]
adv_accuracy_8_4 = 48.8

#-------------------------ImageNet---------------------------------#

#nat

ldg_resize4 = [0.1, 35843]
ldg_resize8 = [0.7, 9339]
ldg_resize16 = [5.8, 2408]
ldg_resize32 = [23.1, 637]
ldg_resize64 = [54.8, 131]
ldg_resize128 = [69.5, 69]

ldg_dec_8 = [0.2, 10080]
ldg_dec_8_025_05 = [0.2, 4919]

ldg_dec_16 = [0.2, 3807]
ldg_dec_16_025_025 = [0.2, 4059]
ldg_dec_16_05_025 = [1.5, 3111]
ldg_dec_16_5_01 = [3, 2855]

ldg_dec_32 = [1.3, 2683]
ldg_dec_32_reverse = [0.2, 2462]
ldg_dec_32_margin = [0.5, 2447]
ldg_dec_32_10000 = [5.8, 1474]
ldg_dec_32_10000_reverse = [4.3, 1494]
ldg_dec_32_10000_margin = [4.5, 1547]
ldg_dec_32_075_05_margin = [2.6, 2068]
ldg_dec_32_075_05_10000_reverse = [4.9, 1401] #[7.7, 1559]
ldg_dec_32_075_05_10000_margin = [7.0, 1384] #[7.7, 1559]
ldg_dec_32_075_025_margin = [12.8, 988] #[12.8, 1117]
ldg_dec_32_05_075_10000_reverse = [6.0, 1261]
ldg_dec_32_05_075_10000_margin = [6.6, 1406]
ldg_dec_32_05_025_reverse = [0.0, 3202]
ldg_dec_32_05_025_margin = [6.0, 1329] #[7.1, 1397]
ldg_dec_32_05_025 = [8.2, 1214]
ldg_dec_v2_32_05_025 = [6.4, 1204]
ldg_dec_32_05_025_10000_reverse = [7.3, 1359] #[6.6, 1473]
ldg_dec_32_05_025_10000_margin = [6.2, 1311] #[6.6, 1473]
ldg_dec_32_05_025_10000 = [8.2, 1205]
ldg_dec_v2_32_05_025_10000 = [6.0, 1206]
ldg_dec_32_05_02 = [10.8, 1038] #[10.1, 1209]
ldg_dec_v2_32_05_02 = [9.7, 1029]
ldg_dec_32_05_01_reverse = [0.1, 3261]
ldg_dec_32_05_01 = [14.8, 886]
ldg_dec_v2_32_05_01 = [14.8, 872]
ldg_dec_32_025 = [0.5, 3829]
ldg_dec_32_025_025_10000_reverse = [10.4, 1746]
ldg_dec_32_025_025 = [1.6, 2456]
ldg_dec_v2_32_025_025 = [3.9, 1337]
ldg_dec_32_025_025_10000 = [5.8, 1519]

ldg_dec_64 = [6.6, 2369]
ldg_dec_v2_64 = [2.7, 1673]
ldg_dec_64_reverse = [2.8, 2467]
ldg_dec_64_margin = [2.1, 2355]
ldg_dec_v2_64_10000 = [5.7, 1156]
ldg_dec_64_10000_reverse = [9.1, 1162]
ldg_dec_64_10000_margin = [8.8, 1209] #[11.4, 1136]
ldg_dec_v2_64_05_075 = [0.2, 2485]
ldg_dec_64_05_075_margin = [0.3, 5111]
ldg_dec_64_05_075 = [0.4, 5755]
ldg_dec_64_05_075_10000 = [14.1, 1006]
ldg_dec_64_v2_05_075_10000 = [5.6, 1243]
ldg_dec_64_05_075_10000_reverse = [23.9, 454]
ldg_dec_64_05_025_margin = [25.4, 428] #[25.5, 466]

ldg_dec_128 = [22.9, 1537]
ldg_dec_v2_128_05_1 = [0.1, 2247]
ldg_dec_v2_128_05_1_10000 = [5.7, 1103]
ldg_dec_v2_128_05_09 = [0.4, 2833]
ldg_dec_v2_128_05_075_10000 = [6.2, 1278]
ldg_dec_128_05_075 = [14.2, 1660] # [1.6, 8590]?
ldg_dec_v2_128_05_075 = [0.1, 3553]

#intersection
ldg_dec_v2_128_05_1_10000 = [5.4, 1178]
ldg_dec_v2_128_05_1_10000_new = [4.3, 988]

#bandit
bandit = [7.4, 1437]
bandit_benchmark = [4.9, 1141]
bandit_benchmark500 = [5.6, 1029]

bandit_tensorflow_torchprocessing = [7.6, 1412]
bandit_tensorflow_fd_eta = [11.6, 1361]
bandit_tensorflow2000_shuffle = [8.3, 1313]
bandit_tensorflow2500_shuffle = [7.3, 1449]
bandit_tensorflow1000all_shuffle = [7.1, 1486]
bandit_tensorflow2500all_shuffle = [7.6, 1419]

bandit_tensorflow1000all_shuffle_new1 = [5.6, 1355]
bandit_tensorflow1000all_shuffle_new2 = [7.5, 1440]
bandit_tensorflow1000test_new = [7.5, 1376]


bandit_tensorflow_newnorm10000 = [7.9, 1408]

bandit_pytorch1000 = [6.4, 1287]
bandit_pytorch2500_newnorm = [6.5, 1330]
bandit_pytorch2000_shuffle = [5.6, 1240]

# first pass
vanilla = [1.8, 686]
early_stop = [1.8, 639]

# rgb
mmp = 1062869
ppm = 1146255
pmp = 988835
mpm = 1009322
mpp = 1174452
pmm = 1162928
ppp = 1193967
mmm = 1201472

p = 13436104
m = 13384196

#adv

ldg_resize8 = [1.3, 9133]
ldg_resize16 = [4.9, 2327]
ldg_resize32 = [15.6, 637]
ldg_resize64 = [52.2, 130]
ldg_resize128 = [69.7, 68]

ldg_dec_8 = [0.4, 9928]
ldg_dec_16 = [0.5, 3562]
ldg_dec_32 = [1.7, 2267]
ldg_dec_64 = [6.2, 2025]
ldg_dec_128 = [22.3, 1312]

x = [(x+1) for x in range(20)]


plt.plot(x, lg)
plt.plot(x, ldg)
plt.plot(x, lrg)
plt.plot(x, ldg_v2)
plt.plot(x, pgd)
plt.legend(['lg', 'ldg', 'lrg', 'ldg_v2', 'pgd'])
plt.xlabel('eps')
plt.ylabel('accuracy')
plt.title('target: Resnet adv_trained w/ PGD. sample size: {}, loss: {}'.format(sample, loss))
plt.savefig('out/results_{}_{}.png'.format(sample, loss))
plt.close()

'''
plt.plot(x, ldg_xent)
plt.plot(x, ldg)
plt.plot(x, pgd)
plt.xlabel('eps')
plt.ylabel('accuracy')
plt.legend(['ldg_xent', 'ldg', 'pgd'])
plt.title('target: Resnet adv_trained w/ PGD. sample:size: {}'.format(sample, loss))
plt.savefig('out/blackbox comparison4.png')
plt.close()
'''

x_ = [1, 2, 3, 4, 5]
plt.plot(x_, success_round_count_8)
#plt.plot(x_, success_round_count_8_1)
plt.plot(x_, success_round_count_8_2)
#plt.plot(x_, success_round_count_8_3)
plt.plot(x_, success_round_count_8_4)
plt.xlabel('round')
plt.ylabel('# of successes')
plt.xticks(np.arange(min(x_), max(x_)+1, 1.0))
plt.legend(['no overlap', 'overlap 2', 'overlap 4'])
plt.title('eps=8, block_size=8, no admm, sample_size=250')
plt.savefig('out/overlap_comparison.png')
plt.close()

bench_mark = [3012, 492, 211, 152, 129, 121, 116, 101, 100, 66, 70, 91, 82, 83, 91, 80, 77, 100, 89, 74]
_1e_9__400 = [3012, 498, 184, 125, 73, 16, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
_1e_9__300 = [3012, 505, 205, 121, 90, 44, 27, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
_1e_9__200 = [3012, 504, 203, 144, 132, 81, 65, 65, 40, 27, 15, 7, 2, 0, 0, 0, 0, 0, 0, 0]
_1e_9__110 = [3012, 499, 203, 162, 135, 118, 111, 77, 84, 70, 83, 101, 87, 85, 78, 72, 69, 69, 99, 102]

plt.plot(x, bench_mark)
plt.plot(x, _1e_9__110)
plt.plot(x, _1e_9__200)
plt.plot(x, _1e_9__300)
plt.plot(x, _1e_9__400)
plt.ylim(0,600)
plt.xlabel('round')
plt.ylabel('# of changed pixels')
plt.legend(['no admm', 'tau=1.1', 'tau=2', 'tau=3', 'tau=4'])
plt.title('eps=8, block_size=8, admm, rho=1e-9, sample_size=1')
plt.savefig('out/admm_comparison.png')
plt.close()
