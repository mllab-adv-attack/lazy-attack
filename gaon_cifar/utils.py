import numpy as np

LOAD_DATA_DIR = '/data/home/gaon/lazy-attack/cifar10_data/'
CIFAR10_TRAIN_DATA_SIZE = 50000
FILE_BATCH_SIZE = 1000


def imp_file_name(args):
    meta_name = 'nat' if args.model_dir == 'naturally_trained' else 'adv'
    meta_name += '_pgd' + '_' + str(args.pgd_eps) \
        + '_' + str(args.pgd_num_steps) \
        + '_' + str(args.pgd_step_size) \
        + ('_rand' if args.pgd_random_start else '') \
        + '_' + str(args.pgd_restarts)
    meta_name += '_imp' + ('_' + str(args.imp_eps)) \
        + ('_' + str(args.imp_num_steps)) \
        + ('_' + str(args.imp_step_size)) \
        + ('_' + str(args.imp_random_start)) \
        + ('_' + str(args.imp_random_seed)) \
        + ('_' + str(args.imp_pp))
    meta_name += ('_rep' if args.imp_rep else '')
    meta_name += ('_adam' if args.imp_adam else '') + ('_nosign' if args.imp_no_sign else '')
    meta_name += ('_corr' if args.corr_only else '') + ('_fail' if args.fail_only else '')
    meta_name += '_val' + ('_' + str(args.val_step_per)) \
        + ('_' + str(args.val_eps)) \
        + ('_' + str(args.val_num_steps)) \
        + ('_' + str(args.val_restarts))
    meta_name += '_' + str(args.soft_label)
    return meta_name


def infer_file_name(args):
    meta_name = 'nat' if args.model_dir == 'naturally_trained' else 'adv'
    meta_name += '_lr' + str(args.g_lr)
    meta_name += '_delta' + str(args.delta)
    meta_name += '_pgd' + '_' + str(args.eps) \
        + '_' + str(args.num_steps) \
        + '_' + str(args.step_size)
    if args.use_d:
        meta_name += '_dlr' + str(args.d_lr)
        meta_name += '_w' + str(args.gan_weight)
    return meta_name


def load_imp_data(args):
    final_dir = 'imp_nat_fixed/' if args.model_dir == 'naturally_trained' else 'imp_adv_fixed/'

    data_dir = LOAD_DATA_DIR + final_dir

    posfix_li = [('imp_train_fixed_'+str(idx)) for idx in range(0, CIFAR10_TRAIN_DATA_SIZE, FILE_BATCH_SIZE)]
    filename_li = [(str_idx + '_' + str(FILE_BATCH_SIZE) + '.npy') for str_idx in posfix_li]
    fullname_li = [(data_dir + filename) for filename in filename_li]
    data_li = [np.load(fullname) for fullname in fullname_li]
    data = np.concatenate(data_li)
    return data
