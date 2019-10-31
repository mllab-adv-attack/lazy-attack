import numpy as np

LOAD_DATA_DIR = '/data/home/gaon/lazy-attack/cifar10_data/'
CIFAR10_TRAIN_DATA_SIZE = 50000
CIFAR10_EVAL_DATA_SIZE = 10000
FILE_BATCH_SIZE = 1000


def imp_file_name(args):
    meta_name = 'nat' if args.model_dir == 'naturally_trained' else 'adv'
    meta_name += '_pgd' + '_' + str(args.pgd_eps) \
        + '_' + str(args.pgd_num_steps) \
        + '_' + str(args.pgd_step_size) \
        + ('_rand' if args.pgd_random_start else '') \
        + '_' + str(args.pgd_restarts)
    meta_name += '_imp' + ('_' + str(args.imp_delta)) \
        + ('_' + str(args.imp_num_steps)) \
        + ('_' + str(args.imp_step_size)) \
        + ('_' + str(args.imp_random_start)) \
        + ('_' + str(args.imp_random_seed)) \
        + ('_' + str(args.imp_pp))
    meta_name += ('_infer' if args.label_infer else '')
    meta_name += ('_rep' if args.imp_rep else '')
    meta_name += ('_adam' if args.imp_adam else '') + ('_rms' if args.imp_rms else '') + ('_adag' if args.imp_adagrad else '')
    meta_name += ('_nosign' if args.imp_no_sign else '')
    meta_name += ('_corr' if args.corr_only else '') + ('_fail' if args.fail_only else '')
    meta_name += '_val' + ('_' + str(args.val_step_per)) \
        + ('_' + str(args.val_eps)) \
        + ('_' + str(args.val_num_steps)) \
        + ('_' + str(args.val_restarts))
    return meta_name


def infer_file_name(args):
    meta_name = 'nat' if args.model_dir == 'naturally_trained' else 'adv'
    meta_name += '_lr' + str(args.g_lr)
    meta_name += '_delta' + str(args.delta)
    meta_name += '_pgd' + '_' + str(args.eps) \
        + '_' + str(args.num_steps) \
        + '_' + str(args.step_size)
    meta_name += '_fdim' + str(args.f_dim)
    meta_name += '_nblk' + str(args.n_blocks)
    meta_name += '_ndwn' + str(args.n_down)
    meta_name += '_unet' if args.use_unet else ''
    meta_name += ('_drop' + str(args.dropout_rate)) if args.dropout else ''
    meta_name += '_nolc' if args.no_lc else ''
    meta_name += '_noise' if args.noise_only else ''
    if args.use_advG:
        meta_name += '_advGlr' + str(args.advG_lr)
    if args.use_d:
        meta_name += '_dlr' + str(args.d_lr)
        meta_name += '_g' + str(args.g_weight)
        meta_name += '_d' + str(args.d_weight)
        meta_name += '_patch' if args.patch else ''
    meta_name += ('_lo' + str(args.l1_weight)) if args.l1_loss else ''
    meta_name += ('_lt' + str(args.l2_weight)) if args.l2_loss else ''
    meta_name += ('_lp' + str(args.lp_weight)) if args.lp_loss else ''
    return meta_name


def load_imp_data(args, eval=False):
    final_dir = 'imp_nat_fixed/' if args.model_dir == 'naturally_trained' else 'imp_adv_fixed/'

    data_dir = LOAD_DATA_DIR + final_dir

    posfix_li = [('imp_' + ('eval' if eval else 'train') + '_fixed_{:.1f}_'.format(args.delta)+str(idx))
                 for idx in range(0, CIFAR10_EVAL_DATA_SIZE if eval else CIFAR10_TRAIN_DATA_SIZE, FILE_BATCH_SIZE)]
    filename_li = [(str_idx + '_' + str(FILE_BATCH_SIZE) + '.npy') for str_idx in posfix_li]
    fullname_li = [(data_dir + filename) for filename in filename_li]
    data_li = [np.load(fullname) for fullname in fullname_li]
    data = np.concatenate(data_li)
    return data
