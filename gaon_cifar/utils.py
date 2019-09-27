import tensorflow as tf

def get_shape(tensor):
    return tensor.get_shape().as_list()

def batch_norm(*args, **kwargs):
    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)
    return bn

def lkrelu(x, slope=0.01):
    return tf.maximum(slope * x, x)

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
    meta_name += '_lr' + str(args.lr)
    meta_name += '_delta' + str(args.delta)
    meta_name += '_pgd' + '_' + str(args.eps) \
        + '_' + str(args.num_steps) \
        + '_' + str(args.step_size)
    return meta_name
