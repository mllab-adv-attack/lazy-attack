import argparse
import inspect
import pickle
import os
import sys
import tensorflow as tf
from PIL import Image

from tools.inception_v3_imagenet import Model
from tools.utils import *
from tools.imagenet_labels import label_to_name
import attacks

NUM_LABELS = 1000

""" The collection of all attack classes """
ATTACK_CLASSES = [
    x for x in attacks.__dict__.values()
    if inspect.isclass(x)
]
for attack in ATTACK_CLASSES:
    setattr(sys.modules[__name__], attack.__name__, attack)


""" Arguments """
parser = argparse.ArgumentParser()


def str2bool(key):
    return key.lower() in ('yes', 'true', 'y', 't')


# MODEL_DIR = '/Users/janghyun/Documents/Codes/Attack/Models/inception_v3.ckpt'
# DATA_DIR = '/Users/janghyun/Documents/Codes/Attack/Data_Test/imageNet'
# SAVE_DIR = '/Users/janghyun/Documents/Results/Attack/imageNet'
MODEL_DIR = '/home/gaon/lazy-attack/lls-imagenet/tools/data/inception_v3.ckpt'
DATA_DIR = '/home/gaon/lazy-attack/imagenet_data'
SAVE_DIR = '/home/gaon/lazy-attack/lls-both/lls-imagenet/results'

# Directory
parser.add_argument('--model_dir', default=MODEL_DIR, type=str)
parser.add_argument('--data_dir', default=DATA_DIR, type=str)
parser.add_argument('--save_dir', default=SAVE_DIR)
parser.add_argument('--name', default='imageNet', type=str)

# Experiment Setting
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=100, type=int)

parser.add_argument('--summary', default='True', type=str2bool)
parser.add_argument('--save_img', default='False', type=str2bool)

# Attack
parser.add_argument('--attack', default='LazyLocalSearchAttack', type=str)
parser.add_argument('--targeted', default='False', type=str2bool)
parser.add_argument('--loss_func', default='xent', type=str)
parser.add_argument('--epsilon', default=0.05, type=int)
parser.add_argument('--noise_size', default=256, help="noise size", type=int)
parser.add_argument('--image_range', default=1, help="image max value", type=float)
parser.add_argument('--max_queries', default=10000, type=int)

# Local Search Setting
parser.add_argument('--lls_iter', default=1, type=int)
parser.add_argument('--lls_block_size', default=32, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--no_hier', default='False', type=str2bool)

# Graph Cut Setting
parser.add_argument('--alpha', default=10000, type=int)
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--radius', default=1, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    # Set verbosity
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('=' * 100)
    tf.logging.info('Adversarial attack on ImageNet')

    # Print hyperparameters
    tf.logging.info('=' * 100)
    tf.logging.info('Configuration')
    for key, val in vars(args).items():
        tf.logging.info('{}={}'.format(key, val))

    # Load pretrained model
    tf.logging.info('=' * 100)
    tf.logging.info('Model')

    model = Model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, args.model_dir)

    # Create attack class
    tf.logging.info("=" * 100)
    tf.logging.info("Attack Method")
    tf.logging.info(args.attack)
    attack_class = getattr(sys.modules[__name__], args.attack)
    attack = attack_class(model, args)

    # Create directory
    tf.logging.info("=" * 100)
    tf.logging.info("Summary Directory")
    summary_dir = os.path.join(args.save_dir, args.name)
    tf.logging.info(summary_dir)
    if args.summary:
        os.makedirs(summary_dir, exist_ok=True)

    # Load indices
    if args.targeted:
        indices = np.load('assets/indices_targeted.npy')
    else:
        indices = np.load('assets/indices_untargeted.npy')

    # Main loop
    count = 0
    index = args.img_index_start

    total_num_corrects = 0
    total_queries = []
    index_to_query = {}

    while count < args.sample_size:
        tf.logging.info("=" * 100)

        # Get image and label.
        initial_img, orig_class = get_image(indices[index], args.data_dir)
        initial_img = np.expand_dims(initial_img, axis=0)

        # Generate target class (same method as in NES attack)
        if args.targeted:
            target_class = pseudorandom_target(indices[index], NUM_LABELS, orig_class)
            target_class = np.expand_dims(target_class, axis=0)

        orig_class = np.expand_dims(orig_class, axis=0)
        count += 1

        # Run attack
        if args.targeted:
            tf.logging.info('Targeted attack on {}th image starts, index: {}, orig class: {}, target class: {}'.format(
                count, indices[index], label_to_name(orig_class[0]), label_to_name(target_class[0])))
            adv_img, num_queries, success = attack.perturb(initial_img, target_class, indices[index], sess)
        else:
            tf.logging.info('Untargeted attack on {}th image starts, index: {}, orig class: {}'.format(
                count, indices[index], label_to_name(orig_class[0])))
            adv_img, num_queries, success = attack.perturb(initial_img, orig_class, indices[index], sess)

        # Check if the adversarial image satisfies the constraint
        assert np.amax(np.abs(adv_img - initial_img)) <= args.epsilon + 1e-3
        assert np.amax(adv_img) <= args.image_range + 1e-3
        assert np.amin(adv_img) >= 0. - 1e-3
        p = sess.run(model.predictions, feed_dict={model.x_input: adv_img})

        # Save Results
        if args.summary:
            with open(os.path.join(summary_dir, 'history_{}.p'.format(index)), 'wb') as f:
                pickle.dump(attack.history, f)

        if args.save_img:
            adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, :, :, :] * 255, np.uint8))
            adv_image.save(os.path.join(args.save_dir, '{}_adv.jpg'.format(indices[index])))

        # Logging
        if success:
            index_to_query[indices[index]] = num_queries

            total_num_corrects += 1
            total_queries.append(num_queries)
            average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
            median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
            success_rate = total_num_corrects / count

            tf.logging.info(
                'Attack successes, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
                    label_to_name(p[0]), average_queries, median_queries, success_rate))
        else:
            index_to_query[indices[index]] = -1

            average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
            median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
            success_rate = total_num_corrects / count

            tf.logging.info(
                'Attack fails, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}'.format(
                    label_to_name(p[0]), average_queries, median_queries, success_rate))

        index += 1

    # np.save('outputs/loss_func_{}_epsilon_{}_img_index_start_{}.npy'.format(args.loss_func, args.epsilon, args.img_index_start), index_to_query)
