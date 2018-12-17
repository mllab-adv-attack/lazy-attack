import torch as ch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules import Upsample
import argparse
import json

#added
import math
import numpy as np
from tools.utils import get_image

IMAGENET_PATH = './../../data/'
#IMAGENET_PATH = './../../torch_data/'

ch.set_default_tensor_type('torch.cuda.FloatTensor')
IMAGENET_SL = 299

model_to_fool = models.inception_v3(pretrained=True).cuda()
model_to_fool = DataParallel(model_to_fool)

model_to_fool.eval()

# added
imagenet = ImageFolder(IMAGENET_PATH,
                        transforms.Compose([
                            transforms.Resize(IMAGENET_SL),
                            transforms.CenterCrop(IMAGENET_SL),
                            transforms.ToTensor(),
                        ]))

def norm(t):
    """
    Takes the norm, treating an n-dimensional tensor as a batch of vectors:
    If x has shape (a, b, c, d), we flatten b, c, d, return the norm along axis 1.
    """
    norm_vec = ch.sqrt(t.pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec

###
# Different optimization steps
# All take the form of func(x, g, lr)
# eg: exponentiated gradients
# l2/linf: projected gradient descent
###

def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*ch.exp(lr*g)
    neg = (1-real_x)*ch.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1

def linf_step(x, g, lr):
    return x + lr*ch.sign(g)

def gd_prior_step(x, g, lr):
    return x + lr*g
   
def l2_image_step(x, g, lr):
    return x + lr*g/norm(g)

##
# Projection steps for l2 and linf constraints:
# All take the form of func(new_x, old_x, epsilon)
##

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        return orig + ch.clamp(new_x - orig, -eps, eps)
    return proj

def batch_norm(image):
    '''new_image = image.clone()
    new_image[:, 0, :, :] = (new_image[:, 0, :, :]-0.485)/0.229
    new_image[:, 1, :, :] = (new_image[:, 1, :, :]-0.456)/0.224
    new_image[:, 2, :, :] = (new_image[:, 2, :, :]-0.406)/0.225
    return new_image
    '''
    return (image-ch.FloatTensor([0.485, 0.456, 0.406]).view(1,-1,1,1).cuda())/(ch.FloatTensor([0.229, 0.224, 0.225]).view(1,-1,1,1).cuda())

##
# Main functions
##

def make_adversarial_examples(image, true_label, args):
    '''
    The main process for generating adversarial examples with priors.
    '''
    # Initial setup
    prior_size = IMAGENET_SL if not args.tiling else args.tile_size
    print("prior size:", prior_size)
    upsampler = Upsample(size=(IMAGENET_SL, IMAGENET_SL))
    total_queries = ch.zeros(args.batch_size)
    prior = ch.zeros(args.batch_size, 3, prior_size, prior_size)
    dim = prior.nelement()/args.batch_size
    prior_step = gd_prior_step if args.mode == 'l2' else eg_step
    image_step = l2_image_step if args.mode == 'l2' else linf_step
    proj_maker = l2_proj if args.mode == 'l2' else linf_proj
    proj_step = proj_maker(image, args.epsilon)

    # Loss function
    criterion = ch.nn.CrossEntropyLoss(reduction='none')
    L =  lambda x: criterion(model_to_fool(batch_norm(x)), true_label)
    losses = L(image)

    # Original classifications
    orig_images = image.clone()
    orig_classes = model_to_fool(batch_norm(image)).argmax(1).cuda()
    correct_classified_mask = (orig_classes == true_label).float()
    total_ims = correct_classified_mask.sum()
    not_dones_mask = correct_classified_mask.clone()

    while not ch.any(total_queries > args.max_queries):
        if not args.nes:
            ## Updating the prior: 
            # Create noise for exporation, estimate the gradient, and take a PGD step
            exp_noise = args.exploration*ch.randn_like(prior)/(dim**0.5) 
            # Query deltas for finite difference estimator
            q1 = upsampler(prior + exp_noise)
            q2 = upsampler(prior - exp_noise)
            # Loss points for finite difference estimator
            l1 = L(image + args.fd_eta*q1/norm(q1)) # L(prior + c*noise)
            l2 = L(image + args.fd_eta*q2/norm(q2)) # L(prior - c*noise)
            # Finite differences estimate of directional derivative
            est_deriv = (l1 - l2)/(args.fd_eta*args.exploration)
            # 2-query gradient estimate
            est_grad = est_deriv.view(-1, 1, 1, 1)*exp_noise
            # Update the prior with the estimated gradient
            prior = prior_step(prior, est_grad, args.online_lr)
        else:
            prior = ch.zeros_like(image)
            for _ in range(args.gradient_iters):
                exp_noise = ch.randn_like(image)/(dim**0.5) 
                est_deriv = (L(image + args.fd_eta*exp_noise) - L(image - args.fd_eta*exp_noise))/args.fd_eta
                prior += est_deriv.view(-1, 1, 1, 1)*exp_noise

        # Preserve images that are already done
        prior = prior*not_dones_mask.view(-1, 1, 1, 1)

        ## Update the image:
        # take a pgd step using the prior
        new_im = image_step(image, upsampler(prior), args.image_lr)
        image = proj_step(new_im)
        image = ch.clamp(image, 0, 1)
        if args.mode == 'l2':
            if not ch.all(norm(image - orig_images) <= args.epsilon + 1e-3):
                raise ValueError("OOB")
        else:
            if not (image - orig_images).max() <= args.epsilon + 1e-3:
                raise ValueError("OOB")

        ## Continue query count
        not_dones_mask = not_dones_mask*((model_to_fool(batch_norm(image)).argmax(1) == true_label).float())
        total_queries += 2*args.gradient_iters*not_dones_mask

        ## Logging stuff
        new_losses = L(image)
        success_mask = (1 - not_dones_mask)*correct_classified_mask
        num_success = success_mask.sum()
        current_success_rate = (num_success/correct_classified_mask.sum()).cpu().item()
        success_queries = ((success_mask*total_queries).sum()/num_success).cpu().item()
        not_done_loss = ((new_losses*not_dones_mask).sum()/not_dones_mask.sum()).cpu().item()
        max_curr_queries = total_queries.max().cpu().item()
        if args.log_progress and max_curr_queries%100==0:
            print("Queries: %d | Success rate: %f | Average queries: %f" % (max_curr_queries, current_success_rate, success_queries))
        if current_success_rate == 1.0:
            break

    # Return results
    return {
            'average_queries': success_queries, # Average queries for this batch
            'num_correctly_classified': correct_classified_mask.sum().cpu().item(), # Number of originally correctly classified images
            'success_rate': current_success_rate, # Success rate
            'images_orig': orig_images.cpu().numpy(), # Original images
            'images_adv': image.cpu().numpy(), # Adversarial images
            'all_queries': total_queries.cpu().numpy(), # Number of queries used for each image
            'correctly_classified': correct_classified_mask.cpu().numpy(), # 0/1 mask for whether image was originally classified
            'success': success_mask.cpu().numpy() # 0/1 mask for whether the attack succeeds on each image
    }

def main(args):
    
    imagenet_loader = DataLoader(imagenet, batch_size=args.batch_size,
                                 shuffle=False)
    
    num_eval_examples = args.sample_size
    eval_batch_size = min(args.batch_size, num_eval_examples)

    if args.test:
        target_indices = np.load('./out/intersection_norm.npy')
    else:
        target_indices = [i for i in range(50000)]
    
    if args.shuffle:
        np.random.shuffle(target_indices)
    
    num_batches = int(math.ceil(num_eval_examples/eval_batch_size))

    assert (num_eval_examples%eval_batch_size==0)
    bstart = 0
    x_full_batch = []
    y_full_batch = []

    attack_set = []
    
    
    print('loading image data')
    while(True):
        x_candid = []
        y_candid = []
        for i in range(100):
            img_batch, y_batch = get_image(target_indices[bstart+i], IMAGENET_PATH)
            img_batch = ch.Tensor(img_batch)
            img_batch = ch.transpose(img_batch, 0, 1)
            img_batch = ch.transpose(img_batch, 0, 2)
            x_candid.append(img_batch.view(-1, 3, 299, 299))
            y_candid.append(y_batch)
        x_candid = ch.cat(x_candid)
        y_candid = np.array(y_candid)
        preds = model_to_fool(batch_norm(x_candid).cuda()).argmax(1).cpu().numpy()
        idx = np.where(preds == y_candid)
        for i in idx[0]:
            attack_set.append(bstart+i)
        x_candid = x_candid.cpu().numpy()
        x_masked = x_candid[idx]
        y_masked = y_candid[idx]
        if bstart == 0:
            x_full_batch = x_masked[:min(num_eval_examples, len(x_masked))]
            y_full_batch = y_masked[:min(num_eval_examples, len(y_masked))]
        else:
            index = min(num_eval_examples-len(x_full_batch), len(x_masked))
            x_full_batch = np.concatenate((x_full_batch, x_masked[:index]))
            y_full_batch = np.concatenate((y_full_batch, y_masked[:index]))
        bstart += 100
        print(len(attack_set), bstart, len(attack_set)/bstart)
        if len(x_full_batch) >= num_eval_examples or (bstart==50000):
            break
    #np.save('./out/pytorch_{}.npy'.format(args.sample_size), attack_set)
    average_queries_per_success = 0.0
    total_correctly_classified_ims = 0.0
    success_rate_total = 0.0
    total_success_ims = 0.0

    '''
    for i, (images, targets) in enumerate(imagenet_loader):
        
        if i*args.batch_size >= args.sample_size:
            return average_queries_per_success/total_correctly_classified_ims, \
                    success_rate_total/total_correctly_classified_ims
        res = make_adversarial_examples(images.cuda(), targets.cuda(), args)
        # The results can be analyzed here!
        average_queries_per_success += res['success_rate']*res['average_queries']*res['num_correctly_classified']
        success_rate_total += res['success_rate']*res['num_correctly_classified']
        total_correctly_classified_ims += res['num_correctly_classified']
        total_success_ims += res['success_rate']*res['num_correctly_classified']
    '''

    print("Iterating over {} batches\n".format(num_batches))
    for ibatch in range(num_batches):
        print('attacking {}th batch...'.format(ibatch))
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)
        images = x_full_batch[bstart:bend, :]
        targets = y_full_batch[bstart:bend]
        res = make_adversarial_examples(ch.Tensor(images).cuda(), ch.Tensor(targets).long().cuda(), args)
        average_queries_per_success += res['success_rate']*res['average_queries']*res['num_correctly_classified']
        success_rate_total += res['success_rate']*res['num_correctly_classified']
        total_correctly_classified_ims += res['num_correctly_classified']
        total_success_ims += res['success_rate']*res['num_correctly_classified']
    

    return average_queries_per_success/total_success_ims, \
        success_rate_total/total_correctly_classified_ims, \
        total_success_ims

class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        self.params = params
    
    def __getattr__(self, x):
        return self.params[x.lower()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-queries', default=10000, type=int)
    parser.add_argument('--fd-eta', default=0.1, type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', default=0.005, type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--online-lr', default=100, type=float, help='Learning rate for the prior')
    parser.add_argument('--mode', default='linf', type=str, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--exploration', default=0.01, type=float, help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument('--tile-size', default=50, type=int, help='the side length of each tile (for the tiling prior)')
    parser.add_argument('--json-config', type=str, help='a config file to be passed in instead of arguments')
    parser.add_argument('--epsilon', default=0.05, type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', default=500, type=int, help='batch size for bandits')
    parser.add_argument('--sample-size', default=1000, type=int, help='sample size for bandits')
    parser.add_argument('--log-progress', action='store_false')
    parser.add_argument('--nes', action='store_true')
    parser.add_argument('--tiling', action='store_false')
    parser.add_argument('--gradient-iters', default=1, type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        if not args.nes:
            assert not any([x is None for x in [args.fd_eta, args.max_queries, args.image_lr, \
                            args.mode, args.exploration, args.batch_size, args.epsilon]])
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = Parameters(defaults)
        args_dict = defaults

    with ch.no_grad():
        print("Queries, Success, orig_correct = ", main(args))

