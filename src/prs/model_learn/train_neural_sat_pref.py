from argparse import ArgumentParser
import time
import torch
import random
import numpy as np
import sys
from prs.model_learn.evaluation import evaluation, compute_nabla_log_ZC
from prs.utils.sat_instance import SAT_Instance, generate_random_solutions_with_preference
from prs.model_learn.draw_from_all_samplers import draw_from_waps, draw_from_weightgen, draw_from_cmsgen, draw_from_unigen, \
    draw_from_prs_series, draw_from_kus, draw_from_quicksampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(10010)
random.seed(10010)
np.random.seed(10010)


class EnergyScore(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.beta = 1
        init_val = torch.randn(input_dim)
        self.theta = torch.nn.Parameter(init_val / torch.linalg.norm(init_val))

    def get_prob(self):
        with torch.no_grad():
            return torch.sigmoid(2 * self.theta - 1)

    def forward(self, x):
        pref_score = torch.sum(torch.mul(self.theta, x) + torch.mul(1 - self.theta, 1 - x), dim=1)
        return pref_score


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input_file', help='read from given file', type=str)
    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')

    parser.add_argument('--epochs', help='The maximum learning iterations', default=100, type=int)
    parser.add_argument('--eval_logZ_freq', help='evaluation frequency', default=20, type=int)
    parser.add_argument('--learning_rate', help="learning rate of SGD", default=1e-3, type=float)

    parser.add_argument('--num_samples', help='number of samples for the sampler', type=int, default=2000)
    parser.add_argument('--file_type', help='cnf', type=str, default='cnf')
    parser.add_argument('--evaluate_freq', help='evaluation frequency', type=int, default=2)
    parser.add_argument('--K', help='cnf', type=int, default=3)
    return parser.parse_args()


def get_dataset(input_file, K=5, device='cuda'):
    instance = SAT_Instance.from_cnf_file(input_file, K)
    clause2var, weight, bias = instance.get_formular_matrix_form()
    clause2var, weight, bias = torch.from_numpy(clause2var).int().to(device), \
                               torch.from_numpy(weight).int().to(device), \
                               torch.from_numpy(bias).int().to(device)

    preferred_xis, less_preferred_xis = generate_random_solutions_with_preference(instance)
    return clause2var, weight, bias, preferred_xis, less_preferred_xis, instance


def learn_sat_pref(args):
    start = time.time()
    clause2var, weight, bias, preferred_xis, less_preferred_xis, Fi = get_dataset(args.input_file, args.K)

    time_used_for_sampler = []
    time_used_for_nn = []
    model = EnergyScore(input_dim=Fi.cnf.nv).to(device)
    step_idx = 0
    for epo_idx in range(1, args.epochs):
        random.shuffle(preferred_xis)
        for xi in preferred_xis:
            step_idx += 1
            prob = model.get_prob()
            st = time.time()
            samples = [torch.from_numpy(xi).float().to(device).reshape(1, -1), ]
            if args.algo == 'quicksampler':
                samples += draw_from_quicksampler(args.num_samples, args.input_file)
            elif args.algo == 'unigen':
                samples += draw_from_unigen(args.num_samples, args.input_file)
            elif args.algo == 'cmsgen':
                samples += draw_from_cmsgen(args.num_samples, args.input_file)
            elif args.algo == 'kus':
                samples += draw_from_kus(args.num_samples, args.input_file)
            elif args.algo == 'waps':
                samples += draw_from_waps(args.num_samples, args.input_file, Fi, prob)
            elif args.algo == 'weightgen':
                samples += draw_from_weightgen(args.num_samples, args.input_file, Fi, prob)
            elif args.algo in ['lll', 'mc', 'prs', 'nelson', 'nelson_batch']:
                samples += draw_from_prs_series(args.algo, Fi, clause2var, weight, bias, prob, args.num_samples)
                samples = [x.reshape(1, -1) for x in samples]

            time_used_for_sampler.append((time.time() - st) * 1000)
            st = time.time()
            new_inputs = torch.concat(samples, dim=0).to(device)
            phi = model(new_inputs)
            loss_bar = phi[0] - torch.sum(phi[1:]) / (len(phi) - 1)
            print("energy: {:.6f} {:.6f} diff: {:.6f}".format(phi[0], torch.sum(phi[1:]) / (len(phi) - 1), loss_bar))
            sys.stdout.flush()
            model.zero_grad()
            loss_bar.backward()
            with torch.no_grad():
                for param in model.parameters():
                    # SGD
                    param -= args.learning_rate * param.grad
                    # regularize the L2 norm of parameter:
                    param /= torch.linalg.norm(param)

            time_used_for_nn.append((time.time() - st) * 1000)

            if step_idx % args.eval_logZ_freq == 0:
                compute_nabla_log_ZC(Fi, model, args.num_samples, args.input_file, prob=prob)

        if epo_idx % args.evaluate_freq == 0:
            evaluation(Fi, clause2var, weight, bias, args.algo, model, preferred_xis, less_preferred_xis, input_file=args.input_file,
                       num_samples=args.num_samples)
            time_used_for_sampler = np.asarray(time_used_for_sampler)
            time_used_for_nn = np.asarray(time_used_for_nn)
            print("{} MRF: {:.3f} {:.3f}, sampler: {:.3f} {:.3f} ms".format(args.algo, np.mean(time_used_for_nn), np.std(time_used_for_nn),
                                                                            np.mean(time_used_for_sampler),
                                                                            np.std(time_used_for_sampler)))
            print("training time: {}".format(time.time() - start))
            exit()
    evaluation(Fi, clause2var, weight, bias, args.algo, model, preferred_xis, less_preferred_xis, input_file=args.input_file,
               num_samples=args.num_samples)
    time_used_for_sampler = np.asarray(time_used_for_sampler)
    time_used_for_nn = np.asarray(time_used_for_nn)
    print("time for NN: {:.3f} {:.3f}, sampler: {:.3f} {:.3f} ms".format(np.mean(time_used_for_nn), np.std(time_used_for_nn),
                                                                         np.mean(time_used_for_sampler), np.std(time_used_for_sampler)))


if __name__ == "__main__":
    args = get_arguments()
    learn_sat_pref(args)
