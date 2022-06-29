from argparse import ArgumentParser

import torch

from sat_instance import SAT_Instance, generate_random_solutions_with_preference
from sampler import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time


class DeepEnergyNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.beta = 1
        init_val = torch.randn(input_dim)
        self.theta = torch.nn.Parameter(init_val / torch.linalg.norm(init_val))

    def get_prob(self):
        return torch.sigmoid(2 * self.theta - 1)

    def forward(self, x):
        pref_score = torch.sum(torch.mul(self.theta, x) + torch.mul(1 - self.theta, 1 - x), dim=1)
        return pref_score


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input_file', help='read from given file', type=str)
    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')
    parser.add_argument('--epochs', help='The maximum learning iterations', default=100, type=int)
    parser.add_argument('--learning_rate', help="learning rate of SGD", default=1e-3, type=float)
    parser.add_argument('--num_samples', help='number of samples for the neural Lovasz sampler', type=int, default=10)
    parser.add_argument('--file_type', help='cnf', type=str, default='cnf')
    parser.add_argument('--evaluate_freq', help='evaluation frequency', type=int, default=10)
    parser.add_argument('--K', help='cnf', type=int, default=3)
    return parser.parse_args()


def test(formula, sampler, neural_net, preferred_inputs, less_preferred_inputs,
         list_of_Ks=[5, 10, 20]):
    # compute log-likelihood
    # see Handbook of Satisfiability Chapter 20
    # URL: https://www.cs.cornell.edu/~sabhar/chapters/ModelCounting-SAT-Handbook-prelim.pdf Page 11
    samples = [torch.from_numpy(preferred_inputs[0]).float().to(device).reshape(1, -1), ]

    prob = neural_net.get_prob()
    while len(samples) <= args.num_samples:
        assignment, _, time_used = sampler(formula, device, prob)

        if len(assignment) > 0:
            samples.append(assignment.reshape(1, -1))

    new_inputs = torch.concat(samples, dim=0).to(device)
    phi = neural_net(new_inputs)
    pseudo_loss = phi[0] - torch.sum(phi[1:]) / (len(phi) - 1)
    uni_samples = []
    while len(uni_samples) <= args.num_samples:

        assignment, _, time_used = sampler(formula, device, 0.5)
        if len(assignment) > 0:
            uni_samples.append(assignment.reshape(1, -1))
    averaged_ratio = args.num_samples / torch.sum(torch.concat(uni_samples, dim=0).to(device), dim=0)
    log_model_count = torch.sum(torch.log(averaged_ratio))
    print("log-likelihood: {:.4f}".format(- pseudo_loss - log_model_count))

    # compute mean averaged precision (MAP@K)
    test_samples = []
    for xi in preferred_inputs:
        test_samples.append(torch.from_numpy(xi).float().to(device).reshape(1, -1))
    for xi in less_preferred_inputs:
        test_samples.append(torch.from_numpy(xi).float().to(device).reshape(1, -1))
    new_inputs = torch.concat(test_samples, dim=0).to(device)
    phi = neural_net(new_inputs)
    sorted_idxes = torch.argsort(phi)
    print("MAP@{}".format(list_of_Ks), end=": ")
    for k in list_of_Ks:
        k = min(k, len(sorted_idxes))
        cnt = 0
        map = 0
        for idx, i in enumerate(sorted_idxes[:k]):
            if i < len(preferred_inputs):
                cnt += 1
                map += cnt / (idx + 1)
        print("{:.4f}".format(map / cnt), end=" ")
    print()


def learn_sat_pref(args):
    # Construct our model by instantiating the class defined above
    instances = []
    if args.file_type == 'cnf':
        instances.append(SAT_Instance.from_cnf_file(args.input_file, args.K))
    else:
        instances.append(SAT_Instance.from_file(args.input_file, K=3))

    sat_train_data = generate_random_solutions_with_preference(instances)

    if args.algo == 'lll':
        sampler = constructive_lovasz_local_lemma_sampler
    elif args.algo == 'mc':
        sampler = Monte_Carlo_sampler
    elif args.algo == 'prs':
        sampler = conditioned_partial_rejection_sampling_sampler
    elif args.algo == 'nls':
        sampler = pytorch_conditioned_partial_rejection_sampling_sampler

    time_used_for_nelson = []
    time_used_for_nn = []
    for preferred_xis, less_preferred_xis, Fi in sat_train_data:
        model = DeepEnergyNet(input_dim=Fi.cnf.nv).to(device)
        for epo_idx in range(1, args.epochs):
            random.shuffle(preferred_xis)
            for xi in preferred_xis:
                samples = [torch.from_numpy(xi).float().to(device).reshape(1, -1), ]

                prob = model.get_prob()
                while len(samples) <= args.num_samples:
                    assignment, _, time_used = sampler(Fi, device, prob)

                    if len(assignment) > 0:
                        samples.append(assignment.reshape(1, -1))
                        time_used_for_nelson.append(time_used)
                st = time.time()
                new_inputs = torch.concat(samples, dim=0).to(device)
                phi = model(new_inputs)
                pseudo_loss = phi[0] - torch.sum(phi[1:]) / (len(phi) - 1)
                print("energy: {:.4f} {:.4f} diff: {:.4f}".format(phi[0], torch.sum(phi[1:]) / (len(phi) - 1), pseudo_loss))

                model.zero_grad()
                pseudo_loss.backward()
                with torch.no_grad():
                    for param in model.parameters():
                        # SGD
                        param -= args.learning_rate * param.grad
                        # regularize the L2 norm of parameter:
                        param /= torch.linalg.norm(param)

                time_used_for_nn.append((time.time() - st) * 1000)

            if epo_idx % args.evaluate_freq == 0:
                test(Fi, sampler, model, preferred_xis, less_preferred_xis)

        # time_used_for_nelson = np.asarray(time_used_for_nelson)
        # time_used_for_nn = np.asarray(time_used_for_nn)
        # print("time for NN: {:.3f} {:.3f}, Nelson: {:.3f} {:.3f} ms".format(np.mean(time_used_for_nn), np.std(time_used_for_nn),
        #                                                                     np.mean(time_used_for_nelson), np.std(time_used_for_nelson)))


if __name__ == "__main__":
    args = get_arguments()
    learn_sat_pref(args)
