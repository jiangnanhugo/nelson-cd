from argparse import ArgumentParser

from sat_instance import SAT_Instance, generate_random_solutions_with_preference
from sampler import *


class DeepEnergyNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.beta = 1
        self.theta = torch.nn.Parameter(torch.randn(input_dim))

    def get_prob(self):
        return torch.sigmoid(self.beta * (2 * self.theta - 1))

    def forward(self, x):
        print("theta={}, x={}".format(self.theta.shape, x.shape))
        pref_score = torch.sum(torch.mul(self.theta, x) + torch.mul(1 - self.theta, 1 - x), dim=1)
        print("pref_score={}".format(pref_score.shape))

        return torch.exp(-self.beta * pref_score)


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input_file', help='read from given file', type=str)
    parser.add_argument('--algo', type=str, help="use which sampler", default='lll')
    parser.add_argument('--t_max', help='The maximum learning iterations', default=100, type=int)
    parser.add_argument('--learning_rate', help="learning rate of SGD", default=1e-4, type=float)
    parser.add_argument('--num_samples', help='number of samples for the neural Lovasz sampler', type=int, default=10)
    return parser.parse_args()


def learn_sat_pref(args):
    # Construct our model by instantiating the class defined above
    instances = []
    with open(args.input_file, 'r') as file:
        instances.append(SAT_Instance.from_file(file, K=3))

    sat_train_data = generate_random_solutions_with_preference(instances)

    if args.algo == 'lll':
        sampler = constructive_lovasz_local_lemma_sampler
    elif args.algo == 'mc':
        sampler = Monte_Carlo_sampler
    elif args.algo == 'prs':
        sampler = conditioned_partial_rejection_sampling_sampler
    elif args.algo == 'nls':
        sampler = pytorch_conditioned_partial_rejection_sampling_sampler

    for xi, Fi in sat_train_data:
        model = DeepEnergyNet(input_dim=Fi.cnf.nv)
        for t in range(args.t_max):
            # Forward pass: Compute predicted y by passing x to the model
            # inst: SAT formula, x is one possible assignment

            samples = [torch.from_numpy(xi).reshape(1, -1)]
            print("the input", samples[0].shape, samples[0].storage_type())
            prob = model.get_prob()
            while len(samples) <= args.num_samples:
                assignment, _ = sampler(Fi, prob)
                if len(assignment) > 0:
                    samples.append(assignment.reshape(1, -1))
                    # print("new assignment", assignment)
            new_inputs = torch.concat(samples, dim=0)
            phi = model(new_inputs)
            pseudo_loss = phi[0] - torch.sum(phi[1:]) / (len(phi) - 1)
            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.

            # Zero the gradients before running the backward pass.
            model.zero_grad()
            pseudo_loss.backward()

            # Update the weights using gradient descent. Each parameter is a Tensor, so
            # we can access its gradients like we did before.
            with torch.no_grad():
                for param in model.parameters():
                    param -= args.learning_rate * param.grad


if __name__ == "__main__":
    args = get_arguments()
    learn_sat_pref(args)
