from argparse import ArgumentParser
import torch
from sat_instance import SAT_Instance, generate_solution_with_preference
from CNF.sampler.lovasz_sat import pyotrch_conditioned_partial_rejection_sampling_sampler
from sampler import *

class DeepEnergyNet(torch.nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.beta = 1
        self.theta = torch.nn.Parameter(torch.randn((input_dim)))
    
    def forward(self, x):
        pref_score = -self.beta * torch.mul(self.theta, x)
        return torch.exp(pref_score)

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--t_max', help='The maximum learning iterations', default=10000, type=int)
    parser.add_argument('--learning_rate', type=str, help="learning rate of SGD", default=1e-4, type=float)
    parser.add_argument('--num_samples', help='number of samples for the neural Lovasz sampler', type=int, default=10)
    return parser.parse_args()



def learn_sat_pref(args):
    # Construct our model by instantiating the class defined above
    model = DeepEnergyNet()

    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    # criterion = torch.nn.MSELoss(reduction='sum')
    instances = []
    with open(args.input_file, 'r') as file:
        instances.append(SAT_Instance.from_file(file, K=3))
    
    sat_train_data = generate_random_solutions_with_preference(instances)

    if args.algo == 'lll':
        sampler =  constructive_lovasz_local_lemma_sampler
    elif args.algo == 'mc':
        sampler = Monte_Carlo_sampler
    elif args.algo == 'prs':
        sampler = conditioned_partial_rejection_sampling_sampler
    elif args.algo == 'nls':
        sampler = pyotrch_conditioned_partial_rejection_sampling_sampler

    for t in range(args.t_max):
        # Forward pass: Compute predicted y by passing x to the model
        # inst: SAT formula, x is one possible assignment
        for xi, Fi in sat_train_data:
            
            C = model(xi)
            samples = []
            while len(samples) <= args.num_samples:
                assignment, _ = sampler(Fi)
                if len(assignment)>0:
                    samples.append(assignment)


            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.

            # Zero the gradients before running the backward pass.
            
            model.zero_grad()
            loss.backward()

            # Update the weights using gradient descent. Each parameter is a Tensor, so
            # we can access its gradients like we did before.
            with torch.no_grad():
                for param in model.parameters():
                    gradient = param.grad + samples.graident
                    param = param - args.learning_rate * gradient


if __name__ == "__main__":
    args = get_arguments()
    learn_sat_pref(args)


