import torch
import random
import numpy as np
from lll.sampler.nelson.lovasz_sat import pytorch_neural_lovasz_sampler
from lll.utils.sat_instance import get_all_solutions
from lll.model_learn.draw_from_all_samplers import draw_from_waps, draw_from_weightgen, draw_from_cmsgen, draw_from_unigen, \
    draw_from_prs_series, draw_from_kus, draw_from_quicksampler, draw_from_xor_sampling, draw_from_gibbs_sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(10010)
random.seed(10010)
np.random.seed(10010)

np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.inf, linewidth=np.inf)


def compute_distance(samples, model, nabla_logZ):
    new_inputs = torch.concat(samples, dim=0).to(device)
    phi = model(new_inputs)
    summed_phi = torch.sum(phi)
    model.zero_grad()
    summed_phi.backward()
    with torch.no_grad():
        apx_nabla_logZ = model.theta.grad / len(samples)
        # print(apx_nabla_logZ.cpu().list())
        print("${:.6f}$ &".format(torch.linalg.norm(apx_nabla_logZ - nabla_logZ, ord=1)), end='\t')


def compute_nabla_log_ZC(cnf_instance, model, num_samples, input_file, **kwargs):
    prob = kwargs['prob']

    all_solutions = get_all_solutions(cnf_instance)
    ZC = 0
    nabla_logZ = 0
    for x in all_solutions:
        inputs = torch.from_numpy(x).to(device).reshape(1, -1)
        phi = model(inputs)
        exp_phi = torch.exp(phi)
        model.zero_grad()
        phi.backward()
        with torch.no_grad():
            grad_phi_x = model.theta.grad
            nabla_logZ += exp_phi * grad_phi_x
            ZC += exp_phi
    nabla_logZ /= ZC
    sample_waps = draw_from_waps(num_samples, input_file, cnf_instance, prob)
    sample_gibbs = draw_from_gibbs_sampling(num_samples=num_samples, cnf_instance=cnf_instance, input_file=input_file, prob=prob)
    sample_quick = draw_from_quicksampler(num_samples, input_file)
    sample_cmsgen = draw_from_cmsgen(num_samples, input_file)
    sample_unigen = draw_from_unigen(num_samples, input_file)
    sample_kus = draw_from_kus(num_samples, input_file)
    clause2var, weight, bias = cnf_instance.get_formular_matrix_form()
    clause2var, weight, bias = torch.from_numpy(clause2var).int().to(device), \
                               torch.from_numpy(weight).int().to(device), \
                               torch.from_numpy(bias).int().to(device)
    samples = draw_from_prs_series('nelson_batch', cnf_instance, clause2var, weight, bias, prob, num_samples=num_samples,
                                   sampler_batch_size=kwargs['sampler_batch_size'])
    sample_nelson = [x.reshape(1, -1) for x in samples]
    print('weightgen')
    sample_weightgen = draw_from_weightgen(num_samples, input_file, cnf_instance, prob)
    print('done')
    for ratio in range(10, 11):
        size = num_samples * ratio // 10
        compute_distance(sample_weightgen[:size], model, nabla_logZ)
        compute_distance(sample_nelson[:size], model, nabla_logZ)
        print("& ", end=" ")
        compute_distance(sample_waps[:size], model, nabla_logZ)
        print(" &", end=" ")
        compute_distance(sample_cmsgen[:size], model, nabla_logZ)
        compute_distance(sample_kus[:size], model, nabla_logZ)
        compute_distance(sample_quick[:size], model, nabla_logZ)
        compute_distance(sample_unigen[:size], model, nabla_logZ)
        compute_distance(sample_gibbs[:size], model, nabla_logZ)

        print(" \\\\")
    #


def compute_nabla_log_ZC_XOR(cnf_instance, model, num_samples, input_file, **kwargs):
    prob = kwargs['prob']
    all_solutions = get_all_solutions(cnf_instance)
    ZC = 0
    nabla_logZ = 0
    for x in all_solutions:
        inputs = torch.from_numpy(x).to(device).reshape(1, -1)
        phi = model(inputs)
        exp_phi = torch.exp(phi)
        model.zero_grad()
        phi.backward()
        with torch.no_grad():
            grad_phi_x = model.theta.grad
            nabla_logZ += exp_phi * grad_phi_x
            ZC += exp_phi
    nabla_logZ /= ZC
    sample_xorsampling = draw_from_xor_sampling(num_samples=num_samples, cnf_instance=cnf_instance, input_file=input_file, prob=prob)
    print('done')
    for ratio in range(10, 11):
        size = num_samples * ratio // 10
        compute_distance(sample_xorsampling[:size], model, nabla_logZ)

        print(" \\\\")
    #


def evaluation(formula, clause2var, weight, bias, algo, neural_net, preferred_inputs, less_preferred_inputs, list_of_Ks=[5, 10, 20],
               **kwargs):
    # compute log-likelihood
    # see Handbook of Satisfiability Chapter 20
    # URL: https://www.cs.cornell.edu/~sabhar/chapters/ModelCounting-SAT-Handbook-prelim.pdf Page 11
    num_samples = kwargs['num_samples']
    input_file = kwargs['input_file']
    samples = [torch.from_numpy(preferred_inputs[0]).float().to(device).reshape(1, -1), ]

    # prob = neural_net.get_prob()
    # if algo == 'quicksampler':
    #     samples += draw_from_quicksampler(num_samples, input_file)
    # elif args.algo == 'unigen':
    #     samples += draw_from_unigen(num_samples, input_file)
    # elif args.algo == 'cmsgen':
    #     samples += draw_from_cmsgen(num_samples, input_file)
    # elif args.algo == 'kus':
    #     samples += draw_from_kus(num_samples, input_file)
    # elif args.algo == 'waps':
    #     samples += draw_from_waps(num_samples, input_file, formula, prob)
    # elif args.algo == 'weightgen':
    #     samples += draw_from_weightgen(args.num_samples, args.input_file, formula, prob)
    # elif algo in ['lll', 'mc', 'lll', 'nelson', 'nelson_batch']:
    #     samples += draw_from_prs_series(args.algo, formula, clause2var, weight, bias, prob, num_samples)
    #     samples = [x.reshape(1, -1) for x in samples]

    new_inputs = torch.concat(samples, dim=0).to(device)
    phi = neural_net(new_inputs)
    pseudo_loss = phi[0]  # - torch.sum(phi[1:]) / (len(phi) - 1)
    uni_samples = []
    while len(uni_samples) <= num_samples:
        assignment, count, time_used = pytorch_neural_lovasz_sampler(formula, clause2var, weight, bias, device=device, prob=0.5)

        if len(assignment) > 0:
            uni_samples.append(assignment.reshape(1, -1))
    averaged_ratio = num_samples / torch.sum(torch.concat(uni_samples, dim=0).to(device), dim=0)
    log_model_count = torch.sum(torch.log(averaged_ratio))
    print("log-likelihood: {:.6f}".format(- pseudo_loss))

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
        print("{:.6f}".format(map / cnt), end=" ")
    print()
