import numpy as np
import torch


# $\nabla \log \det(L)=\nabla \text{tr}(\log L)=(\nabla \log L)^\top=(L^{-1})^\top$
def nabla_log_determint_laplacian(A, size):
    A = np.exp(A)
    diag_mask = 1.0 - np.eye(size)
    A = A * diag_mask
    D = A.sum(axis=1)
    atol = 1e-6
    D += atol
    D = np.diag(D)
    L = D - A
    nabla_log_Z = np.linalg.inv(L).transpose()
    return nabla_log_Z


def determinant_of_laplacian_matrix(A, size):
    A = np.exp(A)
    diag_mask = 1.0 - np.eye(size)
    A = A * diag_mask
    D = A.sum(axis=1)
    atol = 1e-6
    D += atol
    D = np.diag(D)
    L = D - A
    return np.linalg.det(L[1:, 1:])


def sample_by_wilson(phi_matrix, num_of_nodes):
    phi_matrix = np.exp(phi_matrix)
    diag_mask = 1.0 - np.eye(num_of_nodes)
    phi_matrix = phi_matrix * diag_mask

    def random_walk(alist, unnormalized_prob, start_pos):
        sumed = np.sum(unnormalized_prob)
        return np.random.choice(alist, p=unnormalized_prob / sumed)

    # initialized with only root node (the first word in sentence)
    spanning_tree = []
    nodes_in_spanning_tree = set()
    nodes_in_spanning_tree.add(0)
    unvisited = [idx for idx in range(1, num_of_nodes)]
    alist = [idx for idx in range(num_of_nodes)]
    stack = list()
    product_local_z = []
    while len(nodes_in_spanning_tree) < num_of_nodes:
        new_starting_point = np.random.choice(unvisited)
        visited_cycle = set()
        visited_cycle.add(new_starting_point)
        stack.append((new_starting_point, np.sum(phi_matrix[new_starting_point])))
        print("adding {} {}".format(new_starting_point, phi_matrix[new_starting_point]))
        while True:
            rand_next = random_walk(alist, phi_matrix[new_starting_point], new_starting_point)
            if rand_next in nodes_in_spanning_tree:
                stack.append((rand_next, np.sum(phi_matrix[rand_next])))
                break
            if rand_next in visited_cycle:
                elem, _ = stack.pop()
                while elem != rand_next and len(stack) >= 1:
                    elem, _ = stack.pop()
                    visited_cycle.remove(elem)
            stack.append((rand_next, np.sum(phi_matrix[rand_next])))
            visited_cycle.add(rand_next)
        if len(stack) <= 1:
            continue

        for i in range(1, len(stack)):
            nodes_in_spanning_tree.add(stack[i-1][0])
            unvisited.remove(stack[i-1][0])
            product_local_z.append(stack[i-1][1])
            spanning_tree.append((stack[i-1][0], stack[i][0]))
            # prev = next
        stack = list()
        # prev, local_z = stack.pop()
        # product_local_z = [local_z]
        # while len(stack) != 0:
        #     next, local_z = stack.pop()
        #     nodes_in_spanning_tree.add(next)
        #     unvisited.remove(next)
        #     product_local_z.append(local_z)
        #     spanning_tree.append((next, prev))
        #     prev = next

    return sorted(spanning_tree), product_local_z


def torch_log_z(energy, length):
    A = torch.exp(energy)
    # set diagonal elements to 0
    diag_mask = 1.0 - torch.eye(length).type_as(energy)
    A = A * diag_mask

    # get D [length]
    D = A.sum(dim=1)

    atol = 1e-6
    D += atol

    D = torch.diag_embed(D)
    L = D - A
    L = L[1:, 1:]
    z = torch.logdet(L)
    return z


def get_feat(xi, xj, type=0):
    if type == 0:
        return np.concatenate((xi, xj), axis=0)
    else:
        return xi * xj


if __name__ == '__main__':
    N = 10
    dim = 1
    x = np.abs(np.random.randn(N, dim))
    theta = np.abs(np.random.randn(N, N, dim))
    data = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            data[i, j] = np.matmul(theta[i, j], get_feat(x[i], x[j], 1))
            print(get_feat(x[i], x[j], 1), theta[i, j])
    # done with data loading
    print("data from numpy:", data)

    Z = determinant_of_laplacian_matrix(data, N)
    nabla_log_z = nabla_log_determint_laplacian(data, N)

    # The following part is used to estimate the $ nabla log Z $ via sampling + contrastive divergence.
    partial_derative = np.zeros_like(theta)
    for _ in range(1000):
        spanning_tree, product_local_z = sample_by_wilson(data, N)
        print(spanning_tree, product_local_z)
        for mi, hi in spanning_tree:
            partial_derative[mi, hi, :] += get_feat(x[mi], x[hi], 1) * np.product(product_local_z)
    partial_derative /= 1000 * Z

    # by PyTorch
    neural_theta = torch.nn.Parameter(torch.from_numpy(theta), requires_grad=True)
    neural_x = torch.from_numpy(x)
    neural_data = torch.zeros((N, N))
    for i in range(N):
        for j in range(N):
            neural_data[i, j] = torch.matmul(neural_theta[i, j], neural_x[i] * neural_x[j])
            print((neural_x[i] * neural_x[j]).detach().numpy(), neural_theta[i, j].detach().numpy())

    print("neural_data:", neural_data.detach().numpy())
    z = torch_log_z(neural_data, N)
    z.backward()
    neural_derative = neural_theta.grad
    for i in range(N):
        for j in range(N):
            print("numpy vs. torch:", partial_derative[i, j],  neural_derative[i, j].cpu().numpy())
            # print()

    # print("nabla log det(L):",np.matmul(data, nabla_log_determint_laplacian(data, N)))
