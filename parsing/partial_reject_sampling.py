from itertools import cycle
import numpy as np
from numba import njit
import itertools


def get_cycle(parents):
    # dfs algorithm to detect cycle
    visited = set()
    def dfs(cur, parents, stack):
        new_cur = parents[cur]
        if new_cur not in visited:
            visited.add(new_cur)
            stack.append(new_cur)
            dfs(new_cur, parents, stack)
        elif new_cur in stack:
            print('find cycle')
            return stack
        else:
            print("return nothing")
            return None
    vertices_in_cycles=[]
    for cur in range(len(parents)):
        if cur not in visited:
            visited.add(cur)
            stack=list()
            stack.append(cur)
            ret=dfs(cur, parents, stack)
            if ret!=None:
                vertices_in_cycles.append(ret)
    return vertices_in_cycles


def partial_reject_sampling(phi_matrix, num_of_nodes):
    phi_matrix = np.exp(phi_matrix)
    diag_mask = 1.0 - np.eye(num_of_nodes)
    phi_matrix = phi_matrix * diag_mask

    @njit
    def choose_outgoing_edge_randomly(adj_matrix):
        # sample the outgoing edges with probability proportional to the edge weight
        sumed = np.sum(adj_matrix, axis=1)
        return [np.random.choice(range(adj_matrix.shape[1]), p=adj_matrix[i,:] / sumed[i]) for i in range(len(sumed))]

    # the list of vertices to be resamples, initially all vertices need to be (re)-sampled
    candidates = [idx for idx in range(num_of_nodes)]
    # the parent of the i-th node
    par =[idx for idx in range(num_of_nodes)]
    while True:
        print("candidates:", candidates)
        resampled=choose_outgoing_edge_randomly(phi_matrix[candidates,:])
        for c, r in zip(candidates, resampled):
            # update the resampled edges into the global parent array
            par[c]=r
        # check if the parent array contains any cycles
        vertices_in_cycles = get_cycle(par)
        print("cycle detection:", vertices_in_cycles)
        if len(vertices_in_cycles) == 0:
            # length = 0 means there is no cycle detected
            break
        # extract the vertices that require to be resampled
        candidates = list(itertools.chain(*vertices_in_cycles))

    print("spanning tree output:", par)
    return par


    


def get_random_graph(N=10, dim=1):
    def get_feat(xi, xj, type=0):
        if type == 0:
            return np.concatenate((xi, xj), axis=0)
        else:
            return xi * xj
    x = np.abs(np.random.randn(N, dim))
    theta = np.abs(np.random.randn(N, N, dim))
    data = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            data[i, j] = np.matmul(theta[i, j], get_feat(x[i], x[j], 1))
            # print(get_feat(x[i], x[j], 1), theta[i, j])
    print("done with data loading")
    print("data from numpy:", data.shape)
    return data, theta

if __name__ == '__main__':
    graph_dim = 10
    data, theta = get_random_graph(graph_dim)
    # The following part is used to estimate the $ nabla log Z $ via sampling 
    
    for _ in range(10):
        spanning_tree = partial_reject_sampling(data, graph_dim)