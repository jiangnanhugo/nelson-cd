import numpy as np
from dependency_load import read_corpus
from vocab_load import creatVocab
import argparse
import pickle

# $\nabla \log \det(X)=\nabla \text{tr}(\log X)=(\nabla \log X)^\top=(X^{-1})^\top$
def nabla_log_determint_laplacian(theta, size):
    A = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                A[i, j] = - np.exp(theta[i, j])

    r = np.zeros(size)
    for m in range(size):
        A[m, m] = - np.sum(A[:, m])
        r[m] = np.exp(theta[0, m])

    A[0, :] = r
    det = np.linalg.inv(A).transpose()
    return det

def determinant_of_laplacian_matrix(theta, size):
    A = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                A[i, j] = - np.exp(theta[i, j])

    r = np.zeros(size)
    for m in range(size):
        A[m, m] = - np.sum(A[:, m])
        r[m] = np.exp(theta[0, m])

    A[0, :] = r
    # \hat{L}(theta)
    det = np.linalg.det(A)
    return det


def sample_by_wilson(phi_matrix, num_of_nodes):
    def random_walk(alist, unnormalized_prob):
        sumed = np.sum(unnormalized_prob)
        return np.random.choice(alist, p=unnormalized_prob / sumed), sumed

    # initialized with only root node (the first word in sentence)
    spanning_tree = []
    nodes_in_spanning_tree = set()
    nodes_in_spanning_tree.add(0)
    unvisited = [idx for idx in range(1, num_of_nodes)]
    alist = [idx for idx in range(num_of_nodes)]
    stack = list()
    while len(nodes_in_spanning_tree) < num_of_nodes:
        new_starting_point = np.random.choice(unvisited)
        visited_cycle = set()
        visited_cycle.add(new_starting_point)
        stack.append([new_starting_point, 1])
        while True:
            rand_next, local_z = random_walk(alist, np.exp(phi_matrix[new_starting_point]))
            if rand_next in nodes_in_spanning_tree:
                stack.append((rand_next, local_z))
                break
            if rand_next in visited_cycle:
                elem , _ = stack.pop()
                while elem != rand_next and len(stack) >= 1:
                    elem,_ = stack.pop()
                    visited_cycle.remove(elem)
            stack.append([rand_next, local_z])
            visited_cycle.add(rand_next)
        if len(stack) <= 1:
            continue
        prev, prev_local_z = stack.pop()
        product_local_z = prev_local_z
        while len(stack) != 0:
            next, next_local_z = stack.pop()
            product_local_z *= next_local_z
            nodes_in_spanning_tree.add(next)
            unvisited.remove(next)
            spanning_tree.append((prev, next))
            prev = next

    return spanning_tree, product_local_z


def get_arguments():
    parser = argparse.ArgumentParser(description='PALM')
    parser.add_argument('--train_file', type=str, help="train dataset",
                        default="/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_Croatian-SET/hr_set-ud-train.conllu")
    parser.add_argument('--dev_file', type=str, help="dev dataset",
                        default="/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_Croatian-SET/hr_set-ud-dev.conllu")
    parser.add_argument('--min_occur_count', type=int, help="dev dataset", default=1)
    parser.add_argument('--hidden_size', type=int, help="dev dataset", default=100)
    parser.add_argument('--number_of_samples', type=int, help="samples for spanning tree", default=100)
    parser.add_argument('--save_vocab_path', type=str, help="vocab", default="saved_model/vocab.pkl")
    parser.add_argument('--pretrained_embeddings_file', type=str, help="vocab", default="word_embd/glove.6B.100d.txt")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    data = np.random.randn(100, 100)/4
    determint_laplacian = determinant_of_laplacian_matrix(data, 100)
    print(nabla_log_determint_laplacian(data, 100))
    print("determint_laplacian:", determint_laplacian)
    for i in range(10):
        tree, local_z = sample_by_wilson(data, 100)
        for h,m in tree:
            local_z *= np.exp(data[h,m])
        print(tree, local_z)
    exit(1)
    args = get_arguments()
    vocab = creatVocab(args.train_file, args.min_occur_count)
    theta = np.random.rand(vocab.vocab_size, args.hidden_size)
    vec = vocab.load_pretrained_embs(args.pretrained_embeddings_file)
    pickle.dump(vocab, open(args.save_vocab_path, 'wb'))

    train_data = read_corpus(args.train_file, vocab)

    for x in train_data:
        sentence_vec = np.zeros((len(x), args.hidden_size))
        for i in range(len(x)):
            print(x[i].org_form)
            word_idx = vocab.word2id(x[i].org_form)
            sentence_vec[i, :] = vec[word_idx]
        word_matrix = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j:
                    word_matrix[i, j] = np.inner(sentence_vec[i, :], sentence_vec[j, :]) / args.hidden_size
        determint_laplacian = determinant_of_laplacian_matrix(word_matrix, len(x))
        print("determint_laplacian:", determint_laplacian)
        for _ in range(args.number_of_samples):
            tree, local_z = sample_by_wilson(word_matrix, len(x))
            for h, m in tree:
                local_z *= np.exp(word_matrix[h, m])
            print(local_z)


    # data = np.random.randn((20, 20))
    # phi = mtt_model.phi(data)
