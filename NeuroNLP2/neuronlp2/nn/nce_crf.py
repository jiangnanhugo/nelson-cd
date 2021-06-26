__author__ = 'max'

import torch
import torch.nn as nn
import numpy as np
from neuronlp2.nn.modules import BiAffine

device = torch.device('cuda', 0)


class NCETreeCRF(nn.Module):
    # Tree CRF layer.
    def __init__(self, model_dim):
        """ Args:
            model_dim:          int. the dimension of the input.
        """
        super(NCETreeCRF, self).__init__()
        self.model_dim = model_dim
        self.energy = BiAffine(model_dim, model_dim)

    def forward(self, heads, children, mask=None):
        ''' Args:
            heads:           Tensor. the head input tensor with shape = [batch, length, model_dim]
            children:        Tensor. the child input tensor with shape = [batch, length, model_dim]
            mask:    Tensor or None. the mask tensor with shape = [batch, length]
            lengths: Tensor or None. the length tensor with shape = [batch]
            Returns:         Tensor. he energy tensor with shape = [batch, length, length]
        '''
        batch, length, _ = heads.size()
        # [batch, length, length]
        output = self.energy(heads, children, mask_query=mask, mask_key=mask)
        return output

    def loss(self, heads, children, target_heads, mask=None):
        ''' Args:
                    heads:        Tensor. the head input tensor with shape = [batch, length, model_dim].
                    children:     Tensor. the child input tensor with shape = [batch, length, model_dim].
                    target_heads: Tensor. the tensor of target labels with shape [batch, length].
                    mask: Tensor or None. the mask tensor with shape = [batch, length].
                    Returns:      Tensor. A 1D tensor for minus log likelihood loss.
                '''
        batch, length, _ = heads.size()
        # [batch, length, length]
        energy = self(heads, children, mask=mask).double()
        A = torch.exp(energy)
        # mask out invalid positions
        if mask is not None:
            mask = mask.double()
            A = A * mask.unsqueeze(2) * mask.unsqueeze(1)

        # set diagonal elements to 0
        diag_mask = 1.0 - torch.eye(length).unsqueeze(0).type_as(energy)
        A = A * diag_mask
        energy = energy * diag_mask

        # get D [batch, length]
        D = A.sum(dim=1)
        rtol = 1e-4
        atol = 1e-6
        D += atol
        if mask is not None:
            D = D * mask

        # [batch, length, length]
        D = torch.diag_embed(D)

        # compute laplacian matrix
        # [batch, length, length]
        L = D - A

        if mask is not None:
            L = L + torch.diag_embed(1. - mask)

        # compute partition Z(x) [batch]
        L = L[:, 1:, 1:]
        z = torch.logdet(L)

        # first create index matrix [length, batch]
        index = torch.arange(0, length).view(length, 1).expand(length, batch)
        index = index.type_as(energy).long()
        batch_index = torch.arange(0, batch).type_as(index)
        # compute target energy [length-1, batch]
        tgt_energy = energy[batch_index, target_heads.t(), index][1:]
        # sum over dim=0 shape = [batch]
        tgt_energy = tgt_energy.sum(dim=0)

        return (z - tgt_energy).float()

    def sampled_loss(self, heads, children, target_heads, mask=None):
        ''' Args:
            heads:        Tensor. the head input tensor with shape = [batch, length, model_dim]
            children:     Tensor. the child input tensor with shape = [batch, length, model_dim]
            target_heads: Tensor. the tensor of target labels with shape [batch, length]
            mask: Tensor or None. the mask tensor with shape = [batch, length]
            Returns:      Tensor. A 1D tensor for minus log likelihood loss
        '''
        num_sampled_tree = 5
        batch, length, _ = heads.size()
        # [batch, length, length]
        energy = self(heads, children, mask=mask).double()
        A = torch.exp(energy)
        # mask out invalid positions
        if mask is not None:
            mask = mask.double()
            A = A * mask.unsqueeze(2) * mask.unsqueeze(1)

        # set diagonal elements to 0
        diag_mask = 1.0 - torch.eye(length).unsqueeze(0).type_as(energy)
        A = A * diag_mask
        energy = energy * diag_mask

        # first create index matrix [length, batch]
        index = torch.arange(0, length).view(length, 1).expand(length, batch)
        index = index.type_as(energy).long()
        batch_index = torch.arange(0, batch).type_as(index)
        # compute target energy [length-1, batch]
        positive_energy = energy[batch_index, target_heads.t(), index][1:]
        # sum over dim=0 shape = [batch]
        positive_energy = torch.exp(positive_energy.sum(dim=0))

        # sample 100 different spanning tree via random walk
        A_cpu = A.cpu().detach().numpy()
        estiamted_negative_engergy = torch.zeros(batch).to(device)
        for bi in range(batch):
            local_length = np.sum(np.sum(A_cpu[bi, :, :], axis=0) != 0)
            A_i = A_cpu[bi, :local_length, :local_length].reshape(local_length, local_length)
            num_samples = 0
            while num_samples <= num_sampled_tree:
                sampled_tree, score = self.sample_by_wilson(A_i, local_length)
                if sampled_tree == None:
                    continue
                num_samples += 1
                sampled_target_heads = np.zeros(length, dtype=np.int64)
                for mi, hi in sampled_tree:
                    sampled_target_heads[mi] = hi
                sampled_target_heads = torch.from_numpy(sampled_target_heads).to(device)
                one_negative_energy = energy[bi][sampled_target_heads.t(), index[:, bi]]
                estiamted_negative_engergy[bi] += torch.exp(one_negative_energy.sum())
        demoninator = estiamted_negative_engergy + positive_energy
        return -torch.sum(torch.log(positive_energy/demoninator) + torch.log(1 - positive_energy/demoninator))

    def sample_by_wilson(self, phi_matrix, num_of_nodes):
        global ValueError

        def random_walk(alist, unnormalized_prob):
            return np.random.choice(alist, p=unnormalized_prob / np.sum(unnormalized_prob))

        # initialized with only root node (the first word in sentence)
        spanning_tree = []
        nodes_in_spanning_tree = set()
        nodes_in_spanning_tree.add(0)
        unvisited = [idx for idx in range(1, num_of_nodes)]
        alist = [idx for idx in range(num_of_nodes)]
        stack = list()
        product_local_z = []
        try:
            while len(nodes_in_spanning_tree) < num_of_nodes:
                new_starting_point = np.random.choice(unvisited)
                visited_cycle = set()
                visited_cycle.add(new_starting_point)
                stack.append((new_starting_point, np.sum(phi_matrix[:, new_starting_point])))
                step = 0
                while True:
                    step += 1
                    if step >= 500:
                        raise ValueError("infinite loop")
                    rand_next = random_walk(alist, phi_matrix[:, new_starting_point])
                    if rand_next in nodes_in_spanning_tree:
                        stack.append((rand_next, np.sum(phi_matrix[:, rand_next])))
                        break
                    if rand_next in visited_cycle:
                        elem, _ = stack.pop()
                        while elem != rand_next and len(stack) >= 1:
                            elem, _ = stack.pop()
                            visited_cycle.remove(elem)
                    stack.append((rand_next, np.sum(phi_matrix[:, rand_next])))
                    visited_cycle.add(rand_next)
                if len(stack) <= 1:
                    continue
                for i in range(1, len(stack)):
                    nodes_in_spanning_tree.add(stack[i - 1][0])
                    unvisited.remove(stack[i - 1][0])
                    product_local_z.append(stack[i - 1][1])
                    spanning_tree.append((stack[i - 1][0], stack[i][0]))
                stack = list()
        except ValueError:
            # print(phi_matrix)
            print("INFINITE")
            return None, 0
        return sorted(spanning_tree), product_local_z
