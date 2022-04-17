__author__ = 'max'

import torch
import torch.nn as nn
from neuronlp2.nn.modules import BiAffine


class MaxMargin(nn.Module):
    # Tree CRF layer.
    def __init__(self, model_dim):
        """ Args:
            model_dim:          int. the dimension of the input.
        """
        super(MaxMargin, self).__init__()
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