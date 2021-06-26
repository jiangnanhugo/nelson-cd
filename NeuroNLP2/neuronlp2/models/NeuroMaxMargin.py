from overrides import overrides
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.nn import TreeCRF, VarGRU, VarRNN, VarLSTM, VarFastLSTM
from neuronlp2.nn import BiAffine, BiLinear, CharCNN
from neuronlp2.tasks import parser
from neuronlp2.models import DeepBiAffine

class NeuroMaxMargin(DeepBiAffine):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode, hidden_size, num_layers,
                 num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33),
                 pos=True, activation='elu'):
        super(NeuroMaxMargin, self).__init__(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode,
                                       hidden_size, num_layers, num_labels, arc_space, type_space,
                                       embedd_word=embedd_word, embedd_char=embedd_char, embedd_pos=embedd_pos,
                                       p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=pos, activation=activation)
        self.biaffine = None
        self.treecrf = TreeCRF(arc_space)

    def forward(self, input_word, input_char, input_pos, mask=None):
        # output from rnn [batch, length, dim]
        arc, type = self._get_rnn_output(input_word, input_char, input_pos, mask=mask)
        # [batch, length_head, length_child]
        out_arc = self.treecrf(arc[0], arc[1], mask=mask)
        return out_arc, type

    @overrides
    def loss(self, input_word, input_char, input_pos, heads, types, mask=None, constraint_pos_msk=None):
        # output from rnn [batch, length, dim]
        arc, out_type = self._get_rnn_output(input_word, input_char, input_pos, mask=mask)
        # [batch]
        loss_arc = self.treecrf.loss(arc[0], arc[1], heads, mask=mask)
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type

        # get vector for heads [batch, length, type_space],
        type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)
        loss_type = self.criterion(out_type.transpose(1, 2), types)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_type = loss_type * mask

        return loss_arc, loss_type[:, 1:].sum(dim=1)



    @overrides
    def decode(self, input_word, input_char, input_pos, mask=None, leading_symbolic=0):
        """
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and types.

        """
        # out_arc shape [batch, length_h, length_c]
        energy, out_type = self(input_word, input_char, input_pos, mask=mask)
        # compute lengths
        length = mask.sum(dim=1).long()
        heads, _ = parser.decode_MST(energy.cpu().numpy(), length.cpu().numpy(), leading_symbolic=leading_symbolic,
                                     labeled=False)
        types = self._decode_types(out_type, torch.from_numpy(heads).type_as(length), leading_symbolic)
        return heads, types.cpu().numpy()
