__author__ = 'max'

from neuronlp2.io.instance import DependencyInstance
from neuronlp2.io.instance import Sentence
from neuronlp2.io.common import ROOT, ROOT_POS, ROOT_CHAR, ROOT_TYPE, END, END_POS, END_CHAR, END_TYPE
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH
import numpy as np


class CoNLLXReader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet
        self.upos_dict = {}
        self.relation_dict = {}

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False, constraint_pos_dict=None,
                constraint_relation_dict=None):
        line = self.__source_file.readline()
        # skip multiple blank lines.

        while (len(line) > 0 and len(line.strip()) == 0) or line.strip().startswith('#'):
            line = self.__source_file.readline()
        if len(line) == 0:
            return None
        # print(line)
        lines = []
        while len(line.strip()) > 0:
            line = line.strip().split('\t')
            if not "." in line[0]:
                lines.append(line)
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        upostags = []
        types = []
        type_ids = []
        heads = []
        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            upostags.append(ROOT_POS)
            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))
            heads.append(0)

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            upos = tokens[3]
            pos = tokens[4]
            # current modifier points to its head
            head = int(tokens[6])
            type = tokens[7]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))
            upostags.append(upos)

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)
        for i in range(len(upostags)):
            upos = upostags[i]
            head_upos = upostags[heads[i]]
            if upos not in self.upos_dict:
                self.upos_dict[upos] = [head_upos]
            elif head_upos not in self.upos_dict[upos]:
                self.upos_dict[upos].append(head_upos)

        for i in range(len(upostags)):
            upos = upostags[i]
            tp = types[i]
            if upos not in self.relation_dict:
                self.relation_dict[upos] = [tp]
            elif tp not in self.relation_dict[upos]:
                self.relation_dict[upos].append(tp)

        con_upos_mask = np.zeros((len(upostags), len(upostags)))
        for i in range(len(upostags)):
            msk = np.zeros(len(upostags))
            for j in range(len(upostags)):
                if i == j:
                    continue
                if upostags[i] in constraint_pos_dict and upostags[j] in constraint_pos_dict[upostags[i]]:
                    msk[j] = 1.0

            con_upos_mask[:, i] = msk
            if i > 0 and np.sum(msk) == 0:
                raise ValueError("all zero mask")

        con_relation_mask = np.zeros((len(upostags), self.__type_alphabet.size()))
        for i in range(len(upostags)):
            msk = np.zeros(self.__type_alphabet.size())
            if upostags[i] in constraint_relation_dict:
                relation_labels = constraint_relation_dict[upostags[i]]
                for x in relation_labels:
                    idx = self.__type_alphabet.get_index(x)
                    msk[idx] = 1.0
            # else:
            #     print(upostags[i], upostags[heads[i]])
            msk[type_ids[i]]=1.0
            con_relation_mask[i, :] = msk
        # check conflicting
        for bi in range(len(upostags)):
            msk = con_relation_mask[bi,:]
            type_idx= type_ids[bi]
            if msk[type_idx]==0 and bi!=0:
                print(words[bi], upostags[bi], upostags[heads[bi]], types[bi])
                # raise  ValueError("MASK ERROR")

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            heads.append(0)

        return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, heads, types,
                                  type_ids, con_upos_mask, con_relation_mask)
