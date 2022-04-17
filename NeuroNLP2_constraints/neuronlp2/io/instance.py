__author__ = 'max'

__all__ = ['Sentence', 'DependencyInstance']


class Sentence(object):
    def __init__(self, words, word_ids, char_seqs, char_id_seqs):
        self.words = words
        self.word_ids = word_ids
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs

    def length(self):
        return len(self.words)


class DependencyInstance(object):
    def __init__(self, sentence, postags, pos_ids, heads, types, type_ids, constraint_pos_mask=None,
                 con_relation_mask=None):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.heads = heads
        self.types = types
        self.type_ids = type_ids
        self.constraint_pos_mask = constraint_pos_mask
        self.con_relation_mask = con_relation_mask

    def length(self):
        return self.sentence.length()
