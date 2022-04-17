import codecs


class Dependency(object):
    def __init__(self, id, form, tag, head, rel):
        self.id = id
        self.org_form = form
        self.form = form.lower()
        self.tag = tag
        self.head = head
        self.rel = rel

    def __str__(self):
        values = [str(self.id), self.org_form, self.tag, str(self.head), self.rel]
        return '\t'.join(values)

    def __repr__(self):
        values = [str(self.id), self.org_form, str(self.head), self.rel]
        return "\n" + '\t'.join(values)

    @property
    def pseudo(self):
        return self.id == 0 or self.form == '<eos>'


class DepTree(object):
    def __init__(self, sentence):
        self.words = list(sentence)
        self.start = 1
        if sentence[0].id == 1:
            self.start = 0
        elif sentence[0].id == 0:
            self.start = 1
        else:
            self.start = len(self.words)

    def isProj(self):
        n = len(self.words)
        words = self.words
        if self.start > 1:
            return False
        if self.start == 0:
            words = [None] + words
        for i in range(1, n):
            hi = words[i].head
            for j in range(i + 1, hi):
                hj = words[j].head
                if (hj - hi) * (hj - i) > 0:
                    return False
        return True


def readDepTree(file, vocab=None):
    proj = 0
    total = 0
    min_count = 1
    if vocab is None:
        min_count = 0

    if vocab is None:
        sentence = []
    else:
        sentence = [Dependency(0, vocab._root_form, vocab._root, 0, vocab._root)]
    for line in file:
        tok = line.strip().split('\t')
        # print(tok)
        if not tok or line.strip() == '' or line.strip().startswith('#'):
            if len(sentence) > min_count:
                if DepTree(sentence).isProj():
                    proj += 1
                total += 1
                yield sentence
            if vocab is None:
                sentence = []
            else:
                sentence = [Dependency(0, vocab._root_form, vocab._root, 0, vocab._root)]
        elif len(tok) == 10:
            if tok[6] == '_':
                tok[6] = '-1'
            try:
                sentence.append(Dependency(int(tok[0]), tok[1], tok[3], int(tok[6]), tok[7]))
            except Exception:
                pass
        else:
            pass

    if len(sentence) > min_count:
        if DepTree(sentence).isProj():
            proj += 1
        total += 1
        yield sentence

    print("Total num: ", total)
    print("Proj num: ", proj)


def read_corpus(file_path, vocab=None):
    data = []
    with codecs.open(file_path, 'r', encoding="utf-8") as infile:
        for sentence in readDepTree(infile, vocab):
            data.append(sentence)
    # print(data[0])
    return data


if __name__ == '__main__':
    read_corpus("/dataset/UD_Croatian-SET/hr_set-ud-train.conllu")
