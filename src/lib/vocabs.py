from pathlib import Path
from typing import List, Union

from nlcodec import Reserved, Type, learn_vocab, load_scheme, term_freq

from .misc import Filepath, FileReader


def get_ngrams(match_file:Filepath, corps):
    pass

class Vocabs(object):
    def __init__(self, vocab_file:Filepath=None, table:List['Type']=[]):
        self.vocab_file = vocab_file
        self.table = table
        self.tokens = set()
        if self.vocab_file is not None:
            scheme = self.load(self.vocab_file)
            self.table = scheme.table
        if len(self.table) > 0:
            self.tokens = set([x.name for x in self.table])

    def __len__(self):
        return len(self.table)

    def __iter__(self):
        for x in self.table:
            yield x

    def sort(self):
        self.table.sort(key=lambda x: x.freq, reverse=True)

    def add(self, vocab:'Vocabs'):
        for token in vocab.table:
            if token.name not in self.tokens:
                self.append(token)
        self.sort()

    def append(self, token:'Type'):
        self.table.append(token)
        self.tokens.add(token.name)

    def save(self, work_file):
        Type.write_out(self.table, work_file)

    @classmethod
    def trim(cls, vocab, trimmed_size:int):
        vocab.sort()
        return Vocabs(table=vocab.table[:min(len(vocab), trimmed_size)])

    @classmethod
    def merge(cls, vocabs:List['Vocabs']):
        vcb = Vocabs()
        for vocab in vocabs:
            vcb.add(vocab)
        vcb.sort()
        return vcb

    @classmethod
    def make(cls, corpus:List[Union[str, Path]], model_path:Filepath, 
                vocab_size:int=8000, level:str='bpe'):
        fr = FileReader(corpus, segmented=False)
        corp = fr.unique()
        learn_vocab(inp=corp, level=level, model=model_path, vocab_size=vocab_size)

    @classmethod
    def write(cls, table:List['Type'], work_file:Filepath):
        Type.write_out(table, work_file)

    @classmethod
    def load(cls, vocab_file:Filepath):
        return load_scheme(vocab_file)

