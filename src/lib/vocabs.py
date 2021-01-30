from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from nlcodec import Type, learn_vocab, load_scheme, term_freq
from .misc import Filepath, FileReader


def get_ngrams(corps:List[List[Union[str, int]]], match_file:Filepath, bpe_file:Filepath,
                ngram:int=2, min_freq:int=0, max_ngrams:int=0):
    ngrams = dict()
    match = Vocabs(match_file)
    indexes = match.get_indexes()
    base = len(match)+1
    for corp in corps:
        for sent in tqdm(corp):
            words = [1 if tok in indexes else 0 for tok in sent]
            cwords = [word for word in words]
            for i in range(len(cwords)-2, -1, -1):
                cwords[i] = 0 if words[i]==0 else cwords[i+1]+words[i]
            for i, wl in enumerate(cwords):
                if wl >= ngram:
                    hash_val = 0
                    pbase = 1
                    for x in range(ngram):
                        hash_val += sent[i+x]*pbase
                        pbase*=base
                    if hash_val not in ngrams.keys():
                        ngrams[hash_val] = 0
                    ngrams[hash_val] += 1
    ngrams_list = [(key,value) for key,value in ngrams.items()]
    ngrams_list.sort(key=lambda x: x[1], reverse=True)
    
    bpe = Vocabs(bpe_file)
    if max_ngrams == 0:
        max_ngrams = len(bpe)
    ngrams = Vocabs()
    for ngram_pair in ngrams_list:
        hash_val, freq = ngram_pair
        if len(ngrams) >= max_ngrams or freq < min_freq:
            break 
        wlist = []
        while hash_val > 0:
            wlist.append(hash_val % base)
            hash_val = hash_val // base
        name = ''.join([bpe.table[x].name for x in wlist])
        kids = [bpe.table[x] for x in wlist]
        ngrams.append(Type(name, level=1, idx=len(ngrams), freq=freq, kids=kids))
    return ngrams

class Vocabs(object):
    def __init__(self, vocab_file:Filepath=None, table:List['Type']=[]):
        self.vocab_file = vocab_file
        self.table = table
        self.tokens = set()
        self.token_idx = dict()
        if self.vocab_file is not None:
            scheme = self.load(self.vocab_file)
            self.table = scheme.table
        if len(self.table) > 0:
            self.tokens = set([x.name for x in self.table])
            self.token_idx = { x.name:x.idx for x in self.table}

    def __len__(self):
        return len(self.table)

    def __iter__(self):
        for x in self.table:
            yield x

    def sort(self):
        table = self.table[5:]
        table.sort(key=lambda x: x.freq, reverse=True)
        self.table = self.table[:5]
        self.table.extend(table)
        self._reindex()

    def add(self, vocab:'Vocabs'):
        for token in vocab.table:
            if token.name not in self.tokens:
                self.append(token)
        self.sort()

    def append(self, token:'Type'):
        token.idx = len(self)
        self.table.append(token)
        self.tokens.add(token.name)
        self.token_idx[token.name] = token.idx

    def save(self, work_file):
        Type.write_out(self.table, work_file)

    def index(self, name:str):
        if name not in self.tokens:
            return None
        return self.token_idx[name]

    def find(self, name):
        if name in self.tokens:
            return True
        return False

    def get_indexes(self):
        return set(self.token_idx.values())

    def _reindex(self):
        for ix in range(len(self)):
            self.table[ix].ix = ix

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

