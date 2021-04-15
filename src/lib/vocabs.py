import json
from collections import Counter
from typing import Dict, List, Union

from nlcodec import Reseved, Type, learn_vocab, load_scheme, term_freq

from .misc import Filepath, FileReader, FileWriter, get_now, log


class Vocabs(object):
    def __init__(self, vocab_file:Filepath=None, token_list=[]):
        self.vocab_file = vocab_file
        self.table = []
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

    def append(self, token:'Type', reindex:bool=False):
        if reindex:
            token = Type(token.name, level=token.level, idx=len(self), freq=token.freq, kids=token.kids)
        self.table.append(token)
        self.tokens.add(token.name)
        self.token_idx[token.name] = token.idx

    def save(self, work_file):
        try:
            Type.write_out(self.table, work_file)
        except Exception as e:
            self._write_out(work_file)

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

    def _read_in(self, vocab_file:Filepath, delim='\t'):
        fr = open(vocab_file, 'r')
        # print(self)
        self.kids_list = []
        for line in fr:
            line = line.strip()
            if line.startswith('#'):
                continue
            cols = line.split(delim)
            idx, name, level, freq = cols[:4]
            try:
                kids = list(map(int,cols[4].split(' ')))
            except Exception:
                kids = []
            self.append(Type(name, idx=int(idx), freq=int(freq), level=int(level), kids=None))
            self.kids_list.append(kids)
        fr.close()

    def _write_out(self, work_file:Filepath):
        fw = open(work_file, 'w')
        levels = Counter(v.level for v in self.table)
        max_level = max(levels.keys())
        meta = dict(total=len(self.table), levels=levels, max_level=max_level, create=get_now())
        meta = json.dumps(meta)
        fw.write(f'#{meta}\n')
        for i, item in enumerate(self.table):
            fw.write(f'{item.format()}\n')
        fw.close()

    def _reindex(self):
        for ix in range(len(self)):
            self.table[ix].idx = ix

    @classmethod
    def trim(cls, vocab, trimmed_size:int):
        vocab.sort()
        return cls(table=vocab.table[:min(len(vocab), trimmed_size)])

    @classmethod
    def merge(cls, vocabs:List['Vocabs']):
        vcb = cls()
        for vocab in vocabs:
            vcb.add(vocab)
        vcb.sort()
        return vcb

    @classmethod
    def make(cls, corpus:List[Filepath], model_path:Filepath, 
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
