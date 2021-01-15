""" Utils for building and storing the vocabs """

__author__ = "Dipesh Kumar"

import numpy as np
from tqdm import tqdm
from typing import Union
from pathlib import Path

from tokenizer import Tokenizer, Reserved
from misc import FileReader

class Vocabs(object):

    def __init__(self, filter=False):
        self.tokens = set()
        self.token2id = dict()
        self.id2token = dict()
        self.freqs = []
        self.next_index = 0
        self.filter = filter

    @classmethod
    def save(cls, vocabs, out_file, mode="w+"):
        ''' Save vocab to a file '''
        with open(out_file, mode) as f:
            for token in vocabs:
                f.write('{}\n'.format(token))

    @staticmethod
    def load(in_file):
        ''' Loads vocab from a file '''
        vcb = Vocabs()
        with open(in_file, "r+") as f:
            for line in f:
                token = line.strip()
                if token not in vcb.tokens:
                    vcb.append(token)
        return vcb

    @staticmethod
    def merge(*args):
        ''' Merges multiple vocab objects ''' 
        vcb = Vocabs()
        for vocab in args:
            vcb.add(vocab)
        return vcb

    @staticmethod
    def build(data_file, vcb_type="word"):
        ''' Builds vocab from a file '''
        vcb = Vocabs()
        with open(data_file, "r+") as f:
            for line in f:
                line = line.strip()
                tokens = line.split()
                for token in tokens:
                    if vcb_type == "char":
                        for ch in token:
                            vcb.append(ch)
                    else:
                        vcb.append(token)
        return vcb

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        curr_index = 0
        while curr_index < self.next_index:
            yield self.id2token[curr_index]
            curr_index += 1

    def word(self, index: int ):
        ''' Returns the word at a given index '''
        if index < len(self) and index > -1:
            return self.id2token[index]
        return None

    def index(self, word: str):
        ''' Returns the index of the word '''
        if word not in self.tokens:
            return None
        return self.token2id[word]

    def append(self, token: str, force_index=None):
        ''' Adds a token to the object '''
        if token in self.tokens:
            return
        
        if self.filter:
            pass

        self.tokens.add(token)

        if force_index is not None:
            curr = self.id2token[force_index]
            self.id2token[force_index] = token
            self.token2id[token] = force_index
            self.token2id[curr] = self.next_index
            self.next_index += 1
            return

        self.token2id[token] = self.next_index
        self.id2token[self.next_index] = token
        self.next_index += 1

    def add(self, vocab):
        ''' Adds a vocab to the current vocab object '''
        for token in vocab:
            self.append(token)

    @staticmethod
    def add_reserved(vcb=None, vcb_type:Union['char' or 'word'] = 'word'):
        reserved_tokens = Reserved.all(vcb_type=vcb_type)
        vocab = Vocabs()
        if vcb is not None:
            vocab.filter = vcb.filter
        for token in reserved_tokens:
            vocab.append(token)
        if vcb is not None:
            vocab.add(vcb)
        return vocab

    @staticmethod
    def lowered(vocab):
        ''' Returns a lowered vocab and a mapping dictionary '''
        vmap = [-1] * len(vocab)
        rvmap = dict()
        lvcb = Vocabs()
        for i, x in enumerate(vocab):
            lx = x.lower()
            if lvcb.index(lx) is not None:
                vmap[i] = lvcb.index(lx)
            else:
                index = len(lvcb)
                lvcb.append(lx)
                vmap[i] = index

        for i, v in enumerate(vmap):
            if v not in rvmap.keys():
                rvmap[v] = []
            rvmap[v].append(i)
        return lvcb, rvmap
