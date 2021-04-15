import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

from nlcodec import Reseved, Type, learn_vocab, load_scheme, term_freq
from tqdm import tqdm

from .misc import Filepath, FileReader, FileWriter, get_now, log
from .vocabs import Vocabs

class Grams(object):

    @classmethod
    def get_ngrams(cls, corps:List[List[Union[str, int]]], match_file:Filepath, bpe_file:Filepath,
                    word_file:Filepath, ngram:int=2, min_freq:int=0, max_ngrams:int=0, 
                    sorter:str='freq'):
        ngrams = dict()

        bpe = Vocabs(vocab_file=bpe_file)
        base = len(bpe)+1
        max_ngrams = len(bpe) if max_ngrams == 0 else max_ngrams

        match = Vocabs()
        match._read_in(match_file)
        indexes = match.get_indexes()

        for corp in corps:
            for sent in tqdm(corp):
                words = cls._filter(sent, indexes, bpe)
                cwords = cls._cumulate(words)
                for i, wl in enumerate(cwords):
                    if wl >= ngram:
                        hash_val = cls._hash(sent[i:i+ngram], base)
                        if hash_val not in ngrams.keys():
                            ngrams[hash_val] = 0
                        ngrams[hash_val] += 1 
        
        ngrams = { k:v for k,v in ngrams.items() if v >= min_freq}
        ngrams_list = GramsSorter.sort(ngrams, bpe_file, word_file, sorter=sorter)    
        ngrams_list = ngrams_list[:max_ngrams] if len(ngrams_list) > max_ngrams else ngrams_list

        ngrams_vcb = Vocabs()
        for ngram_pair in ngrams_list:
            hash_val, _ = ngram_pair
            freq = ngrams[hash_val]
            if len(ngrams_vcb) < max_ngrams:
                wlist = cls._unhash(hash_val, base)
                token = cls._make_token(wlist, bpe, len(ngrams_vcb), freq)
                ngrams_vcb.append(token)
        log(f'Found {len(ngrams_vcb)} {ngram}-gram [ min_freq : {min_freq}, max_ngrams : {max_ngrams}]', 2)    
        return ngrams_vcb, ngrams_list

    # make shift skip gram : Considers A _ B cases only
    @classmethod
    def get_skipgrams(cls, corps:List[List[Union[str, int]]], match_file:Filepath, bpe_file:Filepath,
                        word_file:Filepath, min_freq:int=0, max_sgrams:int=0, sorter:str='freq'):
        sgrams = dict()

        bpe = Vocabs(vocab_file=bpe_file)
        base = len(bpe)+1
        max_sgrams = len(bpe) if max_sgrams == 0 else max_sgrams

        match = Vocabs()
        match._read_in(match_file)
        indexes = match.get_indexes()

        for corp in corps:
            for sent in tqdm(corp):
                words = cls._filter(sent, indexes, bpe)
                for i in range(len(words)-2):
                    if words[i] and words[i+2]:
                        hash_val = cls._hash([sent[i], sent[i+2]], base)
                        if hash_val not in sgrams.keys():
                            sgrams[hash_val] = 0
                        sgrams[hash_val] += 1

        sgrams = { k:v for k,v in sgrams.items() if v >= min_freq}
        sgrams_list = GramsSorter.sort(sgrams, bpe_file, word_file, sorter=sorter)    
        sgrams_list = sgrams_list[:max_sgrams] if len(sgrams_list) > max_sgrams else sgrams_list

        sgram_vcb = Vocabs()
        skip_tok = Reseved.UNK_TOK[0]+Reseved.SPACE_TOK[0]
        for sgram_pair in sgrams_list:
            hash_val, _ = sgram_pair
            freq = sgrams[hash_val]
            if len(sgram_vcb) < max_sgrams:
                wlist = cls._unhash(hash_val, base)
                token = cls._make_token(wlist, bpe, len(sgram_vcb), freq, sep=skip_tok)
                sgram_vcb.append(token)
        log(f'Found {len(sgram_vcb)} skip-gram [ min_freq : {min_freq}, max_sgrams : {max_sgrams}]', 2)    
        return sgram_vcb, sgrams_list

    @staticmethod
    def _make_token(array, vocab, index, freq, sep=''):
        name = sep.join([vocab.table[x].name for x in array])
        kids = [vocab.table[x] for x in array]
        return Type(name, level=1, idx=index, freq=freq, kids=kids)

    @staticmethod
    def _unhash(hash_val, base):
        wlist = []
        while hash_val > 0:
            wlist.append(hash_val % base)
            hash_val = hash_val // base
        return wlist

    @staticmethod
    def _hash(array, base):
        hash_val = 0
        pbase = 1
        for x in array:
            hash_val += x*pbase
            pbase*=base
        return hash_val

    @staticmethod
    def _filter(sent, indexes, vocab):
        words = [0] * len(sent)
        for ix, tok in enumerate(sent):
            if tok in indexes:
                if ix == 0:
                    words[ix] = 1
                    continue
                ptok = vocab.table[sent[ix-1]]
                if ptok.name.endswith(Reseved.SPACE_TOK[0]):
                    words[ix] = 1
        return words

    @staticmethod
    def _cumulate(array):
        for i in range(len(array)-2, -1, -1):
            if array[i] != 0:
                array[i] += array[i+1]
        return array

class GramsSorter(object):
    
    sorters = ['freq', 'pmi', 'ngd', 'ngdf']

    @classmethod
    def sort(cls, ngrams:Dict[int,int], bpe_file:Filepath=None, 
                word_file:Filepath=None, sorter:str='freq'):
        sorter_func = cls.get_sorter(sorter)
        bpe_vcb = Vocabs(vocab_file=bpe_file) if bpe_file else None
        word_vcb = Vocabs(vocab_file=word_file) if word_file else None
        return sorter_func(ngrams, bpe_vcb, word_vcb)

    @classmethod
    def get_sorter(cls, sorter:str='freq'):
        if sorter not in cls.sorters:
            sorter = 'freq'
        if sorter == 'freq':
            return cls._sort_by_freq
        elif sorter == 'pmi':
            return cls._sort_by_pmi
        elif sorter == 'ngd':
            return cls._sort_by_ngd
        elif sorter == 'ngdf':
            return cls._sort_by_ngd_freq        
        return None

    @classmethod
    def _sort_by_freq(cls, ngrams:Dict[int,int], bpe_vcb=None, word_vcb=None):
        ngrams_list = [(key, val) for key,val in ngrams.items()]
        ngrams_list.sort(key=lambda x: x[1], reverse=True)
        return ngrams_list

    @classmethod
    def _sort_by_pmi(cls, ngrams:Dict[int,int], bpe_vcb, word_vcb):
        ntokens = sum([token.freq for token in word_vcb])
        base = len(bpe_vcb) + 1
        word_probs = { t.name: t.freq/ntokens for t in word_vcb }
        
        ngrams_list = []
        for hash_val, freq in ngrams.items():
            wlist = Grams._unhash(hash_val, base)
            wlist_probs = []
            for x in wlist:
                name = bpe_vcb.table[x].name
                idx = word_vcb.index(name[:-1])
                token = word_vcb.table[idx]
                wlist_probs.append( token.freq / ntokens )
            prob = freq / ntokens
            
            pmi_num = prob
            pmi_dec = 1
            for prob in wlist_probs:
                pmi_dec *= prob
            pmi = math.log(pmi_num/pmi_dec)
            ngrams_list.append((hash_val, pmi))
        ngrams_list.sort(key=lambda x: x[1], reverse=True)
        return ngrams_list

    # NGD was not functioning properly
    # -----------------------------------------------------------------------
    # @classmethod
    # def _sort_by_ngd(cls, ngrams:Dict[int,int], bpe_vcb, word_vcb):
    #     ngrams_list = []
    #     base = len(bpe_vcb)+1
    #     for hash_val in ngrams.keys():
    #         wlist = _unhash(hash_val, base)
    #         x, y = [ bpe_vcb.table[x].name[:-1] for x in wlist[:2] ]
    #         ngrams_list.append((hash_val, NGD(x,y)))
    #     ngrams_list.sort(key=lambda x: x[1])
    #     return ngrams_list

    @classmethod
    def _sort_by_ngd_sent():
        pass

    @classmethod
    def _sort_by_ngd_freq(cls, ngrams:Dict[int,int], bpe_vcb, word_vcb):
        ntokens = sum([token.freq for token in word_vcb])
        base = len(bpe_vcb) + 1
        word_probs = { t.name: t.freq/ntokens for t in word_vcb }
        ngrams_list = []
        for hash_val, freq in ngrams.items():
            wlist = Grams._unhash(hash_val, base)
            wlist_probs = []
            for x in wlist:
                name = bpe_vcb.table[x].name
                idx = word_vcb.index(name[:-1])
                token = word_vcb.table[idx]
                wlist_probs.append( token.freq / ntokens )
            prob = freq / ntokens
            ngd_num = max([math.log(x) for x in wlist_probs]) - math.log(prob)
            ngd_dec = math.log(ntokens) - min([math.log(x) for x in wlist_probs])
            ngd =  ngd_num / ngd_dec
            ngrams_list.append((hash_val, ngd))
        ngrams_list.sort(key=lambda x: x[1])
        return ngrams_list

