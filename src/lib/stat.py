import math
from collections import Counter
from pathlib import Path

from nlcodec import Reseved

class BaseFuncs(object):

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
    def _sort(val_dict, reverse:bool=False):
        val_list = [(hv, val) for hv,val in val_dict.items()]
        val_list.sort(key=lambda x: x[1], reverse=reverse)
        return val_list


class StatLib(BaseFuncs):

    @staticmethod
    def ngrams2hashes(ngrams, bpe):
        ngrams_freq = {}
        base = len(bpe) + 1
        for token in ngrams:
            name = token.name.replace(Reseved.SPACE_TOK[0], f'{Reseved.SPACE_TOK[0]} ' )
            parts = name.strip().split()
            indexes = [bpe.index(part) for part in parts]
            hash_val = BaseFuncs._hash(indexes, base)
            ngrams_freq[hash_val] = token.freq
        return ngrams_freq

    @staticmethod
    def ngrams2matches(ngrams, bpe):
        indexes = set()
        for token in ngrams:
            name = token.name.replace(Reseved.SPACE_TOK[0], f'{Reseved.SPACE_TOK[0]} ' )
            parts = name.strip().split()
            for part in parts:
                ix = bpe.index(part)
                if ix is not None:
                    indexes.add(ix)
        return indexes

    @staticmethod
    def ntokens(vocab):
        count = 0
        for token in vocab:
            count += token.freq
        return count

    @staticmethod
    def freqs2probs(freqs, total):
        return {k:v/total for k,v in freqs.items()}

    @staticmethod
    def vocabs2probs(vocab, total):
        probs = dict()
        for token in vocab:
            probs[token.idx] = token.freq / total
        return probs

    @staticmethod
    def calculate_metric(sorter, ngram_probs, bpe, words):
        assert sorter in ['pmi', 'ngdf', 'freq']
        if sorter == 'pmi':
            return StatLib.calculate_pmi(ngram_probs, bpe, words)
        elif sorter == 'ngdf':
            return StatLib.calculate_ngdf(ngram_probs, bpe, words)

    @staticmethod
    def calculate_pmi(ngram_probs, bpe, words):
        base = len(bpe)+1
        pmis = dict()
        
        ntokens = StatLib.ntokens(words)
        word_probs = StatLib.vocabs2probs(words, ntokens)
        
        for hv, prob in ngram_probs.items():
            wlist = BaseFuncs._unhash(hv, base)
            windexes = [words.index(bpe.table[x].name[:-1]) for x in wlist]
            wprobs = [word_probs[x] for x in windexes]
            pmi_dec = 1
            for pr in wprobs:
                pmi_dec *= pr
            pmi = -10000.0 if prob == 0 else math.log(prob / pmi_dec)
            pmis[hv] = pmi 
        return pmis

    @staticmethod
    def calculate_ngdf(ngram_probs, bpe, words):
        base = len(bpe)+1
        ngdfs = dict()

        ntokens = StatLib.ntokens(words)
        word_probs = StatLib.vocabs2probs(words, ntokens)

        for hv, prob in ngram_probs.items():
            wlist = BaseFuncs._unhash(hv, base)
            windexes = [words.index(bpe.table[x].name[:-1]) for x in wlist]
            wprobs = [word_probs[x] for x in windexes]
            lgprobs = [math.log(x) for x in wprobs]
            ngdf_num = max(lgprobs) - math.log(prob)
            ngdf_dec = math.log(ntokens) - min(lgprobs)
            ngdfs[hv] = ngdf_num / ngdf_dec
        return ngdfs
