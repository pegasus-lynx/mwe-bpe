import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

from nlcodec import Reseved, Type, learn_vocab, load_scheme, term_freq
from tqdm import tqdm

from .misc import Filepath, FileReader, FileWriter, get_now, log
from .vocabs import Vocabs
from .stat import BaseFuncs, StatLib

class GramsBase(BaseFuncs):

    @staticmethod
    def _make_token(array, vocab, index, freq, level=1, sep=''):
        name = sep.join([vocab.table[x].name for x in array])
        kids = [vocab.table[x] for x in array]
        return Type(name, level=level, idx=index, freq=freq, kids=kids)

    @staticmethod
    def _make_mask(sent, indexes):
        words = [0] * len(sent)
        schar = Reseved.SPACE_TOK[0]
        for ix, tok in enumerate(sent):
            if tok in indexes:
                if ix == 0:
                    words[ix] = 1
                    continue
                if sent[ix-1] in indexes:
                    words[ix] = 1
        return words

    @staticmethod
    def _cumulate(array):
        for i in range(len(array)-2, -1, -1):
            if array[i] != 0:
                array[i] += array[i+1]
        return array

    @staticmethod
    def _get_word_probs(word_list, bpe_vcb, word_vcb, ntokens:int=0):
        if ntokens == 0:
            ntokens = sum([token.freq for token in word_vcb])
        probs = []
        for x in word_list:
            name = bpe_vcb.table[x].name
            idx = word_vcb.index(name[:-1])
            token = word_vcb.table[idx]
            probs.append( token.freq / ntokens )
        return probs

    @staticmethod
    def _wlist_to_windexes(wlist, bpe, words, debug=False):
        indexes = []
        for x in wlist:
            name = bpe.table[x].name
            index = words.index(name[:-1])
            indexes.append(index)
        if debug:
            print(indexes)
        return indexes


class NGrams(GramsBase):

    sorters = ['freq', 'pmi', 'ngdf']

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

        # cnt = 0
        # for x in indexes:
        #     name = bpe.table[x].name
        #     if not name.endswith(Reseved.SPACE_TOK[0]):
        #         if cnt < 20:
        #             print(name, x)
        #         cnt += 1
        # print(cnt, len(indexes))

        for corp in corps:
            for sent in tqdm(corp):
                words = cls._make_mask(sent, indexes)
                cwords = cls._cumulate(words)
                for i, wl in enumerate(cwords):
                    if wl >= ngram:
                        hash_val = cls._hash(sent[i:i+ngram], base)
                        if hash_val not in ngrams.keys():
                            ngrams[hash_val] = 0
                        ngrams[hash_val] += 1 
        
        ngrams = { k:v for k,v in ngrams.items() if v >= min_freq}
        ngrams_list = cls.sort(ngrams, bpe_file, word_file, sorter=sorter)

        if max_ngrams >= 0: 
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

    @classmethod
    def sort(cls, ngrams:Dict[int, int], bpe_file:Filepath=None, 
                word_file:Filepath=None, sorter:str='freq'):
        sorter_func = GramsSorter.get_sorter(sorter)
        bpe_vcb = Vocabs(vocab_file=bpe_file) if bpe_file else None
        word_vcb = Vocabs(vocab_file=word_file) if word_file else None
        return sorter_func(ngrams, bpe_vcb, word_vcb)


class SkipGrams(GramsBase):

    sorters = [ 'freq', 'pmi', 'ngdf', 
                'pmi_skip', 'ngdf_skip' ]
    
    needs_ngrams = ['pmi_skip', 'ngdf_skip']
    
    # make shift skip gram : Considers A _ B cases only
    @classmethod
    def get_skipgrams(cls, corps:List[List[Union[str, int]]], match_file:Filepath, bpe_file:Filepath,
                        word_file:Filepath, min_freq:int=0, max_corr:float=0.5, min_center_words:int=0,
                        max_sgrams:int=0, sorter:str='freq'):
        sgrams, ngrams = dict(), dict()

        bpe = Vocabs(vocab_file=bpe_file)
        base = len(bpe)+1
        max_sgrams = len(bpe) if max_sgrams == 0 else max_sgrams

        match = Vocabs()
        match._read_in(match_file)
        indexes = match.get_indexes()

        for corp in corps:
            for sent in tqdm(corp):
                words = cls._make_mask(sent, indexes)
                cwords = cls._cumulate(words)
                for i in range(len(cwords)):
                    if cwords[i] >= 3:
                        hash_val = cls._hash([sent[i], sent[i+2]], base)
                        if hash_val not in sgrams.keys():
                            sgrams[hash_val] = dict()
                        if sent[i+1] not in sgrams[hash_val].keys():
                            sgrams[hash_val][sent[i+1]] = 0
                        sgrams[hash_val][sent[i+1]] += 1

        sgrams = cls._filter(sgrams, min_freq=min_freq, max_corr=max_corr, min_words=min_center_words)
        
        if sorter in cls.needs_ngrams:
            nsorter = sorter.split('_')[0]
            ngrams = NGrams.get_ngrams(corps, match_file, bpe_file, 
                                        word_file, 3, min_freq=min_freq,
                                        max_ngrams=-1, sorter=nsorter)

        sgrams_list = cls.sort(sgrams, bpe_file, word_file, ngrams=ngrams, sorter=sorter)    
        sgrams_list = sgrams_list[:max_sgrams] if len(sgrams_list) > max_sgrams else sgrams_list

        sgrams_vcb = Vocabs()
        skip_tok = '*'
        for sgram_pair in sgrams_list:
            hash_val, _ = sgram_pair
            freq = sum(sgrams[hash_val].values())
            if len(sgrams_vcb) < max_sgrams:
                wlist = cls._unhash(hash_val, base)
                token = cls._make_token(wlist, bpe, len(sgrams_vcb), freq, level=3, sep=skip_tok)
                sgrams_vcb.append(token)
        log(f'Found {len(sgrams_vcb)} skip-gram [ min_freq : {min_freq}, max_sgrams : {max_sgrams}]', 2)    
        return sgrams_vcb, sgrams_list

    @staticmethod
    def _filter(sgrams, min_freq:int=0, max_corr:float=1.0, min_words:int=0):
        filtered_sgrams = dict()
        for k, v in sgrams.items():
            freq = sum(v.values())
            flag = freq >= min_freq
            flag = flag and (max(v.values()) / freq) <= max_corr
            flag = flag and len(v.values()) >= min_words
            if flag:
                filtered_sgrams[k] = v
        return filtered_sgrams

    @classmethod
    def sort(cls, sgrams:Dict[int, Dict[int,int]], bpe_file:Filepath=None, 
                    word_file:Filepath=None, ngrams=None, sorter:str='freq'):
        sorter_func = GramsSorter.get_sorter(sorter)
        bpe_vcb = Vocabs(vocab_file=bpe_file) if bpe_file else None
        word_vcb = Vocabs(vocab_file=word_file) if word_file else None
        sgrams = {k:sum(v.values()) for k,v in sgrams.items()}
        return sorter_func(sgrams, bpe_vcb, word_vcb)

        
class GramsSorter(GramsBase):
    
    all_sorters = [ 'freq', 'pmi', 'ngdf', 
                    'pmi_skip', 'ngdf_skip' ]

    @classmethod
    def get_sorter(cls, sorter:str='freq'):
        sorter_funcs = {
            'freq': cls.sort_by_freq,
            'pmi' : cls.sort_by_pmi,
            'ngdf': cls.sort_by_ngdf,
            'pmi_skip' : cls.sort_by_pmi_skip,
            'ngdf_skip': cls.sort_by_ngdf_skip
        }
        
        if sorter in sorter_funcs.keys():
            return sorter_funcs[sorter]
        return ValueError(f'Sorter {sorter} not in the list')

    @classmethod
    def sort_by_freq(cls, ngrams:Dict[int,int], bpe_vcb=None, word_vcb=None):
        ngrams_list = [(key, val) for key,val in ngrams.items()]
        ngrams_list.sort(key=lambda x: x[1], reverse=True)
        return ngrams_list

    @classmethod
    def sort_by_pmi(cls, ngrams:Dict[int,int], bpe_vcb, word_vcb):
        ntokens = sum([token.freq for token in word_vcb])
        base = len(bpe_vcb) + 1
        word_probs = { t.name: t.freq/ntokens for t in word_vcb }
        
        ngrams_list = []
        for hash_val, freq in ngrams.items():
            wlist = cls._unhash(hash_val, base)
            wlist_probs = cls._get_word_probs(wlist, bpe_vcb, word_vcb, ntokens)
            prob = freq / ntokens
            pmi_num, pmi_dec = prob, 1
            for prob in wlist_probs:
                pmi_dec *= prob
            pmi = math.log(pmi_num/pmi_dec)
            ngrams_list.append((hash_val, pmi))
        ngrams_list.sort(key=lambda x: x[1], reverse=True)
        return ngrams_list

    @classmethod
    def sort_by_ngdf(cls, ngrams:Dict[int,int], bpe_vcb, word_vcb):
        ntokens = sum([token.freq for token in word_vcb])
        base = len(bpe_vcb) + 1
        word_probs = { t.name: t.freq/ntokens for t in word_vcb }
        ngrams_list = []
        for hash_val, freq in ngrams.items():
            wlist = cls._unhash(hash_val, base)
            wlist_probs = cls._get_word_probs(wlist, bpe_vcb, word_vcb, ntokens)            
            prob = freq / ntokens
            ngd_num = max([math.log(x) for x in wlist_probs]) - math.log(prob)
            ngd_dec = math.log(ntokens) - min([math.log(x) for x in wlist_probs])
            ngd =  ngd_num / ngd_dec
            ngrams_list.append((hash_val, ngd))
        ngrams_list.sort(key=lambda x: x[1])
        return ngrams_list

    @classmethod
    def sort_by_ngdf_skip(cls):
        pass

    @classmethod
    def sort_by_pmi_skip(cls):
        pass


class GramsNormalizer(GramsBase):

    @classmethod
    def min_normalize(cls, trigram_vals, bigram_vals, bpe):
        base = len(bpe)+1
        norm_vals = dict()
        for hv, val in trigram_vals.items():
            wlist = cls._unhash(hv, base)
            bilist = [cls._hash(wlist[i:i+2], base) for i in range(len(wlist)-1)]
            bivals = [bigram_vals[x] for x in bilist]
            norm_vals[hv] = min(bivals)
        return norm_vals

    @classmethod
    def avg_normalize(cls, trigram_vals, bigram_vals, bpe):
        base = len(bpe)+1
        norm_vals = dict()
        for hv, val in trigram_vals.items():
            wlist = cls._unhash(hv, base)
            bilist = [cls._hash(wlist[i:i+2], base) for i in range(len(wlist)-1)]
            bivals = [bigram_vals[x] for x in bilist]
            norm_vals[hv] = sum(bivals) / len(bivals)
        return norm_vals


    @classmethod
    def bifreq_normalize(cls, trigram_vals, bigram_freqs, bpe):
        base = len(bpe)+1
        norm_vals = dict()
        for hv, val in trigram_vals.items():
            wlist = cls._unhash(hv, base)
            bilist = [cls._hash(wlist[i:i+2], base) for i in range(len(wlist)-1)]
            bifreqs = [bigram_freqs[x] for x in bilist]
            norm_vals[hv] = val - math.log(1 + max(bifreqs)-min(bifreqs))
        return norm_vals

    @classmethod
    def freq_normalize(cls, trigram_vals, bpe, words):
        base = len(bpe)+1
        norm_vals = dict()
        for hv, val in trigram_vals.items():
            wlist = cls._unhash(hv, base)
            windexes = cls._wlist_to_windexes(wlist, bpe, words)
            try:
                wfreqs = [words.table[x].freq for x in windexes]
            except Exception as e:
                print([bpe.table[x].name for x in wlist])
                cls._wlist_to_windexes(wlist, bpe, words, True)
                raise(e)
            norm_vals[hv] = val - math.log(1 + (max(wfreqs)-min(wfreqs)))
        return norm_vals