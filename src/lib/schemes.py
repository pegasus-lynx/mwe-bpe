import collections as coll
from tqdm import tqdm
import functools as fn
from nlcodec import Type, Reseved, CharScheme, WordScheme, BPEScheme
from typing import Iterator, Optional, Dict, Tuple, List, Union

NGRAM_MIN_FREQ = 100
SKIPGRAM_MIN_FREQ = 100

class PMIFuncs():

    space_tok = Reseved.SPACE_TOK[0]
    ngram_variants = ['naive_pmi', 'avg_pmi', 'min_pmi']
    sgram_variants = ['skip_pmi']

    @classmethod
    def get_pmis(cls, table:List['Type'], nterms:int,
            nlines:int, bigram_freqs:Dict[str,int]=None, 
            pmi_variant:str='naive_pmi') -> List[Tuple['Type',float]]:
        sorter_func = cls.get_pmi_func(pmi_variant)
        table_pmis = []
        for token in table:
            pmi = sorter_func(token, nterms, nlines, bigram_freqs)
            table_pmis.append((token, pmi))
        return table_pmis

    @classmethod
    def get_pmi_func(cls, variant:str):
        if variant == 'naive_pmi':
            return cls.naive_pmi
        elif variant == 'avg_pmi':
            return cls.avg_pmi
        elif variant == 'min_pmi':
            return cls.min_pmi
        elif variant == 'skip_pmi':
            return cls.skip_pmi
        return ValueError(f'Variant {variant} not available. \
                            Options : naive_pmi, avg_pmi, min_pmi')

    @classmethod
    def skip_pmi(cls, tok:'Type', nterms:int, nlines:int, *args) -> float:
        ngram = len(tok.kids)
        word_probs = [k/nterms for k in tok.kids if k > 0]
        sgram_prob = tok.freq / (nterms - (nlines*(ngram-1)))
        return cls._naive_pmi(sgram_prob, word_probs)

    @classmethod
    def naive_pmi(cls, tok:'Type', nterms:int, nlines:int, *args) -> float:
        ngram = len(tok.kids)
        word_probs = [ k/nterms for k in tok.kids ]
        ngram_prob = tok.freq / (nterms - (nlines*(ngram-1)))
        return cls._naive_pmi(ngram_prob, word_probs)

    @classmethod
    def avg_pmi(cls, tok:'Type', nterms:int, nlines:int, 
                bigram_freqs:Dict[str,int]) -> float:
        bigram_probs = { name : freq/(nterms-nlines) for name,freq in bigram_freqs.items()}
        pmis_list = cls._get_bigram_pmis(tok, nterms, bigram_probs)
        return fn.reduce(lambda a,b: a+b, pmis_list) / len(pmis_list)

    @classmethod
    def min_pmi(cls, tok:'Type', nterms:int, nlines:int,
                bigram_freqs:Dict[str,int]) -> float:
        bigram_probs = { name : freq/(nterms-nlines) for name,freq in bigram_freqs.items()}
        pmis_list = cls._get_bigram_pmis(tok, nterms, bigram_probs)
        return min(pmis_list)

    @staticmethod
    def _naive_pmi(ngram_prob:float, word_probs:List[float]) -> float:
        pmi_num = ngram_prob
        pmi_dec = fn.reduce(lambda a,b: a*b, word_probs)
        return pmi_num / pmi_dec

    @classmethod
    def _get_bigram_pmis(cls, token:'Type', nterms:int, 
                        bigram_probs:Dict[str,float]) -> List[float]:
        parts = token.name.replace(cls.space_tok, f'{cls.space_tok} ').split()[:-1]
        bigrams = [''.join(parts[i:i+2]) for i in range(len(parts)-1)]
        word_probs = [ k/nterms for k in token.kids]
        pmis_list = []
        for x, bigram in enumerate(bigrams):
            prob = bigram_probs[bigram]
            pmi = cls._naive_pmi(prob, word_probs[x:x+2])
            pmis_list.append(pmi)
        return pmis_list


class NgramScheme(BPEScheme):

    def __init__(self, table:List['Type']):
        super().__init__(table=table, invertible=False)

    def decode(self, seq:List[int]) -> str:
        pieces = [self.table[x].name for x in seq]
        return self.decode_str(pieces)

    def decode_str(self, seq:List[str]) -> str:
        return ''.join(seq).replace(self.space_char, ' ').strip()

    @classmethod
    def ngram_frequencies(cls, data: Iterator[str], 
                            ngram:int) -> Dict[str,int]:
        ngram_freqs = coll.Counter()
        for line in tqdm(data, mininterval=1):
            words = WordScheme.encode_str(line)
            ngrams = [ cls.space_char.join([*words[i:i+ngram], '']) 
                        for i in range(len(words)-ngram)]
            ngram_freqs.update(ngrams)
        return ngram_freqs

    @classmethod
    def sorted_ngrams(cls, ngram_freqs:Dict[str,int], 
        term_freqs:Dict[str, int], nlines:int, metric:str, bigram_freqs=None, 
        min_freq:int=NGRAM_MIN_FREQ) -> List[Tuple['Type', Union[int,float]]]:
        
        nterms = sum(term_freqs.values())
        ngrams_list = []
        for name, freq in ngram_freqs.items():
            if freq < min_freq:
                continue
            words = name.split(cls.space_char)[:-1]
            word_freqs = [term_freqs[word] for word in words]
            ngrams_list.append(Type(name, freq=freq, idx=0, 
                                    level=3, kids=word_freqs))

        if metric == 'freq':
            sorted_list = [(x,x.freq) for x in ngrams_list]
        else:
            sorted_list = PMIFuncs.get_pmis(ngrams_list, nterms, nlines, 
                                            bigrams=bigram_freqs,
                                            pmi_variant=metric)
        sorted_list.sort(key=lambda x: x[1], reverse=True)
        return sorted_list
            
    @classmethod
    def filtered_ngrams(cls, ngrams_list:List[Tuple['Type',Union[int,float]]],
            bpes:List['Type']) -> List[Tuple['Type', Union[int,float]]]:
        
        rev_idx = {t.name:t.idx for t in bpes}
        words = set([t.name for t in bpes if t.name.endswith(cls.space_char)])
        filtered = []
        for pair in ngrams_list:
            tok, _ = pair
            parts = tok.name.replace(cls.space_char, f'{cls.space_char} ').split()[:-1]
            not_word = [ part not in words for part in parts]
            if not any(not_word):
                tok.kids = [bpes[rev_idx[x]] for x in parts]
                filtered.append(pair)
        return filtered

    @classmethod
    def learn(cls, data:Iterator[str], vocab_size:int=0, ngrams:List[int]=None, 
            max_ngrams:int=0, merge_func:str='freq', toks_list:List[int]=[],
            min_freq:int=NGRAM_MIN_FREQ) -> List['Type']:

        assert ngrams is not None
        assert merge_func == 'freq' or merge_func in PMIFuncs.ngram_variants
        assert len(toks_list) == len(ngrams) or max_ngrams > 0
        
        base = BPEScheme.learn(data, vocab_size)
        term_freqs, nlines = WordScheme.term_frequencies(data)
        ngrams_lists = {}
        
        bigram_freqs = cls.ngram_frequencies(data, 2)
        for ng in ngrams:
            ngram_freqs = cls.ngram_frequencies(data, ng)
            sorted_ngrams = cls.sorted_ngrams(ngram_freqs, term_freqs,
                                    nlines, merge_func, 
                                    bigram_freqs=bigram_freqs,
                                    min_freq=min_freq)
            ngrams_lists[ng] = cls.filtered_ngrams(sorted_ngrams, base)
        
        # Currently equal number of ngrams from each list are included
        # or else provided by the user themselves
        if len(toks_list) == 0:
            toks_list = [max_ngrams // len(ngrams)] * len(ngrams)
        assert vocab_size > sum(toks_list)
        
        unk_idx = Reseved.UNK_IDX
        base_len = vocab_size - sum(toks_list)

        vocab = base[:base_len]
        for ng, ntoks in zip(ngrams, toks_list):
            trimmed_list = ngrams_lists[ng][:ntoks]
            for pair in trimmed_list:
                tok, _ = pair
                # Doubt : How to add the ngrams ??
                # 1. Use a global list of ngrams irrespective of the words in the vocabs
                # 2. Consider words in the base vocab (all) [current]
                # 3. Consider words in the base vocab (non-replaced)
                # if any([t.idx >= base_len for t in tok.kids]):
                tok.kids = [t if t.idx < base_len else base[unk_idx] for t in tok.kids]
                vocab.append(tok)
        return vocab


# --------------------------------------------------------------------------------------------------- #

SCHEMES_REGISTRY = {
    'char': CharScheme,
    'word': WordScheme,
    'bpe' : BPEScheme,
    # 'mwe' : MWEScheme,
    'ngram': NgramScheme,
    # 'skipgram': SkipScheme
}

def load_scheme(pieces:str):
    if pieces in SCHEMES_REGISTRY.keys():
        return SCHEMES_REGISTRY[pieces]
    return ValueError(f'Piece {pieces} not available. \
                Choices : [ char, word, bpe, ngram, skipgram, mwe ]')