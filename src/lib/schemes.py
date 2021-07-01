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


class SkipScheme(BPEScheme):

    skip_tok = '▂' # U+2582 ??
    
    def __init__(self, table:List['Type']):
        super().__init__(table=table)
        self.root = self.make_vocab_prefix_trie(self.table)
        assert self.unk_idx

    def encode_str(self, line:str) -> List[str]:
        seq = self.space_char.join(line.strip().split()) + self.space_char
        res: List[int] = []

        data_node, data_idx = None, -1
        prev_node, idx = self.root, 0

        ahead_pairs = []
        ahead_node, ahead_idx = self.root, 0

        pos = 0
        while pos < len(seq):
            if prev_node.has_data:
                data_node = prev_node
                data_idx = idx
            if self.skip_tok in prev_node.kids:
                pass

    @classmethod
    def decode_str(cls, seq:List[str]) -> str:
        pass

    @classmethod
    def skipgram_frequencies(cls, data:Iterator[str], 
                    sgram:Tuple[int,int]) -> Dict[str, Dict[str,int]]:
        sgram_freqs = dict()
        _, skip = sgram
        skip_str = cls.space_char.join([cls.skip_tok]*skip) + cls.space_char
        for line in tqdm(data, mininterval=1):
            words = WordScheme.encode_str(line)
            nwords = len(words)
            if nwords > skip+1:
                words = [ f'{word}{cls.space_char}' for word in words ]
                for i in range(nwords-(skip+1)):
                    name = f'{words[i]}{skip_str}{words[i+skip+1]}'
                    if name not in sgram_freqs.keys():
                        sgram_freqs[name] = coll.Counter()
                    sgram_freqs[name].update(''.join(words[i+1:i+skip+1]))
        return sgram_freqs

    @classmethod
    def sorted_sgrams(cls, sgram_freqs:Dict[str, Dict[str,int]],
            term_freqs:Dict[str,int], nlines:int, metric:str,
            min_freq:int=0) -> List[Tuple['Type', Union[int,float], Tuple[int,float]]]:
        nterms = sum(term_freqs.values())
        sgrams_list = []
        sgrams_stats = {}
        for name, instances in sgram_freqs.items():
            freq = sum(instances.values())
            if freq < min_freq:
                continue
            words = name.split(cls.space_char)[:-1]
            word_freqs = [0 if cls.skip_tok in word else term_freqs[word]
                            for word in words]
            ninstances = len(instances.keys())
            max_prob =  max([val/freq for val in instances.values()])
            sgrams_list.append(Type(name, freq=freq, idx=0,
                                    level=3, kids=word_freqs))
            sgrams_stats[name] = (ninstances, max_prob)

        if metric == 'freq':
            sorted_list = [(x,x.freq) for x in sgrams_list]
        else:
            sorted_list = PMIFuncs.get_pmis(sgrams_list, 
                                            nterms, nlines,  
                                            pmi_variant=metric)
        sorted_list.sort(key=lambda x: x[1], reverse=True)
        sorted_list = [ (tok, val, sgrams_stats[tok.name]) for tok, val in sorted_list]
        return sorted_list

    @classmethod
    def filtered_sgrams(cls, sgrams_list:List[Tuple['Type', Union[int,float], 
            Tuple[int,float]]], bpes:List['Type'], max_instance_prob:float=1.0,
            min_instances:int=0) -> List[Tuple['Type', Union[int,float], 
            Tuple[int,float]]]:
        rev_idx = {t.name:t.idx for t in bpes}
        words = set([t.name for t in bpes if t.name.endswith(cls.space_char)])
        filtered = []
        for trp in sgrams_list:
            tok, _, stats = trp
            ninstances, max_prob = stats
            if ninstances < min_instances or max_prob > max_instance_prob:
                continue
            parts = tok.name.replace(cls.space_char, 
                                    f'{cls.space_char} ').split()[:-1]
            not_word = [ part not in words or cls.skip_tok in part 
                                for part in parts]
            if not any(not_word):
                tok.kids = [bpes[rev_idx[x]] for x in parts]
                filtered.append(trp)
        return filtered

    @classmethod
    def learn(cls, data:Iterator[str], vocab_size:int=0, 
                sgrams:List[Tuple[int,int]]=None, max_sgrams:int=0, 
                merge_func:str='freq', toks_list:List[int]=[],
                min_freq:int=SKIPGRAM_MIN_FREQ, min_instances:int=15, 
                max_instance_prob:float=0.1) -> List['Type']:
        assert sgrams is not None
        assert merge_func == 'freq' or merge_func in PMIFuncs.sgram_variants
        assert max_sgrams > 0 or len(toks_list) == len(sgrams)

        ## Currently support for n-skip-2 grams only. Need to discuss
        #  about this with AVT

        base = BPEScheme.learn(data, vocab_size)
        term_freqs, nlines = WordScheme.term_frequencies(data)
        sgrams_list = {}

        for sg in sgrams:
            sgram_freqs = cls.skipgram_frequencies(data, sg)
            sorted_sgrams = cls.sorted_sgrams(sgram_freqs, term_freqs,
                                            nlines, merge_func, min_freq)
            sgrams_list[sg] = cls.filtered_sgrams(sorted_sgrams, base,
                                                    max_instance_prob, 
                                                    min_instances)

        if len(toks_list) == 0:
            toks_list = [max_sgrams // len(sgrams)] * len(sgrams)
        assert vocab_size > sum(toks_list)

        unk_idx = Reseved.UNK_IDX
        base_len = vocab_size - sum(toks_list)

        vocab = base[:base_len]
        for sg, stoks in zip(sgrams, toks_list):
            trimmed_list = sgrams_list[sg][:stoks]
            for pair in trimmed_list:
                tok, _ = pair
                tok.kids = [t if t.idx < base_len else base[unk_idx] for t in tok.kids]
                vocab.append(tok)
        return vocab


class MWEScheme(BPEScheme):
    pass

# --------------------------------------------------------------------------- #

SCHEMES_REGISTRY = {
    'char': CharScheme,
    'word': WordScheme,
    'bpe' : BPEScheme,
    'mwe' : MWEScheme,
    'ngram': NgramScheme,
    'skipgram': SkipScheme
}

def load_scheme(pieces:str):
    if pieces in SCHEMES_REGISTRY.keys():
        return SCHEMES_REGISTRY[pieces]
    return ValueError(f'Piece {pieces} not available. \
                Choices : [ char, word, bpe, ngram, skipgram, mwe ]')