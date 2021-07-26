import collections as coll
from math import log
from logging import makeLogRecord
from tqdm import tqdm
import functools as fn
from nlcodec import Type, Reseved, CharScheme, WordScheme, BPEScheme
from nlcodec import DEF_CHAR_COVERAGE, DEF_MIN_CO_EV, DEF_WORD_MIN_FREQ, DEF_CHAR_MIN_FREQ
from nlcodec.bpe import BPELearn
from typing import Iterator, Optional, Dict, Tuple, List, Union, Set

## NGRAM and SKIPGRAM to be used for more customizability
NGRAM_MIN_FREQ = 100
SKIPGRAM_MIN_FREQ = 100
MWE_MIN_FREQ = 100

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
        return log(pmi_num / pmi_dec)

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
        super().__init__(table=table)

    def decode(self, seq:List[int]) -> str:
        pieces = [self.table[x].name for x in seq]
        return self.decode_str(pieces)

    def decode_str(self, seq:List[str]) -> str:
        return ''.join(seq).replace(self.space_char, ' ').strip()

    @classmethod
    def get_bpe_words(cls, bpe, words_set):
        available_words = set()
        for token in bpe:
            name = token.name
            if name.endswith(Reseved.SPACE_TOK[0]):
                if name[:-1] in words_set:
                    available_words.add(name)
        return available_words
        
    @classmethod
    def ngram_frequencies(cls, data: Iterator[str], 
                            ngram:int) -> Dict[str,int]:
        ngram_freqs = coll.Counter()
        for line in tqdm(data, mininterval=1):
            words = WordScheme.encode_str(line)
            ngrams = [ cls.space_char.join([*words[i:i+ngram], '']) 
                        for i in range((len(words)+1)-ngram)]
            ngram_freqs.update(ngrams)
        return ngram_freqs

    @classmethod
    def sorted_ngrams(cls, ngram_freqs:Dict[str,int], 
        term_freqs:Dict[str, int], nlines:int, metric:str, bigram_freqs=None, 
        min_freq:int=MWE_MIN_FREQ) -> List[Tuple['Type', Union[int,float]]]:
        
        nterms = sum(term_freqs.values())
        ngrams_list = []
        for name, freq in ngram_freqs.items():
            # if name == 'aya▁pehn▁':
            #     print(name, freq)
            if freq >= min_freq:
                words = name.split(cls.space_char)[:-1]
                word_freqs = [term_freqs[word] for word in words]
                ngrams_list.append(Type(name, freq=freq, idx=0, 
                                    level=1, kids=word_freqs))

        if metric == 'freq':
            sorted_list = [(x,x.freq) for x in ngrams_list]
        else:
            sorted_list = PMIFuncs.get_pmis(ngrams_list, nterms, nlines, 
                                            bigram_freqs=bigram_freqs,
                                            pmi_variant=metric)
        sorted_list.sort(key=lambda x: x[1], reverse=True)
        return sorted_list
            
    @classmethod
    def filtered_ngrams(cls, 
            ngrams_list:List[Tuple['Type',Union[int,float]]],
            bpes:List['Type'],
            words_set:Set[str]) -> List[Tuple['Type', Union[int,float]]]:
        
        rev_idx = {t.name:t.idx for t in bpes}
        filtered = []
        for pair in ngrams_list:
            tok, val = pair
            if tok.name == 'aya▁pehn▁':
                print(tok, val)
            parts = tok.name.replace(cls.space_char, f'{cls.space_char} ').split()
            not_word = [ part not in words_set for part in parts]
            if not any(not_word):
                kids = [bpes[rev_idx[x]] for x in parts]
                tok = Type(tok.name, tok.level, tok.idx, tok.freq, kids=kids)
                filtered.append((tok, val))
        return filtered

    @classmethod
    def get_ngrams_lists(cls, data:Iterator[str], ngrams:List[int]=None, 
                        sorter_func:str='freq', min_freq:int=MWE_MIN_FREQ, 
                        vocab_size:int=0, bpe_vocab:List['Type']=None):
        assert ngrams is not None
        assert vocab_size != 0 or bpe_vocab is not None
        assert sorter_func == 'freq' or sorter_func in PMIFuncs.ngram_variants

        if bpe_vocab is None:
            bpe_vocab = BPEScheme.learn(data, vocab_size)

        term_freqs, nlines = WordScheme.term_frequencies(data)
        all_words = set(term_freqs.keys())
        bpe_words = cls.get_bpe_words(bpe_vocab, all_words)

        ngrams_lists = {}
        
        bigram_freqs = cls.ngram_frequencies(data, 2)
        for ng in ngrams:
            ngram_freqs = cls.ngram_frequencies(data, ng)
            sorted_ngrams = cls.sorted_ngrams(ngram_freqs, term_freqs,
                                    nlines, sorter_func, 
                                    bigram_freqs=bigram_freqs,
                                    min_freq=min_freq)
            ngrams_lists[ng] = cls.filtered_ngrams(sorted_ngrams, 
                                                bpe_vocab, bpe_words)
        return ngrams_lists, bpe_vocab

    @classmethod
    def merge_lists(cls, base, lists, vocab_size, grams, toks_list):
        unk_idx = Reseved.UNK_IDX
        base_len = vocab_size - sum(toks_list)

        vocab = base[:base_len]
        for gram, toks in zip(grams, toks_list):
            trimmed_list =lists[gram][:toks]
            for pair in trimmed_list:
                tok = pair[0]
                # Doubt : How to add the ngrams ??
                # 1. Use a global list of ngrams irrespective of the words in the vocabs
                # 2. Consider words in the base vocab (all) [current]
                # 3. Consider words in the base vocab (non-replaced)
                # if any([t.idx >= base_len for t in tok.kids]):
                kids = [t if t.idx < base_len else base[unk_idx] for t in tok.kids]
                vocab.append(Type(tok.name, tok.level, len(vocab), 
                                tok.freq, kids))
        return vocab

    @classmethod
    def learn(cls, data:Iterator[str], vocab_size:int=0, ngrams:List[int]=None, 
            max_ngrams:int=0, ngram_sorter:str='freq', toks_list:List[int]=[],
            min_freq:int=MWE_MIN_FREQ, **kwargs) -> List['Type']:

        assert ngrams is not None
        assert len(toks_list) == len(ngrams) or max_ngrams > 0
        
        base = BPEScheme.learn(data, vocab_size)
        ngrams_lists, _ = cls.get_ngrams_lists(data, ngrams, ngram_sorter,
                                            min_freq, bpe_vocab=base)

        # Currently equal number of ngrams from each list are included
        # or else provided by the user themselves
        if len(toks_list) == 0:
            toks_list = [max_ngrams // len(ngrams)] * len(ngrams)
        assert vocab_size > sum(toks_list)
        
        return cls.merge_lists(base, ngrams_lists, vocab_size, 
                                ngrams, toks_list)


class SkipScheme(BPEScheme):

    PLACE_TOK = '▂', 6 # U+2582 ??
    SKIP_TOK = '<skp>', 7
    TOKS = [PLACE_TOK, SKIP_TOK]
    
    skip_char = PLACE_TOK[0]
    skip_tok = SKIP_TOK[0]

    count = 1

    hash_prime = 9973 # prime number to hash the list
    
    def __init__(self, table:List['Type']):
        super().__init__(table=table)
        self.root = self.make_vocab_prefix_trie(self.table)
        assert self.unk_idx

    def encode(self, line: str, split_ratio: float = 0.) -> List[int]:
        pieces = self.encode_str(line, split_ratio=split_ratio)
        return [self.str_to_idx.get(piece, self.unk_idx) for piece in pieces]

    def encode_str(self, line:str, split_ratio=None) -> List[str]:
        seq = self.space_char.join(line.strip().split()) + self.space_char
        res: List[int] = []

        def _set_default():
            return None, False, True

        back_pairs = []
        data_node, data_idx = None, -1
        prev_node, idx = self.root, 0
        tokens, is_skip, check_skip = _set_default()

        # Thought of a little tweak to make it work for a_b_c tokens.
        # Setting global tokens to some value. And naming while token to 
        # something else.

        while seq and idx < len(seq)+1:

            # try:
            #     print(data_node.data.name, data_idx)
            # except Exception as e:
            #     print('None', data_idx)

            # try:
            #     print(prev_node.data.name, idx)
            # except Exception as e:
            #     print('None', idx)

            # print()

            if prev_node.has_data:
                data_node, data_idx = prev_node, idx

            if self.skip_char in prev_node.kids and check_skip and idx != 0:
                # print('Checking Skip ...')
                tokens, ahead_pair = self.get_skips(seq, idx, prev_node)
                if tokens is not None:
                    # print('Tokens : ', tokens)
                    back_pair = (prev_node, idx)
                    back_pairs.append(back_pair)
                    prev_node, idx = ahead_pair
                    is_skip = True
                    check_skip = False
                else:
                    check_skip = False
            else:
                if idx < len(seq) and seq[idx] in prev_node.kids:
                    prev_node = prev_node.kids[seq[idx]]
                    idx += 1
                else:
                    if data_node:
                        res.append(data_node.data.idx)
                        seq = seq[data_idx:]
                        if is_skip:
                            res.extend(tokens)
                            is_skip = False
                    else:
                        res.append(self.unk_idx)
                        seq = seq[1:]

                    back_pairs = []
                    prev_node, idx = self.root, 0
                    data_node, data_idx = None, -1
                    tokens, is_skip, check_skip = _set_default()
        
        return [self.table[idx].name for idx in res]
                    
    def check_skippable(self, seq, pos, curr_node):
        # tseq = self.skip_char + seq[pos:]
        tseq = seq[pos:]
        # print('Skippable :', tseq)
        ix = 0
        while ix < len(tseq):
            if tseq[ix] in curr_node.kids:
                # print(f'In : {tseq[ix]}')
                curr_node = curr_node.kids[tseq[ix]]
                ix += 1
                if curr_node.has_data:
                    return True
            else:
                return False        
        return False
   
    def get_skips(self, seq, pos, node):
        # print('Getting skips :')
        next_idxs = []
        next_tokens = []
        prev_node, idx = self.root, pos
        while idx < len(seq):
            if seq[idx] in prev_node.kids:
                # print(seq[idx], prev_node.data.name if prev_node.data is not None else 'root')
                prev_node = prev_node.kids[seq[idx]]
                idx += 1
                if prev_node.has_data:
                    next_idxs.append(idx)
                    next_tokens.append(prev_node)
            else:
                break

        next_node = node.kids[self.skip_char]
        next_idxs.reverse()
        next_tokens.reverse()

        if self.skip_char in next_node.kids:
            # print('Checking for next skip tokens ...')
            for token, next_idx in zip(next_tokens, next_idxs):
                tokens, ahead_pair = self.get_skips(seq, next_idx, next_node)
                if tokens is not None:
                    tokens.insert(0, token.data.idx)
                    return tokens, ahead_pair

        for token, next_idx in zip(next_tokens, next_idxs):
            if self.check_skippable(seq, next_idx, next_node):
                return [token.data.idx, self.SKIP_TOK[1]], (next_node, next_idx)

        return None, None

    @classmethod
    def decode_str(cls, seq:List[str]) -> str:

        ordered_seq = [None]*len(seq)
        pos = 0
        skipped_pos = []
        for tok in seq:
            # print(ordered_seq)
            # print(pos)
            if tok == cls.skip_tok:
                continue
            nskips = coll.Counter(tok).get(cls.skip_char,0)
            if nskips:
                xtok = tok.replace(cls.skip_char, f'{cls.skip_char} ')
                xtok = xtok.replace(cls.space_char, f'{cls.space_char} ')
                parts = xtok.split()
                # print(parts)
                for ix, part in enumerate(parts):
                    if part != cls.skip_char:
                        ordered_seq[pos+ix] = part
                    else:
                        skipped_pos.append(pos+ix)
                pos += len(parts)
            else:
                if len(skipped_pos) != 0:
                    cpos = skipped_pos[0]
                    skipped_pos = skipped_pos[1:]
                    ordered_seq[cpos] = tok
                else:
                    ordered_seq[pos] = tok
                    pos += 1

        return ''.join(ordered_seq).replace(cls.space_char, ' ')

        # line, parts = [], []
        # nskips = 0
        # for tok in seq:
        #     if tok == cls.skip_char:
        #         continue
        #     if cls.skip_char in tok:
        #         parts = tok.split(cls.skip_char)
        #         for part in parts:
        #             if tok == cls.skip_char:
        #                 nskips -= 1
        #         continue
        #     if nskips:
        #         for ix, part in enumerate(parts):
        #             if part == cls.skip_char:
        #                 parts[ix] = tok
        #                 break
        #         if not nskips:
        #             line.extend(parts)
        #             parts = []
        #     else:
        #         line.append(tok)
        # if len(parts) != 0:
        #     line.extend(parts)
        #     parts = []
        # return ''.join(line).replace(cls.space_char, ' ').strip()

    @classmethod
    def skipgram_frequencies(cls, data:Iterator[str], 
                    sgram:Tuple[int,int]) -> Dict[str, Dict[str,int]]:
        sgram_freqs = dict()
        _, skip = sgram
        # skip_str = cls.space_char.join([cls.skip_char]*skip) + cls.space_char
        skip_str = cls.skip_char * skip
        for line in tqdm(data, mininterval=1):
            words = WordScheme.encode_str(line)
            nwords = len(words)
            if nwords > skip+1:
                words = [ f'{word}{cls.space_char}' for word in words ]
                for i in range(nwords-(skip+1)):
                    name = f'{words[i]}{skip_str}{words[i+skip+1]}'
                    if name not in sgram_freqs.keys():
                        sgram_freqs[name] = coll.Counter()
                    sgram_freqs[name].update([''.join(words[i+1:i+skip+1])])
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
            
            exname = name.replace(cls.space_char, f'{cls.space_char} ')
            exname = exname.replace(cls.skip_char, f'{cls.skip_char} ')
            words = exname.split()
            word_freqs = [0 if cls.skip_char in word else term_freqs[word[:-1]]
                            for word in words]
            ninstances = len(instances.keys())
            max_prob =  max([val/freq for val in instances.values()])
            sgrams_list.append(Type(name, freq=freq, idx=0,
                                    level=1, kids=word_freqs))
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
            Tuple[int,float]]], bpes:List['Type'], all_words:Set[str], max_instance_prob:float=1.0, 
            min_instances:int=0) -> List[Tuple['Type', Union[int,float], Tuple[int,float]]]:
        rev_idx = {t.name:t.idx for t in bpes}
        all_words.add(cls.skip_char)
        filtered = []
        # cnt = 2
        # na = 0
        # cons = 0

        # def _twrite(fr, tok, val, stats, status, not_word=None):
        #     fr.write('\t'.join(map(str,[status, tok.name, tok.freq, val, stats, not_word])))
        #     fr.write('\n')

        # print(f' Min Instances : {min_instances} , Max Instance Prob : {max_instance_prob}')
        # fr = open(f'.data.simp/checks/check.{cls.count}.txt', 'w')
        # cls.count += 1
        for trp in sgrams_list:
            tok, val, stats = trp
            ninstances, max_prob = stats
            if ninstances < min_instances or max_prob > max_instance_prob:
                # cons += 1
                # _twrite(fr, tok, val, stats, 'Failed')
                continue
            xname = tok.name.replace(cls.skip_char, f'{cls.skip_char} ')
            parts = xname.replace(cls.space_char, f'{cls.space_char} ').split()
            not_word = [ part not in all_words 
                                for part in parts]
            # na += 1
            if not any(not_word):
                # if cnt <10:
                #     print(parts)
                #     print(not_word)
                #     print()
                #     cnt+= 1
                # na -= 1
                kids = [bpes[rev_idx[x]] for x in parts]
                tok = Type(tok.name, tok.level, tok.idx, 
                                tok.freq, kids)
                filtered.append((tok, val, stats))
                # _twrite(fr, tok, val, stats, 'Passed', not_word)
            # else:
                # _twrite(fr, tok, val, stats, 'Failed', not_word)
        # print('Not Passed :', na)
        # print('Failed constraint : ', cons)
        # fr.close()
        return filtered

    @classmethod
    def get_sgrams_lists(cls, data:Iterator[str], sgrams:List[Tuple[int,int]]=None,
                        sorter_func:str='freq', min_freq:int=MWE_MIN_FREQ,
                        min_instances:int=15, max_instance_prob:float=0.1,
                        vocab_size:int=0, bpe_vocab:List['Type']=None):
        assert sgrams is not None
        assert sorter_func == 'freq' or sorter_func in PMIFuncs.sgram_variants
        assert vocab_size != 0 or bpe_vocab is not None

        print('Found term freqs and nlines')
        term_freqs, nlines = WordScheme.term_frequencies(data)
        all_words = set(term_freqs.keys())

        sgrams_list = {}

        if bpe_vocab is None:
            
            def init_vocab_factory(char_types):
                tvcb = CharScheme.get_init_vocab(char_types, line_count=nlines,
                                                coverage=DEF_CHAR_COVERAGE, 
                                                min_freq=1)
                vocab = Reseved.with_reserved_types()
                for tok in cls.TOKS:
                    name, idx = tok
                    vocab.append(Type(name, level=-1, idx=idx, freq=0))
                for tok in tvcb:
                    if tok.level < 0:
                        continue
                    vocab.append(Type(tok.name, level=tok.level, 
                                    idx=len(vocab), freq=tok.freq,
                                    kids=tok.kids))
                return vocab

            bpe_vocab = BPELearn.learn_subwords(term_freqs=term_freqs, 
                                            vocab_size=vocab_size,
                                            init_vocab_factory=init_vocab_factory,
                                            min_co_evidence=DEF_MIN_CO_EV)

        bpe_words = NgramScheme.get_bpe_words(bpe_vocab, all_words)
        print('Total Words : ', len(bpe_words))

        print('Making skipgrams')
        for sg in sgrams:
            print(f'> Preparing skipgrams {str(sg)}')
            sgram_freqs = cls.skipgram_frequencies(data, sg)
            sorted_sgrams = cls.sorted_sgrams(sgram_freqs, term_freqs,
                                            nlines, sorter_func, min_freq)
            del(sgram_freqs)
            hash = (sg[0]*cls.hash_prime) + sg[1]
            sgrams_list[hash] = cls.filtered_sgrams(sorted_sgrams, bpe_vocab,
                                                    bpe_words,
                                                    max_instance_prob, 
                                                    min_instances)
        return sgrams_list, bpe_vocab

    @classmethod
    def merge_lists(cls, base, lists, vocab_size, grams, toks_list):
        return NgramScheme.merge_lists(base, lists, vocab_size, grams, toks_list)

    @classmethod
    def learn(cls, data:Iterator[str], vocab_size:int=0, 
                sgrams:List[Tuple[int,int]]=None, max_sgrams:int=0, 
                skipgram_sorter:str='freq', toks_list:List[int]=[],
                min_freq:int=MWE_MIN_FREQ, min_instances:int=15, 
                max_instance_prob:float=0.1, **kwargs) -> List['Type']:
        assert sgrams is not None
        assert max_sgrams > 0 or len(toks_list) == len(sgrams)

        ## Currently support for n-skip-2 grams only. Need to discuss
        #  about this with AVT

        sgrams_lists, base = cls.get_sgrams_lists(data, sgrams, skipgram_sorter,
                                            min_freq, min_instances,
                                            max_instance_prob,
                                            vocab_size=vocab_size)

        if len(toks_list) == 0:
            toks_list = [max_sgrams // len(sgrams)] * len(sgrams)
        assert vocab_size > sum(toks_list)

        hashed_sgrams = list(map(lambda x: (x[0]*cls.hash_prime) + x[1], sgrams))
        return cls.merge_lists(base, sgrams_lists, vocab_size,
                                hashed_sgrams, toks_list)


class MWEScheme(SkipScheme):
    
    get_ngrams_lists = NgramScheme.get_ngrams_lists

    @classmethod
    def learn(cls, data:Iterator[str], vocab_size:int=0, 
            mwes:List[Union[int,Tuple[int,int]]]=None, 
            ngram_sorter:str='freq', skipgram_sorter:str='freq',
            toks_list:List[int]=[], max_mwes:int=0,
            min_freq:int=MWE_MIN_FREQ, min_instances:int=15,
            max_instance_prob:float=0.1, **kwargs) -> List['Type']:
        assert mwes is not None
        assert max_mwes > 0 or len(toks_list) == len(mwes)

        base = BPEScheme.learn(data, vocab_size)
        term_freqs, nlines = WordScheme.term_frequencies(data)
        mwes_list = {}

        ngrams_lists, _ = cls.get_ngrams_lists(data, 
                            [x for x in mwes if type(x)==int],
                            merge_func, min_freq, bpe_vocab=base)
        sgrams_lists, _ = cls.get_sgrams_lists(data, 
                            [x for x in mwes if type(x)!=int],
                            merge_func, min_freq, min_instances,
                            bpe_vocab=base)
        
        mwes_list.update(ngrams_lists)
        mwes_list.update(sgrams_lists)

        if len(toks_list) == 0:
            toks_list = [max_mwes // len(mwes)] * len(mwes)
        assert vocab_size > sum(toks_list)

        return cls.merge_lists(base, mwes_list, vocab_size,
                                mwes, toks_list)


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