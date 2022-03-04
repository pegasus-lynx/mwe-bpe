from sacrebleu import corpus_chrf, corpus_bleu
from typing import List, Union


class Scores():
    
    @classmethod
    def corpus_bleu(cls, hyps:List[str], refs:List[str], lowercase=True):
        return corpus_bleu(hyps, [refs], lowercase=lowercase)

    @classmethod
    def line_bleu(cls):
        pass

    @classmethod
    def corpus_chrf(cls, hyps:List[str], refs:List[str], beta=2):
        return corpus_chrf(hyps, [refs], beta=beta)

    @classmethod
    def token_specific_corpus_blue(cls, hyps:List[str], refs:List[str], token:str):
        assert token is not None, "token should not be None. For general cases use corpus_bleu()"        

    @classmethod
    def score():
        pass


class Analysis():

    @classmethod
    def get_strlines_with_token(cls, lines:List[str], token:str, get_indexes:bool=True):
        present_in = []
        for ix, line in enumerate(lines):
            if token in line:
                present_in.append(ix)
        if get_indexes:
            return present_in
        return [lines[x] for x in present_in]

    @classmethod
    def get_strlines_with_tokens(cls, lines:List[str], tokens:List[str], get_indexes:bool=True):
        present_in = []
        for ix, line in enumerate(lines):
            for token in tokens:
                if token in line:
                    present_in.append(ix)
                    break
        if get_indexes:
            return present_in
        return [lines[x] for x in present_in]

    @classmethod
    def get_lines_with_token(cls, lines:List[List[Union[str,int]]], token:Union[str,int], get_indexes:bool=False):
        assert lines is not None and len(lines) != 0
        assert token is not None
        #assert type(token) == tpe(lines[0][0])

        present_in = []
        for ix, line in enumerate(lines):
            if token in line:
                present_in.append(ix)

        if get_indexes:
            return present_in

        return [ lines[x] for x in present_in ]


    @classmethod
    def get_lines_with_tokens(cls, lines:List[List[Union[str,int]]], tokens:List[Union[str,int]], get_indexes:bool=False):
        present_in = []
        for ix, line in enumerate(lines):
            for tok in line:
                if tok in tokens:
                    present_in.append(ix)
                    break
        if get_indexes:
            return present_in

        return [lines[x] for x in present_in]


    @classmethod
    def get_tokenwise_lines(cls, lines:List[List[Union[str,int]]]):
        assert lines is not None

        tokenwise_lines = dict()
        for ix, line in enumerate(lines):
            for token in line:
                if token not in tokenwise_lines.keys():
                    tokenwise_lines[token] = []
                tokenwise_lines[token].append(ix)

        return tokenwise_lines


    @classmethod
    def get_token_freqs(cls, lines:List[List[Union[str,int]]]):
        assert lines is not None
        token_freqs = dict()
        ntokens = 0
        for ix,line in enumerate(lines):
            uniq_toks = set()
            ntokens += len(line)
            for token in line:
                if token not in token_freqs.keys():
                    token_freqs[token] = [0,0]
                token_freqs[token][1] += 1
                uniq_toks.add(token)
            for tok in uniq_toks:
                token_freqs[token][0] += 1

        return token_freqs, ntokens


    @classmethod
    def get_all_tokens(cls, lines:List[List[Union[str,int]]]):

        tokens = set()

        for line in lines:
            for token in line:
                tokens.add(token)

        return tokens
