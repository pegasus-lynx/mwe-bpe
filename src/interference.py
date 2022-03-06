import argparse
import os
import json

from pathlib import Path 
import collections as coll
from typing import List, Dict, Tuple
from tqdm import tqdm

from nlcodec import Type
from nlcodec.codec import ExtMWEScheme as ems
from rtg.data.dataset import TSVData, SqliteFile

from lib.analysis import Scores, Analysis
from nlcodec.codec import load_scheme
from ruamel.yaml import YAML


def read_tok_file(filepath, vocab=None):
    lines = []
    with open(filepath) as fp:
        for line in fp:
            lines.append(vocab.encode(line) if vocab is not None else line)
    return lines


def read_out_file(filepath, vocab=None):
    lines = []
    tok_2_idx = dict()
    
    if vocab is not None:
        for tok in vocab.table:
            tok_2_idx[tok.name] = tok.idx

    with open(filepath) as fp:
        for line in fp:
            text, _ = line.split('\t')
            tokens = text.split()
            lines.append([tok_2_idx[tok] for tok in tokens] if vocab is not None else text)
    
    return lines


def get_ngram_tokens(vocab):
    tokens = []
    skip_char = ems.skip_char
    for tok in vocab.table:
        if tok.level < 3:
            continue
        if skip_char in tok.name:
            continue
        tokens.append(tok)
    return tokens


def get_missing_tokens(base_vocab, freq_vocab):
    freq_toks = set([t.name for t in freq_vocab.table])
    missing_tok = []
    for ix, tok in enumerate(base_vocab.table):
        if tok.name not in freq_toks:
            missing_tok.append(tok)

    return missing_tok


def _get_count(seq, toks):
    cnt = 0
    if type(toks) == int:
        for x in seq:
            if x == toks:
                cnt += 1
        return cnt
    if len(seq) < len(toks):
        return 0
    for i in range(len(seq)-len(toks)+1):
        match = True
        for p in range(len(toks)):
            if toks[p] != seq[i+p]:
                match=False
                break
        if match:
            cnt += 1
    return cnt


def _get_str_count(text, tok, add_space=True):
    if add_space:
        pattern = f'{ems.space_char}{tok}'
    else:
        pattern = tok
    return text.count(pattern)


def _get_tok_count(seq:List[str], tok:str):
    cnt = 0
    for i in range(len(seq)):
        if seq[i]==tok:
            if i==0 or seq[i-1].endswith(ems.space_char):
                cnt += 1
    return cnt


def _get_tok_kids_count(seq:List[str], tok_kids:List[str]):
    l = len(tok_kids)
    cnt = 0
    pattern = ' '.join(tok_kids)
    for i in range((len(seq)-l)+1):
        if ' '.join(seq[i:i+l]) == pattern:
            if i == 0 or seq[i-1].endswith(ems.space_char):
                cnt += 1
    return cnt


def _get_words_rep(tokens):
    words = dict()
    
    space_ends = [-1]
    for i in range(len(tokens)):
        if tokens[i].endswith(ems.space_char):
            space_ends.append(i)
    
    for i in range(len(space_ends)-1):
        st = space_ends[i]+1
        en = space_ends[i+1]+1
        word = ''.join(tokens[st:en])
        rep = ' '.join(tokens[st:en])
        
        if word not in words:
            words[word] = dict()
        if rep not in words[word]:
            words[word][rep] = 0

        words[word][rep] += 1 

    return words

def _get_ngram_reps(out_lines, tok:str):
    reps = dict()

    for out in out_lines:
        parts = out.split()
        text = ''.join(parts)
        if tok not in text:
            continue
        
        text_rev = []
        for i, x in enumerate(parts):
            for p in range(len(x)):
                text_rev.append(i)
        
        l = len(tok)
        for i in range(len(text)-len(tok)+1):
            if text[i:i+l] == tok:
                st = text_rev[i]
                en = text_rev[i+l-1]
                rep = ' '.join(parts[st:en+1])

                if rep not in reps:
                    reps[rep] = 0

                reps[rep] += 1
            
    return reps


def get_diff_subword_reps(out_lines, vocab, save_file=None):

    areps = dict()

    for out in out_lines:
        tokens = out.split()
        words = _get_words_rep(tokens)

        for wrd, reps in words.items():
            if wrd not in areps:
                areps[wrd] = dict()
            for rep in reps:
                if rep not in areps[wrd]:
                    areps[wrd][rep] = 0
                areps[wrd][rep] += reps[rep]

    if save_file:
        yaml = YAML()
        with open(save_file, 'w') as fw:
            yaml.dump(areps, fw)

    return areps


def get_diff_ngram_reps(out_lines, vocab, save_file):

    ngrams = get_ngram_tokens(vocab)

    areps = dict()

    for tok in ngrams:
        areps[tok.name] = _get_ngram_reps(out_lines, tok.name)

    if save_file:
        yaml = YAML()
        with open(save_file, 'w') as fw:
            yaml.dump(areps, fw)
    return areps

def interference_analysis(vocabs, out_lines, base_lines, out_dir):
    #get_diff_subword_reps(out_lines['base'], vocabs, out_dir / 'reps.base.subword.yml')
    get_diff_ngram_reps(out_lines['freq'], vocabs['freq'], out_dir / 'reps.freq.ngram.yml')

def parse_args():
    parser = argparse.ArgumentParser(prog="interference_eval", description="Used to get the f1 analysis per mwe token.")
    #parser.add_argument('-tb', '--tgt_base_vcb', type=Path)
    parser.add_argument('-tf', '--tgt_freq_vcb', type=Path)
    #parser.add_argument('-vt', '--valid_tgt', type=Path)
    parser.add_argument('-ff', '--freq_out_file', type=Path)
    parser.add_argument('-bf', '--base_out_file', type=Path)
    parser.add_argument('-o' , '--out_dir'  , type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    vocabs = dict()
    #vocabs['base'] = load_scheme(args.tgt_base_vcb)
    vocabs['freq'] = load_scheme(args.tgt_freq_vcb)
   
    out_lines = dict()
    out_lines['base'] = read_out_file(args.base_out_file)
    out_lines['freq'] = read_out_file(args.freq_out_file)

    #base_lines = read_tok_file(args.valid_tgt)
    
    interference_analysis(vocabs, out_lines, None, args.out_dir)


if __name__ == "__main__":
    main()
