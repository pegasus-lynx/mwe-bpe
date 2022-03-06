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

def get_ngram_kids_base(ngrams, freq_vocabs, base_vocabs):
    pass

pr = 50000

def _hash(seq):
    if type(seq) == int:
        return seq
    val = 0
    for x in seq:
        val = val*pr + x
    return val


def _unhash(val):
    if val < pr:
        return val
    seq = []
    while val > 0:
        seq.append(val % pr)
        val = val // pr
    seq.reverse()
    return seq
    

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

def get_tfpn(out_lines, base_lines, tokens, vocab=None, check_kids=False):

    # The things that we need to calculate are : Precision and Recall 
    # For calculating these we require : TP / TN / FP / FN
    # Defining these terms ( with of_the_ as example ) :
    #   TP : Where of_the_ should appear ( from base_lines ) and where it appears ( from out_lines ) 
    #   However, there can be a difference in the word sequence , there can be more or less number of tokens before that. 
    #   In that case what to do . And suppose the tokens appears 2 or more times then how should we handle that ?? 
    #   The main question is how should we consider if that is the right position for the token or not ?? ]
    #
    #   FP : When of_the_ is not present actually, but in the predicted place it is present 
    #
    #   TN : When of_the_ is not present and should not be present as well
    #
    #   FN : When of_the_ is present actually but was not there in the outcome.
    
    stats = dict()

    vals = []
    for token in tokens:
        val = _hash(token)
        vals.append(val)
        stats[token] = dict.fromkeys(['tp', 'fp', 'fn'], 0)

    for base, out in zip(base_lines, out_lines) :

        act_set = set(base)
        pred_set = set(out)
        act_count = coll.Counter(base)
        pred_count = coll.Counter(out)

        for tok, val in zip(tokens, vals):
            af = _get_count(base, tok)
            pf = _get_count(out, tok)

            if check_kids:
                kids = vocab.table[val].kids
                cnt = _get_count(out, kids)
                pf += cnt

            tp = min(af,pf)
            fn, fp = 0, 0

            stats[val]['tp'] += tp
            
            af -= tp
            pf -= tp
            
            if af != 0:
                fn = af
            elif pf != 0:
                fp = pf
            else:
                pass
        
            stats[val]['fp'] += fp
            stats[val]['fn'] += fn

    return stats


def f1_analysis(vocabs, out_lines, base_lines, out_dir):
    ngram_toks =  get_ngram_tokens(vocabs['tgt']['freq'])
    ngram_toks_kids = get_ngram_kids_base(ngram_toks, vocabs['tgt']['freq'], vocabs['tgt']['base'])
    stats = dict()
    stats['base'] = get_tfpn(out_lines['base'], base_lines, ngram_toks_kids)
    stats['freq'] = get_tfpn(out_lines['freq'], base_lines, ngram_toks, vocabs['tgt']['freq'], True)
    
    astats = merge_stats(stats, vocabs, ngram_toks)

    if out_dir:
        pth = out_dir / 'f1-analysis.yml'
        # Look at this again
        yaml = YAML()
        with open(pth, 'w') as fw:
            yaml.dump(astats, fw)

    return stats


def _get_prec_rec_f1(tfpn):
    res = dict()

    prec = tfpn['tp'] / (tfpn['tp'] + tfpn['fp'])
    rec  = tfpn['tp'] / (tfpn['tp'] + tfpn['fn'])

    res['prec'] = prec
    res['rec'] = rec
    res['f1'] = ( 2*prec*rec ) / (prec + rec)

    return res


def merge_stats(stats, vocabs, ngrams):
    astats = dict()
    for ng in ngrams:
        tok = vocabs['tgt']['freq'].table[tok]
        name = tok.name

        val = _hash(tok.kids)
        
        bstats = stats['base'][val]
        bstats.update(_get_prec_rec_f1(bstats))
        fstats = stats['freq'][tok] 
        fstats.update(_get_prec_rec_f1(fstats))

        astats[name] = dict()
        astats[name]['freq'] = fstats
        astats[name]['base'] = bstats

    return astats

def parse_args():
    parser = argparse.ArgumentParser(prog="f1_eval", description="Used to get the f1 analysis per mwe token.")
    parser.add_argument('-sb', '--src_base_vcb', type=Path)
    parser.add_argument('-tb', '--tgt_base_vcb', type=Path)
    parser.add_argument('-sf', '--src_freq_vcb', type=Path)
    parser.add_argument('-tf', '--tgt_freq_vcb', type=Path)
    parser.add_argument('-vt', '--valid_tgt', type=Path)
    parser.add_argument('-ff', '--freq_out_file', type=Path)
    parser.add_argument('-bf', '--base_out_file', type=Path)
    parser.add_argument('-o' , '--out_dir'  , type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    vocabs = dict()
    vocabs['src'] = dict()
    vocabs['tgt'] = dict()
    vocabs['src']['base'] = load_scheme(args.src_base_vcb)
    vocabs['tgt']['base'] = load_scheme(args.tgt_base_vcb)
    vocabs['src']['freq'] = load_scheme(args.src_freq_vcb)
    vocabs['tgt']['freq'] = load_scheme(args.tgt_freq_vcb)
   
    out_lines = dict()
    out_lines['base'] = read_out_file(args.base_out_file)
    out_lines['freq'] = read_out_file(args.freq_out_file)

    base_lines = read_tok_file(args.valid_tgt)
    
    f1_analysis(vocabs, out_lines, base_lines, args.out_dir)


if __name__ == "__main__":
    main()
