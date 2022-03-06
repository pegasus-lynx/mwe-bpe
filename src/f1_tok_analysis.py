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


def convert_af_pf(af,pf):
    tp = min(af,pf)
    fn, fp = 0, 0
    
    af -= tp
    pf -= tp

    if af != 0:
        fn = af
    elif pf != 0:
        fp = pf
    else:
        pass

    return tp, fp, fn


def get_tok_tfpn(out_lines, base_lines, toks, vocab):
    
    stats = dict()
    for tok in toks:
        stats[tok.name] = dict.fromkeys(['tp', 'fp', 'fn', 'tpk', 'fpk', 'fnk'], 0)

    for base, out in zip(base_lines, out_lines):
        base_toks = vocab.encode_str(base)
        pred_toks = out.split()

        for tok in toks:
            af = _get_tok_count(base_toks, tok.name)
            pf = _get_tok_count(pred_toks, tok.name)
            
            afk = _get_tok_kids_count(base_toks, [t.name for t in tok.kids])
            pfk = _get_tok_kids_count(pred_toks, [t.name for t in tok.kids])

            tp, fp, fn = convert_af_pf(af,pf)
            tpk, fpk, fnk = convert_af_pf(afk, pfk)

            stats[tok.name]['tp'] += tp
            stats[tok.name]['fp'] += fp
            stats[tok.name]['fn'] += fn
    
            stats[tok.name]['tpk'] += tpk
            stats[tok.name]['fpk'] += fpk
            stats[tok.name]['fnk'] += fnk
            

    return stats 


def get_tfpn(out_lines, base_lines, toks, vocab=None, full_word_check=True):

    #   TP : Where of_the_ should appear ( from base_lines ) and where it appears ( from out_lines ) 
    #   FP : When of_the_ is not present actually, but in the predicted place it is present 
    #   TN : When of_the_ is not present and should not be present as well
    #   FN : When of_the_ is present actually but was not there in the outcome.
    
    stats = dict()

    for tok in toks:
        stats[tok.name] = dict.fromkeys(['tp', 'fp', 'fn'], 0)

    for base, out in zip(base_lines, out_lines) :

        base_str = base.replace(' ', f'{ems.space_char}')

        if False:
            pred_str = vocab.decode_str(out.split()).replace(' ', f'{ems.space_char} ')
        else:
            pred_str = out.replace(' ', f'{ems.space_char}')

        for tok in toks:
            
            af = _get_str_count(base_str,tok.name, full_word_check)
            pf = _get_str_count(pred_str,tok.name, full_word_check)

            tp, fp, fn = convert_af_pf(af,pf)
            stats[tok.name]['tp'] += tp
            stats[tok.name]['fp'] += fp
            stats[tok.name]['fn'] += fn

    return stats

def _f1_analysis(vocabs, out_lines, base_lines, toks, out_dir, suite, full_word_check=True):
    stats = dict()
    stats['base'] = get_tfpn(out_lines['base'], base_lines, toks, full_word_check)
    stats['freq'] = get_tok_tfpn(out_lines['freq'], base_lines, toks, vocabs['freq'])
    
    astats = merge_stats(stats, vocabs, toks)

    if out_dir:
        pth = out_dir / f'f1-analysis-tok-{suite}.yml'
        # Look at this again
        yaml = YAML()
        with open(pth, 'w') as fw:
            yaml.dump(astats, fw)

    return stats

def f1_analysis(vocabs, out_lines, base_lines, out_dir):
    ngram_toks =  get_ngram_tokens(vocabs['freq'])
    missing_toks = get_missing_tokens(vocabs['base'], vocabs['freq'])

    _f1_analysis(vocabs, out_lines, base_lines, ngram_toks, out_dir, 'ngrams')
    #_f1_analysis(vocabs, out_lines, base_lines, missing_toks, out_dir, 'missing', False)


def _get_prec_rec_f1(tfpn):
    res = dict()

    try:
        prec = tfpn['tp'] / (tfpn['tp'] + tfpn['fp'])
    except ZeroDivisionError:
        prec = -1

    try:
        rec  = tfpn['tp'] / (tfpn['tp'] + tfpn['fn'])
    except ZeroDivisionError:
        rec = -1

    res['prec'] = prec
    res['rec'] = rec
    try:
        res['f1'] = ( 2*prec*rec ) / (prec + rec)
    except ZeroDivisionError:
        #cnt += 1
        res['f1'] = -1

    return res


def split_stats(mixed_stats):
    stats = dict.fromkeys(['tp', 'fp', 'fn'], 0)
    kids_stats = dict.fromkeys(['tp', 'fp', 'fn'], 0)

    stats['tp'] = mixed_stats['tp']
    stats['fp'] = mixed_stats['fp']
    stats['fn'] = mixed_stats['fn']
    kids_stats['tp'] = mixed_stats['tpk']
    kids_stats['fp'] = mixed_stats['fpk']
    kids_stats['fn'] = mixed_stats['fnk']

    return stats, kids_stats


def merge_stats(stats, vocabs, ngrams):
    astats = dict()
    for tok in ngrams:
        name = tok.name
        bstats = stats['base'][tok.name]
        bstats.update(_get_prec_rec_f1(bstats))
        
        fstats = stats['freq'][tok.name]
        fstats, fstats_k = split_stats(fstats)
        fstats.update(_get_prec_rec_f1(fstats))
        fstats_k.update(_get_prec_rec_f1(fstats_k))

        astats[name] = dict()
        astats[name]['freq'] = fstats
        astats[name]['freq-kids'] = fstats_k
        astats[name]['base'] = bstats

    return astats


def parse_args():
    parser = argparse.ArgumentParser(prog="f1_eval", description="Used to get the f1 analysis per mwe token.")
    parser.add_argument('-tb', '--tgt_base_vcb', type=Path)
    parser.add_argument('-tf', '--tgt_freq_vcb', type=Path)
    parser.add_argument('-vt', '--valid_tgt', type=Path)
    parser.add_argument('-ff', '--freq_out_file', type=Path)
    parser.add_argument('-bf', '--base_out_file', type=Path)
    parser.add_argument('-o' , '--out_dir'  , type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    vocabs = dict()
    vocabs['base'] = load_scheme(args.tgt_base_vcb)
    vocabs['freq'] = load_scheme(args.tgt_freq_vcb)
   
    out_lines = dict()
    out_lines['base'] = read_out_file(args.base_out_file)
    out_lines['freq'] = read_out_file(args.freq_out_file)
    print(out_lines['freq'][0])


    base_lines = read_tok_file(args.valid_tgt)
    
    f1_analysis(vocabs, out_lines, base_lines, args.out_dir)


if __name__ == "__main__":
    main()
