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

def read_file(filepath, vocab=None):
    lines = []
    with open(filepath) as fp:
        for line in fp:
            lines.append(vocab.encode(line) if vocab is not None else line)
    return lines

def read_out_tsv_file(filepath, vocab=None):
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

def parse_args():
    parser = argparse.ArgumentParser(prog="freq_eval", description="Used to get the performance per token.")
    parser.add_argument('-sb', '--src_base_vcb', type=Path)
    parser.add_argument('-tb', '--tgt_base_vcb', type=Path)
    #parser.add_argument('-df', '--detok_files', nargs='+', type=str)
    parser.add_argument('-sf', '--src_freq_vcb', type=Path)
    parser.add_argument('-of', '--detok_files', nargs='+', type=str)
    parser.add_argument('-tf', '--tgt_freq_vcb', type=Path)
    parser.add_argument('-r', '--ref_file', type=Path)
    parser.add_argument('-vs', '--valid_src', type=Path)
    parser.add_argument('-vt', '--valid_tgt', type=Path)
    #parser.add_argument('-e', '--eval_type', type=str, choices=['group_bleu', 'bleu_line_sort', 'both', 'coverage'], default='group_bleu')
    parser.add_argument('-o', '--out_dir', type=Path)
    parser.add_argument('-suite', '--suite_name', type=str, default="")
    return parser.parse_args()

def get_coverage(src_lines, tgt_lines, out_tsv_file, src_vocab, tgt_vocab, out_dir, suite_name="test"):
    out_tgt_dist = None
    out_tokens = 10000000
    if out_tsv_file is not None:
        outs = read_out_tsv_file(out_tsv_file, tgt_vocab)
        out_tgt_dist, out_tokens = Analysis.get_token_freqs(outs)

    nlines = len(src_lines)
    src_dist, src_tokens = Analysis.get_token_freqs(src_lines)
    inp_tgt_dist, inp_tokens = Analysis.get_token_freqs(tgt_lines)
    print(src_tokens, inp_tokens)

    with open( out_dir/ f"cov.src.{suite_name}.txt", 'w') as fw:
        for idx in range(len(src_vocab)):
            if idx not in src_dist.keys():
                cov = 0
                freq = 0
            else:
                cov = src_dist[idx][0] / nlines
                freq = src_dist[idx][1]
            tok = src_vocab.table[idx]
            fw.write("\t".join([tok.name, str(cov), str(freq), str(freq / src_tokens)]))
            fw.write("\n")

    with open(out_dir / f"cov.tgt.{suite_name}.txt", 'w') as fw:
        for idx in range(len(tgt_vocab)):
            tok = tgt_vocab.table[idx]
            acov = 0
            afreq = 0
            if out_tgt_dist is not None:
                if idx in out_tgt_dist.keys():
                    acov = out_tgt_dist[tok.idx][0] / nlines
                    afreq = out_tgt_dist[tok.idx][1]

            ecov = 0
            efreq = 0
            if idx in inp_tgt_dist.keys():
                ecov = inp_tgt_dist[tok.idx][0] / nlines
                efreq = inp_tgt_dist[tok.idx][1]
            
            fw.write("\t".join([tok.name, str(ecov), str(efreq), str(efreq/inp_tokens), str(acov), str(afreq), str(afreq/out_tokens)]))
            fw.write("\n")


def get_scores(out_files, ref_file, lines, calculate_rev=True, bleu_str=True):    
    refs = read_file(ref_file)
    frefs = [refs[x] for x in lines]
    
    if calculate_rev:
        rev_frefs = [refs[x] for x in range(len(refs)) if x not in lines]

    nlines = len(lines)
    scores = dict()
    rev_scores = dict()
    
    for key, ofile in out_files:
        outs = read_file(ofile)
        fouts = [outs[x] for x in lines]
        
        if calculate_rev:
            rev_fouts = [outs[x] for x in range(len(outs)) if x not in lines]

        scores[key] = dict()
        rev_scores[key] = dict()

        bleu = 0 if nlines==0 else Scores.corpus_bleu(fouts, frefs)
        sbleu =  str(bleu) if bleu_str else bleu
        scores[key]['bleu'] = 0 if nlines==0 else sbleu
        scores[key]['chrf'] = 0 if nlines==0 else str(Scores.corpus_chrf(fouts, frefs))
        
        if calculate_rev:
            rev_scores[key]['bleu'] = 0 if nlines==0 else str(Scores.corpus_bleu(rev_fouts, rev_frefs))
            rev_scores[key]['chrf'] = 0 if nlines==0 else str(Scores.corpus_chrf(rev_fouts, rev_frefs))

    return scores, rev_scores


def one_side_eval(test_lines, vocab, out_files, ref_file, save_file=None):
    tokenwise_lines = Analysis.get_tokenwise_lines(test_lines)

    stats = dict()

    for token in vocab.table:
        if token.level < 3:
            continue
        n = token.name
        stats[n] = dict()
        stats[n]['idx'] = token.idx
        
        lines = [] if token.idx not in tokenwise_lines.keys() else tokenwise_lines[token.idx]

        stats[n]['freq'] = len(lines)
        stats[n]['lines'] = lines

        scores, rev_scores = get_scores(out_files, ref_file, lines)
        stats[n]['scores'] = scores
        stats[n]['rev_scores'] = rev_scores


    if save_file is not None:
        yaml = YAML()
        with open(save_file, 'w') as fw:
            yaml.dump(stats, fw)

    return stats


def aggregate_stats(stats, out_files, ref_file, save_file=None):
    
    freqs = [0,1,2,3,4,5,10]
    appears_more_than = { freq:[] for freq in freqs}

    for n in stats.keys():
        fq = stats[n]['freq']
        for freq in freqs:
            if fq > freq:
                appears_more_than[freq].append(n)
    
    ostats = dict()
    for f in freqs:
        ostats[f] = dict()
        tokens = appears_more_than[f]
        ostats[f]['ntokens'] = len(tokens)

        if len(tokens) == 0:
            ostats[f]['nlines'] = 0
            ostats[f]['scores'] = 0
            continue

        lines = set()
        for n in tokens:
            lines.update(stats[n]['lines'])

        lines = list(lines)
        ostats[f]['nlines'] = len(lines)
        scores, rev_scores = get_scores(out_files, ref_file, lines)
        ostats[f]['scores'] = scores
        ostats[f]['rev_scores'] = rev_scores

    if save_file is not None:
        yaml = YAML()
        with open(save_file, 'w') as fw:
            yaml.dump(ostats, fw)

    return ostats


def get_mwe_splits(vocab):
    tokens = set()

    space_tok = ems.space_char
    skip_tok = ems.skip_char
    
    for tok in vocab.table:
        if tok.level > 3:
            parts = tok.name.replace(space_tok, f'{space_tok} ').replace(skip_tok, f'{skip_tok} ').split()
            for part in parts:
                tokens.add(part)

    vcb_tokens = list()
    for tok in vocab.table:
        if tok.name in tokens:
            vcb_tokens.append(tok)

    return vcb_tokens

def get_missing_tokens(base_vocab, freq_vocab):
    tokens = []
    freq_tokens = set()
    for tok in freq_vocab.table:
        freq_tokens.add(tok.name)
    for tok in base_vocab.table:
        if tok.name not in freq_tokens:
            tokens.append(tok)
    return tokens

def _set_token_stats(idx, flines, detok_files, ref_file):
    sc, _ = get_scores(detok_files, ref_file, flines, False)
    return dict({'idx':idx, 'sfreq':len(flines), 'scores':sc})

def get_score_missing(lines, base_vocab, freq_vocab, detok_files, ref_file, save_file):
    missing_tokens = get_missing_tokens(base_vocab, freq_vocab)
    print("Missing Tokens : ", len(missing_tokens))
    stats = dict()

    flines = Analysis.get_strlines_with_tokens(lines, [ tok.name for tok in missing_tokens], True)
    stats['all'] = _set_token_stats('-', flines, detok_files, ref_file)
    
    for token in tqdm(missing_tokens):
        n = token.name
        flines = Analysis.get_strlines_with_token(lines, n, True)
        stats[n] = _set_token_stats(token.idx, flines, detok_files, ref_file)

    if save_file is not None:
        yaml = YAML()
        with open(save_file, 'w') as fw:
            yaml.dump(stats, fw)

    return stats

def get_score_mwe(lines, freq_vocab, detok_files, ref_file, save_file):
    tlines = [freq_vocab.encode(l) for l in lines]
    slines = [l.replace(' ', ems.space_char) for l in lines ]

    stats = dict()
    split_stats = get_score_mwe_splits(slines, freq_vocab, detok_files, ref_file)
    mwe_stats = get_score_mwe_toks(tlines, freq_vocab, detok_files, ref_file)
    stats.update(mwe_stats)
    stats.update(split_stats)

    if save_file is not None:
        yaml = YAML()
        with open(save_file, 'w') as fw:
            yaml.dump(stats, fw)

    return stats

def get_score_mwe_toks(lines, freq_vocab, detok_files, ref_file, save_path=None):
    mwe_tokens = [ tok.idx for tok in freq_vocab.table if tok.level >= 3]
    flines = Analysis.get_lines_with_tokens(lines, mwe_tokens, True)
    stats = dict()
    stats['all-mwe'] = _set_token_stats('-', flines, detok_files, ref_file)

    for x in tqdm(mwe_tokens):
        tok = freq_vocab.table[x]
        flines = Analysis.get_lines_with_token(lines, tok.idx, True)
        stats[tok.name] = _set_token_stats(tok.idx, flines, detok_files, ref_file)

    if save_path:
        pass

    return stats

def get_score_mwe_splits(lines, freq_vocab, detok_files, ref_file, save_path=None):
    mwe_splits = get_mwe_splits(freq_vocab)

    stats = dict()

    for tok in tqdm(mwe_splits):
        n = tok.name
        flines = Analysis.get_strlines_with_token(lines, n, True)
        stats[n] = _set_token_stats(tok.idx, flines, detok_files, ref_file)

    if save_path:
        pass

    return stats

def main():
    args = parse_args()

    vocabs = dict()
    vocabs['src'] = dict()
    vocabs['tgt'] = dict()
    vocabs['src']['base'] = load_scheme(args.src_base_vcb)
    vocabs['tgt']['base'] = load_scheme(args.tgt_base_vcb)
    vocabs['src']['freq'] = load_scheme(args.src_freq_vcb)
    vocabs['tgt']['freq'] = load_scheme(args.tgt_freq_vcb)

    file_lines = dict()
    file_lines['src'] = read_file(args.valid_src)
    file_lines['tgt'] = read_file(args.valid_tgt)
    
    assert len(args.detok_files) % 2 == 0
    detok_files = []

    for i in range(len(args.detok_files)//2):
        key = args.detok_files[(2*i)]
        detok_fpath = Path(args.detok_files[(2*i)+1])
        # out_fpath = Path(args.detok_files[(3*i)+2])
        detok_files.append((key, detok_fpath))
   
    # detok_files = [(key, val[1]) for key,val in out_files]

    for key in ['src', 'tgt']:
        # BLEU Scores for sentences with missing tokens
        slines = [l.replace(' ', ems.space_char) for l in file_lines[key] ]
        
        fpath = args.out_dir / Path(f'{key}.missing.yml')
        get_score_missing(slines, vocabs[key]['base'], vocabs[key]['freq'], detok_files, args.ref_file, fpath)

        # BLEU Scores for mwe tokens and token splits
        fpath = args.out_dir / Path(f'{key}.mwe.split.yml')
        get_score_mwe(file_lines[key], vocabs[key]['freq'], detok_files, args.ref_file, fpath)


if __name__ == "__main__":
    main()
