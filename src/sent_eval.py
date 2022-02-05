import argparse
import os
import json

from pathlib import Path 
import collections as coll
from typing import List, Dict, Tuple

from tqdm import tqdm
from rtg.data.codec import SPField
import sentencepiece as spm
from lib.analysis import Scores, Analysis
from nlcodec.codec import load_scheme
from ruamel.yaml import YAML

space_char = '▁'

def read_file(filepath, vocab=None):
    lines = []
    with open(filepath) as fp:
        for line in fp:
            lines.append(vocab.encode_as_ids(line) if vocab is not None else line)
    return lines

def parse_args():
    parser = argparse.ArgumentParser(prog="token_eval", description="Used to get the performance per token.")
    parser.add_argument('-s', '--src_vcb', type=str)
    parser.add_argument('-t', '--tgt_vcb', type=str)
    parser.add_argument('-d', '--detok_files', nargs='+', type=str)
    parser.add_argument('-r', '--ref_file', type=Path)
    parser.add_argument('-vs', '--valid_src', type=Path)
    parser.add_argument('-vt', '--valid_tgt', type=Path)
    parser.add_argument('-e', '--eval_type', type=str, choices=['group_bleu', 'bleu_line_sort', 'both'], default='group_bleu')
    parser.add_argument('-o', '--out_dir', type=Path)
    return parser.parse_args()

def get_scores(ofiles, refs, lines, calculate_rev=True):    
    #refs = read_file(ref_file)
    
    frefs, rev_frefs = [], []
    for x in range(len(refs)):
        if x in lines:
            frefs.append(refs[x])
        else:
            rev_frefs.append(refs[x])
    
    #frefs = [refs[x] for x in lines]
    
    #if calculate_rev:
     #   rev_frefs = [refs[x] for x in range(len(refs)) if x not in lines]

    nlines = len(lines)
    scores = dict()
    rev_scores = dict()
    
    for key, outs in ofiles:
        #outs = read_file(ofile)
        
        #fouts = [outs[x] for x in lines]
        #
        #if calculate_rev:
        #    rev_fouts = [outs[x] for x in range(len(outs)) if x not in lines]

        fouts, rev_fouts = [], []

        for x in range(len(refs)):
            if x in lines:
                fouts.append(outs[x])
            else:
                rev_fouts.append(outs[x])

        scores[key] = dict()
        rev_scores[key] = dict()

        scores[key]['bleu'] = 0 if nlines==0 else str(Scores.corpus_bleu(fouts, frefs))
        # scores[key]['chrf'] = 0 if nlines==0 else str(Scores.corpus_chrf(fouts, frefs))
        
        if calculate_rev:
            rev_scores[key]['bleu'] = 0 if nlines==0 else str(Scores.corpus_bleu(rev_fouts, rev_frefs))
            # rev_scores[key]['chrf'] = 0 if nlines==0 else str(Scores.corpus_chrf(rev_fouts, rev_frefs))

    return scores, rev_scores


def one_side_eval(test_lines, vocab, out_files, ref_file, save_file=None):
    tokenwise_lines = Analysis.get_tokenwise_lines(test_lines)

    stats = dict()

    refs = read_file(ref_file)
    ofiles = []
    for key, out_file in out_files:
        ofiles.append((key, read_file(out_file)))

    for ix in tqdm(range(len(vocab)), mininterval=1):
        n = vocab.id_to_piece(ix)
        cnt = n.count(space_char)
        if cnt<2:
            continue

        stats[n] = dict()
        stats[n]['idx'] = ix
        
        lines = [] if ix not in tokenwise_lines.keys() else tokenwise_lines[ix]

        stats[n]['freq'] = len(lines)
        stats[n]['lines'] = lines

        scores, rev_scores = get_scores(ofiles, refs, lines)
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

    refs = read_file(ref_file)
    ofiles = []
    for key, out_file in out_files:
        ofiles.append((key, read_file(out_file)))

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
        scores, rev_scores = get_scores(ofiles, refs, lines)
        ostats[f]['scores'] = scores
        ostats[f]['rev_scores'] = rev_scores

    if save_file is not None:
        yaml = YAML()
        with open(save_file, 'w') as fw:
            yaml.dump(ostats, fw)

    return ostats

def bleu_based_line_sort(out_files, ref_file, valid_src_lines, valid_tgt_lines, src_vocab, tgt_vocab, save_file=None):


    refs = read_file(ref_file)
    nlines = len(refs)

    ofiles = dict()
    for key, out_file in out_files:
        ofiles[key] = read_file(out_file)

    scores_list = []
    for ix in range(nlines):
        scores, _ = get_scores(out_files, ref_file, [ix], False)
        bleu_diff = scores['skip100']['bleu'].score - scores['base']['bleu'].score
        
        scores_list.append((ix, bleu_diff, scores))

    scores_list.sort(key=lambda x:x[1], reverse=True)

    if save_file is not None:
        with open(save_file, 'w') as fw:
            for ix, bleu_diff, scores in scores_list:
                fw.write(f'{ix}  |  Diff : {bleu_diff}    |  Skip : {str(scores["skip100"]["bleu"].score)}  |  Base : {str(scores["base"]["bleu"].score)}\n')
                src_line = src_vocab.decode(valid_src_lines[ix])
                fw.write(f'SRC   |  {src_line}\n')
                fw.write(f'REF   |  {refs[ix]}')
                fw.write(f'BASE  |  {ofiles["base"][ix]}')
                fw.write(f'SKIP  |  {ofiles["skip100"][ix]}\n\n')

    return scores_list


def main():
    args = parse_args()

    src_vocab = spm.SentencePieceProcessor(model_file=args.src_vcb)
    tgt_vocab = spm.SentencePieceProcessor(model_file=args.tgt_vcb)
    valid_src_lines = read_file(args.valid_src, src_vocab)
    valid_tgt_lines = read_file(args.valid_tgt, tgt_vocab)
    
    assert len(args.detok_files) % 2 == 0
    out_files = []

    for i in range(len(args.detok_files)//2):
        key = args.detok_files[(2*i)]
        fpath = Path(args.detok_files[(2*i)+1])
        out_files.append((key, fpath))
    
    if args.eval_type == 'group_bleu' or args.eval_type == 'both':
        src_stats = one_side_eval(valid_src_lines, src_vocab, out_files, args.ref_file, args.out_dir / Path('stats.src.yml'))
        tgt_stats = one_side_eval(valid_tgt_lines, tgt_vocab, out_files, args.ref_file, args.out_dir / Path('stats.tgt.yml'))
        aggregate_stats(src_stats, out_files, args.ref_file, args.out_dir / Path('stats.src.aggregated.yml'))
        aggregate_stats(tgt_stats, out_files, args.ref_file, args.out_dir / Path('stats.tgt.aggregated.yml'))
    
    #if args.eval_type == 'bleu_line_sort' or args.eval_type == 'both':
    #    bleu_based_line_sort(out_files, args.ref_file, valid_src_lines, valid_tgt_lines, src_vocab, tgt_vocab, args.out_dir / Path('stats.sort.txt'))

if __name__ == "__main__":
    main()
