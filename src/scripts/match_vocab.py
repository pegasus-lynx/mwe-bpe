import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

from lib.misc import FileWriter, log, make_dir
from lib.vocabs import Vocabs
from nlcodec import Reseved, Type

#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.match_vocab', description='Matches bpe vocab tokens with word vocab')
    parser.add_argument('-b', '--bpe_vocab', type=Path, help='Bpe Vocab file from where to find the full tokens')
    parser.add_argument('-v', '--word_vocab', type=Path, help='Word vocab file to match the full tokens')
    parser.add_argument('-w', '--work_dir', type=Path, help='Working Experiment Directory')
    parser.add_argument('-x', '--save_file', type=str)
    return parser.parse_args()

def save_meta(args, work_dir, work_file):
    mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')
    mw.heading('match_vocab')
    mw.section('Work File :', [work_file.name])
    mw.section('Files :', [
        f'BPE Vocab File  : {args.bpe_vocab}',
        f'Word Vocab File : {args.word_vocab}'
    ])
    mw.close(add_dashline=True)

def args_validation(args):
    assert args.word_vocab.exists()
    assert args.bpe_vocab.exists()

# ----------------------------------------------------------------------------

def match_vocab(bpe_vocab:Path, word_vocab:Path, work_file:Path, write=True):
    bpe_vcb = Vocabs(bpe_vocab)
    word_vcb = Vocabs(word_vocab)
    match_vcb = Vocabs()

    for token in [x[0] for x in Reseved.ALL]:
        if token in word_vcb.tokens:
            word_vcb.tokens.remove(token)
    for x in bpe_vcb:
        # if x.name in word_vcb.tokens:
        #     match_vcb.append(x)
        if x.name.endswith(Reseved.SPACE_TOK[0]) and x.name[:-1] in word_vcb.tokens:
            match_vcb.append(x)
    if write:
        match_vcb._write_out(work_file)
    return match_vcb

def main():
    log('Starting script : match_vocab')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    wdir = make_dir(args.work_dir)

    work_file = wdir / Path(f'match.{args.bpe_vocab.with_suffix(".word.model").name}')
    if args.save_file is not None:
        work_file = wdir / Path(args.save_file)

    log('Matching bpe with word vocab', 1)
    match_vocab(args.bpe_vocab, args.word_vocab, work_file)
    log('Process completed')
    
    save_meta(args, wdir, work_file)
    log('Writing meta')

if __name__ == "__main__":
    main()
