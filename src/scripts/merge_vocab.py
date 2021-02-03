import argparse
from pathlib import Path
from typing import Union, List, Dict, Set, Tuple

import numpy as np
from tqdm import tqdm
from lib.misc import Filepath, log, make_dir, FileWriter
from lib.vocabs import Vocabs
from lib.dataset import read_parallel, Dataset
from nlcodec import Type

#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.merge_vocabs', description='Merges vocab files in order of frequency')
    parser.add_argument('-w', '--work_dir', type=Path, help='Path of the working directory')
    parser.add_argument('-d', '--data_files', type=Path, nargs='+', help='Path of the corpus files')
    parser.add_argument('-b', '--bpe_file', type=Path, help='Path to the base bpe vocab file')
    parser.add_argument('-s', '--vocab_size', type=int, default=8000, help='Vocab size of the merged file')
    # parser.add_argument('-f', '--min_freqs', type=int, default=0, help='Minimum frequncy upto which the n-grams are to be considered')
    # parser.add_argument('-t', '--max_tokens', type=int, default=0, help='Maximum n-grams to be considered.')
    parser.add_argument('-x', '--save_file', type=str, help='Name of the file for the merged vocab.')
    return parser.parse_args()

def save_meta(args, work_dir, work_file):
    mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')
    mw.heading('merge_vocab')
    mw.section('Work File :', [work_file.name])
    mw.section('Arguments :', [f'Vocab Size : {args.vocab_size}'])
    lines = [f'Bpe Vocab File : {args.bpe_file}', 'Other Files :']
    lines.extend(args.data_files)
    mw.section('Vocab Files :', lines)
    mw.close(add_dashline=True)

def args_validation(args):
    assert args.work_dir is not None
    assert args.bpe_file.exists()
    for filepath in args.data_files:
        assert filepath.exists()

# ----------------------------------------------------------------------------

def merge_vocabs(bpe_file:Filepath, vocab_files:List[Filepath], vocab_size:int=8000, parts:List[int]=[], fparts:List[float]=[]):
    # To Do : Implement use of parts and fparts
    bpe_vocab = Vocabs(bpe_file)
    vocabs = [bpe_vocab]
    for vocab_file in vocab_files:
        try:
            vcb = Vocabs(vocab_file)
        except Exception:
            vcb = Vocabs()
            vcb._read_in(vocab_file)
        vocabs.append(vcb)
    indexes = [0 for i in range(len(vocab_files)+1)]
    merged_vcb = Vocabs()
    while len(merged_vcb) < vocab_size:
        freqs = [ vocab.table[ix].freq if ix < len(vocab) else 0 for vocab, ix in zip(vocabs, indexes) ]
        levels = [ vocab.table[ix].level if ix < len(vocab) else 1 for vocab, ix in zip(vocabs, indexes) ]
        min_level = min(levels)
        max_freq = max(freqs)
        if min_level < 1:
            for ix, level in enumerate(levels):
                if level == min_level:
                    token = vocabs[ix].table[indexes[ix]]
                    if token.name in merged_vcb.tokens:
                        indexes[ix] += 1
                        break
                    merged_vcb.append(Type(token.name, level=token.level, idx=len(merged_vcb), freq=token.freq, kids=None))
                    indexes[ix] += 1
                    break
            continue
        if max_freq == 0:
            break
        for ix, freq in enumerate(freqs):
            if max_freq == freq:
                token = vocabs[ix].table[indexes[ix]]
                kids = None
                if token.kids is not None:
                    kids = []
                    for kid in token.kids:
                        pos = merged_vcb.index(kid.name)
                        kids.append(merged_vcb.table[pos])
                else:
                    token_kids = None
                    try:
                        token_kids = vocabs[ix].kids_list[indexes[ix]]
                        kids = []
                    except Exception:
                        pass
                    if token_kids is not None:
                        for kid in token_kids:
                            pos = merged_vcb.index(bpe_vocab.table[kid].name)
                            if pos is not None:
                                kids.append(merged_vcb.table[pos])
                if token.name in merged_vcb.tokens:
                    indexes[ix] += 1
                    break
                merged_vcb.append(Type(token.name, level=1, idx=len(merged_vcb), freq=freq, kids=kids))
                indexes[ix] += 1
                break
    return merged_vcb

def main():
    log('Starting Script : merge_vocabs')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    wdir = make_dir(args.work_dir)
    work_file = wdir/Path(f'merge.{args.vocab_size//1000}.{"_".join([x.with_suffix("").name for x in args.data_files])}.model')
    if args.save_file is not None:
        work_file = wdir / Path(args.save_file)

    log('> Merging Vocabs', 1)
    vcb = merge_vocabs(args.bpe_file, args.data_files, vocab_size=args.vocab_size)
    vcb.save(work_file)
    log('Proces completed')
    save_meta(args, wdir, work_file)
    log('Writing Meta')

if __name__ == '__main__':
    main()