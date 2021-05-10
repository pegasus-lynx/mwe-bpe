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
    parser.add_argument('-m', '--merge_mode', type=str, choices=['freq', 'replace', 'append'], 
                        default='freq', help='Mode for merging the vocabs.')
    parser.add_argument('-t', '--tokens_list', type=int, nargs='+', help="""Number of tokens to be 
                        appended by/replaced with tokens from corresponding to the vocab files""")
    # parser.add_argument('-f', '--min_freqs', type=int, default=0, help='Minimum frequncy upto which the n-grams are to be considered')
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
    assert len(args.tokens_list) == len(args.data_files)

# ----------------------------------------------------------------------------

def _load_vocab_files(vocab_files:List[Filepath]):
    vocabs = []
    for vocab_file in vocab_files:
        try:
            vcb = Vocabs(vocab_file)
        except Exception:
            vcb = Vocabs()
            vcb._read_in(vocab_file)
        vocabs.append(vcb)
    return vocabs

def merge_vocabs_by_freq(bpe_file:Filepath, vocab_files:List[Filepath], vocab_size:int=8000):
    bpe_vocab = Vocabs(bpe_file)
    vocabs = [bpe_vocab]
    vocabs.extend(_load_vocab_files(vocab_files))
    indexes = [0 for i in range(len(vocabs))]
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

def merge_vocabs_by_append(bpe_file:Filepath, vocab_files:List[Filepath], tokens_list:List[int]=[]):
    bpe_vocab = Vocabs(bpe_file)
    vocabs = _load_vocab_files(vocab_files)
    for ntokens, vocab in zip(tokens_list, vocabs):
        for ix, token in enumerate(vocab):
            if ix >= ntokens:
                break
            bpe_vocab.append(Type(token.name, level=1, idx=len(bpe_vocab), freq=token.freq, kids=token.kids))
    return bpe_vocab

def merge_vocabs_by_replace(bpe_file:Filepath, vocab_files:List[Filepath], tokens_list:List[int]=[]):
    merged_vocab = Vocabs()
    bpe_vocab = Vocabs(bpe_file)
    vocabs = _load_vocab_files(vocab_files)
    rtokens = sum(tokens_list)
    bpe_len = len(bpe_vocab) - rtokens
    assert bpe_len > 0
    vocabs.insert(0,bpe_vocab)
    tokens_list.insert(0,bpe_len)
    for ntokens, vocab in zip(tokens_list, vocabs):
        for ix, token in enumerate(vocab):
            if ix >= ntokens:
                break
            merged_vocab.append(Type(token.name, level=token.level, idx=len(merged_vocab), freq=token.freq, kids=token.kids))        
    return merged_vocab

def merge_vocabs(bpe_file:Filepath, vocab_files:List[Filepath], vocab_size:int=8000, mode:str='freq', tokens_list:List[int]=[]):
    # To Do : Implement use of parts and fparts
    vocab = None
    if mode == 'freq':
        vocab = merge_vocabs_by_freq(bpe_file, vocab_files, vocab_size=vocab_size)
    elif mode == 'append':
        vocab = merge_vocabs_by_append(bpe_file, vocab_files, tokens_list)
    elif mode == 'replace':
        vocab = merge_vocabs_by_replace(bpe_file, vocab_files, tokens_list)
    return vocab

def main():
    log('Starting Script : merge_vocabs')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    wdir = make_dir(args.work_dir)
    work_file = wdir/Path(f'merge.{args.vocab_size//1000}.{"_".join([str(x.with_suffix("")) for x in args.data_files])}.model')
    if args.save_file is not None:
        work_file = wdir / Path(args.save_file)

    log('> Merging Vocabs', 1)
    vcb = merge_vocabs(args.bpe_file, args.data_files, vocab_size=args.vocab_size, mode=args.merge_mode, tokens_list=args.tokens_list)
    vcb.save(work_file)
    log('Proces completed')
    save_meta(args, wdir, work_file)
    log('Writing Meta')

if __name__ == '__main__':
    main()