# First we will prepare the data with the base  [ Done ]
# Then, we will take all the skipgram tokens to be added and then make the encoded text. [ Done ]
# Now after we have the encoded text, we will replace all the tokens left ( for the tokens to be replaced ) and split them. [ Done ]
# Till they are made up of tokens to be in the final vocab
# [ Requires Testing ?? ]
# In case of append we will not have to split the tokens . Rest process reamins the same.

import argparse
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from lib.dataset import SqliteFile, TSVData, read_parallel
from lib.misc import FileWriter, log, Filepath, make_dir
from lib.vocabs import Vocabs, Reseved
from lib.grams import GramsBase as Gb

from nlcodec import EncoderScheme, Type, load_scheme
from tqdm import tqdm

#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.prep_skip', description='Prepare skip_gram data for NMT experiment given the vocabs')
    parser.add_argument('-d', '--data_file', type=Path)
    parser.add_argument('-w', '--work_dir', type=Path, help='Path to the working directory')
    parser.add_argument('--src_len', type=int, default=0)
    parser.add_argument('--tgt_len', type=int, default=0)
    parser.add_argument('--truncate', type=bool, default=False)
    parser.add_argument('--vocabs', type=str, nargs='+', help='List of pairs : [(shared, src, tgt), Path of the vocab file]')
    parser.add_argument('--skipgram_vocabs', type=str, nargs='+', help='List of pairs : [(shared, src, tgt), Path of the skipgram file]')
    parser.add_argument('-v', '--variant', type=str, choices=['bo', 'so', 'to'], help='Variant of the experiment to prepare')
    parser.add_argument('-t', '--skipgram_tokens', type=int)
    parser.add_argument('-tm', '--token_mode', choices=['a', 'r'], type=str, default='r', help='Mode for making final vocab')
    parser.add_argument('-m', '--save_mode', type=str, default='101', help='Binary string of \
                        length 3, on bit saving file as .db, .tsv, .tsv.gz respectively')
    parser.add_argument('-x', '--save_file', type=str, help='Name of the file to store the processed data files')
    return parser.parse_args()

def save_meta(args, work_dir, work_file, vocab_files, skipgram_files):
    mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')
    mw.heading('prep_skipgram_data')
    mw.section('Work File :', [work_file.name])
    mw.section('Data File :', [args.data_file])

    mw.section('Vocab Files :', [ f'{key} : {value}' for key, value in vocab_files.items()])
    mw.section('Skipgram Files :', [ f'{key} : {value}' for key, value in skipgram_files.items()])
    mw.section('Arguments :', [
        f'Variant : {args.variant}', 
        f'Source Len : {args.src_len}',
        f'Target Len : {args.tgt_len}',
        f'Truncate : {args.truncate}',
        f'Save Mode : {args.save_mode} [ {" ".join([x for x, b in zip(suffixes,list(args.save_mode)) if b=="1"])} ]'
    ])
    mw.close(add_dashline=True)

def args_validation(args):
    assert args.data_file.exists()
    assert args.save_file is not None
    assert len(args.save_mode) == 3
    assert len(args.skipgram_vocabs) % 2 == 0
    assert len(args.vocabs) % 2 == 0

# ----------------------------------------------------------------------------

suffixes = ['.db', '.tsv', '.tsv.gz']

def make_files_dict(file_str:List[str]):
    files = dict()
    for i in range(0, len(file_str), 2):
        files[file_str[i]] = Path(file_str[i+1])
    return files

def _load_vocab(vocab_file:Filepath):
    vocab = None
    try:
        vocab = Vocabs(vocab_file)
    except Exception:
        vocab = Vocabs()
        vocab._read_in(vocab_file)
    return vocab

def _merge(vcb, sgrams, token_mode:str, ntokens:int):
    vsize, ssize = len(vcb), len(sgrams)
    if token_mode == 'r':
        vsize -= ntokens

    fvcb = Vocabs(table=vcb.table[:vsize])
    ntokens = min(ntokens, len(sgrams))
    for p in range(ntokens):
        token = sgrams.table[p]
        kids = [ vcb.table[x] for x in token.kids ]
        fvcb.append(Type(token.name, idx=vsize+p, freq=token.freq, 
                        level=token.level, kids=kids))
    return fvcb

def process(data_file:Filepath, vocab_files:Dict[str,Path], skipgram_files:List[str], 
            truncate:bool, src_len:int, tgt_len:int, work_file:Filepath, ntokens:int,
            token_mode:str='r', variant:str='so', save_mode:str='101'):

    ds = read_parallel(data_file)
    ds, vocabs = make_sgram_data(ds, vocab_files, skipgram_files, ntokens, token_mode=token_mode, variant=variant)

    if truncate:
        for p in range(len(ds)):
            src = ds.lists['src'][p]
            ds.lists['src'][p] = src[:src_len] if len(src) > src_len else src
            tgt = ds.lists['tgt'][p]
            ds.lists['tgt'][p] = tgt[:tgt_len] if len(tgt) > tgt_len else tgt

    work_dir = work_file.parent
    for key, vcb in vocabs.items():
        vcb.save(work_dir / Path(f'nlcodec.{key}.model'))

    save_flags = [ False if x=='0' else True for x in list(save_mode)]

    for save_flag, suffix in zip(save_flags, suffixes):
        log('Reading parallel recs', 2)
        recs = ((id2np(src), id2np(tgt)) for src, tgt in ds)
        if save_flag:
            save_file = work_file.with_suffix(suffix)
            log(f'Writing : {save_file.name}', 2)
            write_parallel_recs(recs, save_file)

def make_sgram_data(dataset, vocab_files, sgram_files, ntokens:int, 
                    token_mode:str='r', variant:str='so'):

    codecs = { k: load_scheme(v) for k,v in vocab_files.items() if v is not None }
    shared = True if 'shared' in codecs.keys() else False
    size = len(codecs['shared'] if shared else codecs['src'])

    skipgrams = { k: Vocabs.trim(_load_vocab(v), ntokens) for k,v in sgram_files.items() if v is not None }
    
    if shared:
        variant = 'bo'

    if variant == 'so':
        mkeys = ['src']
    elif variant == 'to':
        mkeys = ['tgt']
    else:
        mkeys = ['src', 'tgt']

    print('Encoding Skip Grams')
    for key in mkeys:
        codec = codecs['shared'] if shared else codecs[key]
        sgrams = skipgrams['shared'] if shared else skipgrams[key]
        for p in range(len(dataset)):
            seq = dataset.lists[key][p]
            dataset.lists[key][p] =_encode(seq, sgrams, codec)

    if token_mode == 'r':
        max_idx = size - ntokens
        for key in mkeys:
            codec = codecs['shared'] if shared else codecs[key]
            sgrams = skipgrams['shared'] if shared else skipgrams[key]
            offset = len(sgrams)
            replace = _get_replace_dict(codec, min(ntokens, offset))
            for p in range(len(dataset)):
                seq = dataset.lists[key][p]
                seq = _remove(seq, replace)
                seq = _shift(seq, offset, size)
                dataset.lists[key][p] = seq  

    vocabs = dict()
    if shared:
        mkeys = ['shared']

    for key in mkeys:
        vcb = Vocabs(vocab_files[key])
        vocabs[key] = _merge(vcb, skipgrams[key], token_mode, ntokens)

    return dataset, vocabs

def id2np(indices):
    return np.array(indices, dtype=np.int32)

def _get_replace_dict(codec, ntokens:int):
    size = len(codec.table)
    replace = {size-(x+1) : [] for x in range(ntokens)}
    for idx in range(size-ntokens, size):
        kids = [ k.idx for k in codec.table[idx].kids ]
        replace[idx] = _expand(kids, codec, replace.keys())
    return replace
    
def _expand(array, codec, keys):
    narray = []
    for x in array:
        if x in keys:
            kids = [k.idx for k in codec.table[x].kids]
            narray.extend(_expand(kids, codec, keys))
        else:
            narray.append(x)
    return narray

def _encode(seq, sgrams, codec):
    stok = Reseved.UNK_TOK[0] + Reseved.SPACE_TOK[0]
    size = len(codec.table)
    base = size + len(sgrams) + 1
    spairs = { x.idx : list(map(int, x.kids)) for x in sgrams }
    shashs = { Gb._hash(v, base) : k for k,v in spairs.items() }
    nseq = []
    flag = 0
    for i, x in enumerate(seq):
        if flag:
            flag -= 1
            continue
        chash = Gb._hash([seq[i], seq[i+2]], base) if i+2<len(seq) else -1
        if chash in shashs.keys():
            nseq.append(size + shashs[chash])
            nseq.append(seq[i+1])
            nseq.append(1)
            flag = 2
        else:
            nseq.append(x)
    return nseq

def _remove(seq, replace_dict):
    nseq = []
    for x in seq:
        if x in replace_dict.keys():
            nseq.extend(replace_dict[x])
        else:
            nseq.append(x)
    return nseq

def _shift(seq, offset, size):
    for p, x in enumerate(seq):
        if x >= size:
            seq[p] = x - offset
    return seq

def write_parallel_recs(records, path:Union[str, Path]):
    if path.name.endswith('.db'):
        SqliteFile.write(path, records)
    else:
        TSVData.write_parallel_recs(records, path)

def main():
    log('Starting Script : prep_skipgram_data')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    wdir = make_dir(args.work_dir)

    vocab_files = make_files_dict(args.vocabs)
    skipgram_files = make_files_dict(args.skipgram_vocabs)

    work_file = wdir / Path(f'{str(args.save_file)}.model')

    log('Processing data files', 1)
    process(args.data_file, vocab_files, skipgram_files, 
            args.truncate, args.src_len, args.tgt_len, work_file, 
            args.skipgram_tokens, args.token_mode, args.variant, 
            args.save_mode)
    log('Process completed')
    save_meta(args, wdir, work_file, vocab_files, skipgram_files)
    log('Writing meta')
    
if __name__ == "__main__":
    main()
