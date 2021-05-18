import argparse
from pathlib import Path
from typing import Dict, List, Union

from lib.misc import FileReader, FileWriter, log, make_dir
from lib.misc import ScriptFuncs as Sf
from nlcodec import load_scheme
from lib.vocabs import Vocabs
from lib.stat import StatLib
from lib.stat import BaseFuncs as Bf
from lib.dataset import read_parallel, Dataset


#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='analysis.prep_list')
    parser.add_argument('-d', '--data_file', type=Path, help='List of data files to be used for comparision.')
    parser.add_argument('-w', '--work_dir', type=Path, help='Path to the working directory')
    parser.add_argument('-b', '--bpe_vocabs', type=str, nargs='+')
    parser.add_argument('-s', '--word_vocabs', type=str, nargs='+')
    parser.add_argument('-g', '--grams', type=str, nargs='+')
    parser.add_argument('-n', '--ng', type=int, default=2)
    parser.add_argument('-t', '--gram_type', type=str, choices=['ngrams', 'sgrams'])
    parser.add_argument('--shared', type=bool, default=False)
    parser.add_argument('--metric', type=str, choices=['pmi', 'ngdf'])
    return parser.parse_args()

def save_meta(args, work_dir, gram_files):
    mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')
    mw.heading('prep_list')
    mw.section('Work Dir :', [work_dir])
    mw.section('Data Files :', [args.data_file])
    mw.section('Gram Files :', [f'{k} : {v}' for k,v in gram_files.items()])
    mw.close(add_dashline=True)

def args_validation(args):
    assert args.work_dir is not None
    assert args.data_file.exists()
    assert len(args.grams) % 2 == 0
    assert len(args.bpe_vocabs) % 2 == 0
    assert len(args.word_vocabs) % 2 == 0
    assert args.metric in ['pmi', 'ngdf']

# ----------------------------------------------------------------------------

def load_vocabs(vocab_files):
    vocabs = dict()
    for k,v in vocab_files.items():
        vcb = Vocabs()
        vcb._read_in(v)
        vocabs[k] = vcb
    return vocabs

def save_list(nglist, bpe, save_file, is_sgram=False):
    base = len(bpe)+1
    with open(save_file, 'w') as fw:
        for pair in nglist:
            hv, val = pair
            wlist = Bf._unhash(hv, base)
            name = ''.join([bpe.table[x].name for x in wlist])
            if is_sgram:
                name = '*'.join([bpe.table[x].name for x in wlist])
            fw.write('\t'.join([name, str(val), '\n']))

def make_nglists(ngrams, bpe, words, metric, nsents, ng):
    ntokens  = StatLib.ntokens(words)
    ngfreqs  = StatLib.ngrams2hashes(ngrams, bpe)
    ngprobs  = StatLib.freqs2probs(ngfreqs, ntokens - (ng-1)*nsents)
    ngvals   = StatLib.calculate_metric(metric, ngprobs, bpe, words)
    ngsorted = StatLib._sort(ngvals, False if metric=='ngdf' else True)
    return ngsorted 

def make_sglists(sgrams, bpe, words, metric, nsents):
    ntokens = StatLib.ntokens(words)
    sgfreqs = StatLib.sgrams2hashes(sgrams, bpe)
    sgprobs = StatLib.freqs2probs(sgfreqs, ntokens - 2*nsents)
    sgvals  = StatLib.calculate_metric(metric, sgprobs, bpe, words)
    sgsorted = StatLib._sort(sgvals, False if metric=='ngdf' else True)
    return sgsorted

def make_lists(data_file, work_dir, gram_files, bpe_files, word_files, 
                metric, gram_type, ng, shared=False):
    ds = Dataset(['src', 'tgt'])
    ds.add(read_parallel(data_file))
    nsents = len(ds.lists['src'])
    
    grams = load_vocabs(gram_files)
    bpes = load_vocabs(bpe_files)
    words = load_vocabs(word_files)
    
    keys = ['src', 'tgt']
    if shared:
        keys = ['shared']
        nsents = 2*nsents

    for key in keys:
        if gram_type == 'ngrams':
            nglist = make_nglists(grams[key], bpes[key], words[key], metric, nsents, ng)
            save_list(nglist, bpes[key], work_dir / Path(f'lists.{gram_files[key].name}'))
        elif gram_type == 'sgrams':
            sglist = make_sglists(grams[key], bpes[key], words[key], metric, nsents)
            save_list(sglist, bpes[key], work_dir / Path(f'lists.{gram_files[key].name}'), True)

def main():
    log('Starting Script : make_list')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    wdir = args.work_dir
    make_dir(wdir)
    log(f'> Work Dir : {wdir}', 1)

    log(f'> Validating Vocab Files', 1)
    bpe_files = Sf.make_files_dict(args.bpe_vocabs)
    Sf.validate_vocab_files(bpe_files, args.shared)

    log(f'> Validating Word Vocab Files', 1)
    word_files = Sf.make_files_dict(args.word_vocabs)
    Sf.validate_vocab_files(word_files, args.shared)

    log(f'> Validating Ngram Files', 1)
    gram_files = Sf.make_files_dict(args.grams)
    Sf.validate_vocab_files(gram_files, args.shared)

    make_lists(args.data_file, wdir, gram_files, bpe_files, word_files, 
                args.metric, args.gram_type, args.ng, args.shared)
    log('Process Completed')
    log('Writing Meta')

if __name__ == "__main__":
    main()
