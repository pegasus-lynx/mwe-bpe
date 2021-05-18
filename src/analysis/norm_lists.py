import argparse
from pathlib import Path
from typing import Dict, List, Union

from nlcodec import load_scheme, Reseved

from lib.misc import FileReader, FileWriter, log, make_dir
from lib.misc import ScriptFuncs as Sf
from lib.vocabs import Vocabs
from lib.stat import StatLib
from lib.stat import BaseFuncs as Bf
from lib.grams import GramsBase as Gb
from lib.grams import GramsNormalizer as Gn
from lib.dataset import read_parallel, Dataset


#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='analysis.prep_list')
    parser.add_argument('-d', '--data_file', type=Path, help='List of data files to be used for comparision.')
    parser.add_argument('-w', '--work_dir', type=Path, help='Path to the working directory')
    parser.add_argument('-b', '--bpe_vocabs', type=str, nargs='+')
    parser.add_argument('-s', '--word_vocabs', type=str, nargs='+')
    parser.add_argument('-t', '--trigrams', type=str, nargs='+')
    parser.add_argument('--metric', type=str, choices=['pmi', 'ngdf'])
    parser.add_argument('--shared', type=bool, default=False)
    parser.add_argument('--norm', type=str, choices=['min', 'bifreq', 'freq', 'avg'])
    parser.add_argument('-g', '--gram_type', type=str, choices=['tgrams', 'sgrams'])
    return parser.parse_args()

def save_meta(args, work_dir, trigram_files):
    mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')
    mw.heading('trigram_bigram_comparision')
    mw.section('Work Dir :', [work_dir])
    mw.section('Data Files :', [args.data_file])
    mw.section('Trigram Files :', [f'{k} : {v}' for k,v in trigram_files.items()])
    mw.section('Arguments :', [
        f'Shared : {args.shared}',
        f'Norm : {args.norm}'
    ])
    mw.close(add_dashline=True)

def args_validation(args):
    assert args.work_dir is not None
    assert args.data_file.exists()
    assert len(args.trigrams) % 2 == 0
    assert len(args.bpe_vocabs) % 2 == 0
    assert len(args.word_vocabs) % 2 == 0
    assert args.norm in ['min', 'bifreq', 'freq', 'avg']

# ----------------------------------------------------------------------------

def load_vocabs(vocab_files):
    vocabs = dict()
    for k,v in vocab_files.items():
        vcb = Vocabs()
        vcb._read_in(v)
        vocabs[k] = vcb
    return vocabs

def _load_list(list_file, bpe):
    base = len(bpe)+1
    vals = dict()
    with open(list_file, 'r') as fr:
        for line in fr:
            name, val = line.strip().split('\t')
            if '*' in name:
                pieces = name.split('*')
            else:
                name = name.replace(Reseved.SPACE_TOK[0], Reseved.SPACE_TOK[0]+' ')
                pieces = name.split()
            wlist = [bpe.index(x) for x in pieces]
            hv = Bf._hash(wlist, base)
            vals[hv] = float(val)
    return vals

def load_lists(list_files, bpes):
    lists = dict()
    # print(type(list_files))
    for k,v in list_files.items():
        lists[k] = _load_list(v, bpes[k])
    return lists

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

def derive_bigrams(corps, trigram_vals, bpe, words):
    bigram_freqs = dict()
    
    base = len(bpe)+1
    bitoks, indices = set(), set()
    for hv in trigram_vals.keys():
        wlist = Bf._unhash(hv, base)
        bilist = [Bf._hash(wlist[i:i+2], base) for i in range(2)]
        for x in wlist:
            indices.add(x)
        for x in bilist:
            bitoks.add(x)
    bigram_freqs = {x:0 for x in bitoks}
    for corp in corps:
        for sent in corp:
            wordseq = Gb._make_mask(sent, indices)
            cwords = Gb._cumulate(wordseq)
            for i, wl in enumerate(cwords):
                if wl >= 2:
                    hash_val = Gb._hash(sent[i:i+2], base)
                    if hash_val in bitoks:
                        bigram_freqs[hash_val] += 1
    return bigram_freqs
            

def normalize(trigram_vals, bigram_freqs, bpe, words, norm, metric, nsents):
    if norm == 'min' or norm == 'avg':
        ntokens = StatLib.ntokens(words)
        bigram_probs = StatLib.freqs2probs(bigram_freqs, ntokens - nsents)
        bigram_vals = StatLib.calculate_metric(metric, bigram_probs, bpe, words)
        if norm == 'min':
            return Gn.min_normalize(trigram_vals, bigram_vals, bpe)
        return Gn.avg_normalize(trigram_vals, bigram_vals, bpe)
    elif norm == 'bifreq':
        return Gn.bifreq_normalize(trigram_vals, bigram_freqs, bpe)
    elif norm == 'freq':
        return Gn.freq_normalize(trigram_vals, bpe, words)
    else:
        return None

def norm_lists(data_file, work_dir, trigram_files, bpe_files, word_files, norm, metric, shared=False):
    ds = Dataset(['src', 'tgt'])
    ds.add(read_parallel(data_file))
    nsents = len(ds.lists['src'])

    bpes = load_vocabs(bpe_files)
    words = load_vocabs(word_files)
    trigram_lists = load_lists(trigram_files, bpes)

    keys = ['src', 'tgt']
    if shared:
        keys = ['shared']
        nsents = 2*nsents

    for key in keys:
        if key == 'shared':
            corps = ds.lists.values()
        else:
            corps = [ds.lists[key]]
        bigram_freqs = derive_bigrams(corps, trigram_lists[key], bpes[key], words[key])
        norm_vals = normalize(trigram_lists[key], bigram_freqs, bpes[key], words[key], norm, metric, nsents)
        norm_list = StatLib._sort(norm_vals, False if metric=='ngdf' else True)
        save_list(norm_list, bpes[key], work_dir / Path(f'norm.{norm}.{trigram_files[key].name}'))


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
    trigram_files = Sf.make_files_dict(args.trigrams)
    Sf.validate_vocab_files(trigram_files, args.shared)

    norm_lists(args.data_file, wdir, trigram_files, bpe_files, word_files, 
                args.norm, args.metric, args.shared)
    log('Process Completed')
    log('Writing Meta')

if __name__ == "__main__":
    main()
