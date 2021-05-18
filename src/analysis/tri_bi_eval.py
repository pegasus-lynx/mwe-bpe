import argparse
from pathlib import Path
from typing import Dict, List, Union
import math

from lib.misc import FileReader, FileWriter, log, make_dir
from lib.misc import ScriptFuncs as Sf
from nlcodec import load_scheme
from lib.vocabs import Vocabs
from lib.stat import StatLib
from lib.grams import GramsSorter
from lib.grams import GramsBase as Gb
from lib.dataset import read_parallel, Dataset

#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='analysis.tri_bi_comp', description='Comparision between trigrams and bigrams.')
    parser.add_argument('-d', '--data_file', type=Path, help='List of data files to be used for comparision.')
    parser.add_argument('-w', '--work_dir', type=Path, help='Path to the working directory')
    parser.add_argument('-v', '--vocabs', type=str, nargs='+')
    parser.add_argument('-s', '--word_vocabs', type=str, nargs='+')
    parser.add_argument('-t', '--trigrams', type=str, nargs='+')
    parser.add_argument('-b', '--bigrams', type=str, nargs='+')
    parser.add_argument('--shared', type=bool, default=False)
    parser.add_argument('--sorter', type=str, choices=['pmi', 'ngdf'])
    return parser.parse_args()

def save_meta(args, work_dir, trigram_files, bigram_files):
    mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')
    mw.heading('trigram_bigram_comparision')
    mw.section('Work Dir :', [work_dir])
    mw.section('Data Files :', [args.data_file])
    mw.section('Trigram Files :', [f'{k} : {v}' for k,v in trigram_files.items()])
    mw.section('Bigram Files :', [f'{k} : {v}' for k,v in bigram_files.items()])
    mw.close(add_dashline=True)

def args_validation(args):
    assert args.work_dir is not None
    assert args.data_file.exists()
    assert len(args.trigrams) % 2 == 0
    assert len(args.bigrams) % 2 == 0
    assert len(args.vocabs) % 2 == 0
    assert len(args.word_vocabs) % 2 == 0

# ----------------------------------------------------------------------------

def load_vocabs(vocab_files):
    vocabs = dict()
    for k,v in vocab_files.items():
        vcb = Vocabs()
        vcb._read_in(v)
        vocabs[k] = vcb
    return vocabs

def make_lists(ngrams, bpe, word, sorter:str, save_file):
    assert sorter in ['pmi', 'ngdf']
    sorter_func = GramsSorter.get_sorter(sorter)
    ntokens = StatLib.ntokens(word)
    indexes = StatLib.ngrams2matches(ngrams, bpe)
    ngfreqs = StatLib.ngrams2hashes(ngrams, bpe)
    sorted_ngrams = sorter_func(ngfreqs, bpe, word)

    with open(save_file, 'w') as fw:
        fw.write('\t'.join(['Ngrams', 'Freq', sorter, '\n']))
        for token, pair in zip(ngrams, sorted_ngrams):
            _, value = pair
            fw.write(f'{token.name}\t{token.freq}\t{value}\n')

def tri2bi_lists(corps, trigrams, bigrams, bpe, words, sorter:str, save_file):

    def _wlist2bilist(wlist, base):
        bilist = [Gb._hash(wlist[i:i+2], base) for i in range(2)]
        # bilist.append(Gb._hash([wlist[0], wlist[2]], base))
        return bilist    
    
    trifreqs = StatLib.ngrams2hashes(trigrams, bpe)
    tri_words = dict()

    base = len(bpe)+1
    indices, bitoks, skiptoks = set(), set(), set()
    for tri in trifreqs.keys():
        wlist = Gb._unhash(tri, base)
        tri_words[tri] = wlist
        for w in wlist:
            indices.add(w)
        bilist = _wlist2bilist(wlist, base)
        for bi in bilist:
            bitoks.add(bi)
        skiptoks.add(Gb._hash([wlist[0], wlist[2]], base))

    trifreqs = { k:0 for k in trifreqs.keys()}
    bifreqs = {k:0 for k in bitoks}
    skipfreqs = {k:0 for k in skiptoks}
    
    ntokens = StatLib.ntokens(words)
    nsents = sum([len(corp) for corp in corps])
    nbitokens = ntokens - nsents
    ntritokens = ntokens - 2*nsents

    for corp in corps:
        for sent in corp:
            wordseq = Gb._make_mask(sent, indices)
            cwords = Gb._cumulate(wordseq)
            for i, wl in enumerate(cwords):
                if wl >= 2:
                    hash_val = Gb._hash(sent[i:i+2], base)
                    if hash_val in bitoks:
                        bifreqs[hash_val] += 1
                if wl >= 3:
                    hash_val = Gb._hash([sent[i], sent[i+2]], base)
                    if hash_val in skiptoks:
                        skipfreqs[hash_val] += 1
                    hash_val = Gb._hash(sent[i:i+3], base)
                    if hash_val in trifreqs.keys():
                        trifreqs[hash_val] += 1

    biprobs = StatLib.freqs2probs(bifreqs, nbitokens)
    triprobs = StatLib.freqs2probs(trifreqs, ntritokens)
    skipprobs = StatLib.freqs2probs(skipfreqs, ntritokens)

    bitok_vals = StatLib.calculate_metric(sorter, biprobs, bpe, words)
    tritok_vals = StatLib.calculate_metric(sorter, triprobs, bpe, words)
    skiptok_vals = StatLib.calculate_metric(sorter, skipprobs, bpe, words)

    tri_sorted = StatLib._sort(tritok_vals, False if sorter=='ngdf' else True)
    tri_ndiff_tok = dict()
    tri_ndiff_bitok = dict()

    with open(save_file, 'w') as fw:
        for pair in tri_sorted:
            hv, val = pair
            freq = trifreqs[hv]

            vals, freqs = [], []

            wlist = tri_words[hv]
            name = ''.join([bpe.table[x].name for x in wlist])
            bilist = _wlist2bilist(wlist, base)
            skiptok = Gb._hash([wlist[0], wlist[2]], base)

            vals.extend([bitok_vals[x] for x in bilist])
            vals.append(skiptok_vals[skiptok])

            freqs.extend([bifreqs[x] for x in bilist])
            freqs.append(skipfreqs[skiptok])
            
            diff = max(freqs)-min(freqs)
            tri_ndiff_bitok[hv] = val - math.log(1+diff)
            freqs.extend([diff, tri_ndiff_bitok[hv]])

            fw.write('\t'.join([name, str(val), str(freq), '\n']))
            fw.write(''.join(['\t', '\t'.join(list(map(str,vals))) ,'\n']))
            fw.write(''.join(['\t', '\t'.join(list(map(str,freqs))) ,'\n']))

            # Token Level Freqs
            windexes = [words.index(bpe.table[x].name[:-1]) for x in wlist]
            freqs = [words.table[x].freq for x in windexes]

            diff = max(freqs)-min(freqs)
            tri_ndiff_tok[hv] = val - math.log(1+diff)
            freqs.extend([diff, tri_ndiff_tok[hv]])

            fw.write(''.join(['\t', '\t'.join(list(map(str,freqs))) ,'\n']))
            fw.write('\n')

    tri_ndtok_sorted = StatLib._sort(tri_ndiff_tok, True)
    tri_ndbitok_sorted = StatLib._sort(tri_ndiff_bitok, True)

    with open(save_file.parent / Path(f'norm.tok.{save_file.name}'), 'w') as fw:
        for hv, val in tri_ndtok_sorted:
            wlist = tri_words[hv]
            name = ''.join([bpe.table[x].name for x in wlist])
            fw.write(f'{name}\t{val}\n')

    with open(save_file.parent / Path(f'norm.bitok.{save_file.name}'), 'w') as fw:
        for hv, val in tri_ndbitok_sorted:
            wlist = tri_words[hv]
            name = ''.join([bpe.table[x].name for x in wlist])
            fw.write(f'{name}\t{val}\n')


def analyze(data_file, bpes, words, bigrams, trigrams, sorter:str, work_dir, shared:bool=False):
    ds = Dataset(['src', 'tgt'])
    ds.add(read_parallel(data_file))

    if shared:
        make_lists(bigrams['shared'], bpes['shared'], words['shared'], 
                    sorter, work_dir / Path(f'bigram.list.{sorter}.shared.tsv'))
        make_lists(trigrams['shared'], bpes['shared'], words['shared'], 
                    sorter, work_dir / Path(f'trigram.list.{sorter}.shared.tsv'))
        tri2bi_lists(ds.lists.values(), trigrams['shared'], bigrams['shared'],
                    bpes['shared'], words['shared'], sorter,
                    work_dir / Path(f'tri_bi.list.{sorter}.shared.tsv'))
    else:
        make_lists(bigrams['src'], bpes['src'], words['src'], sorter, 
                    work_dir / Path(f'bigram.list.{sorter}.src.tsv'))
        make_lists(bigrams['tgt'], bpes['tgt'], words['tgt'], sorter,
                    work_dir / Path(f'bigram.list.{sorter}.tgt.tsv'))
        make_lists(trigrams['src'], bpes['src'], words['src'], sorter, 
                    work_dir / Path(f'trigram.list.{sorter}.src.tsv'))
        make_lists(trigrams['tgt'], bpes['tgt'], words['tgt'], sorter,
                    work_dir / Path(f'trigram.list.{sorter}.tgt.tsv'))
        tri2bi_lists(ds.lists.values(), trigrams['src'], bigrams['src'],
                    bpes['src'], words['src'], sorter,
                    work_dir / Path(f'tri_bi.list.{sorter}.src.tsv'))
        tri2bi_lists(ds.lists.values(), trigrams['tgt'], bigrams['tgt'],
                    bpes['tgt'], words['tgt'], sorter,
                    work_dir / Path(f'tri_bi.list.{sorter}.tgt.tsv'))

# ----------------------------------------------------------------------------

def main():
    log('Starting Script : tri_bi_comp')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    wdir = args.work_dir
    make_dir(wdir)
    log(f'> Work Dir : {wdir}', 1)

    log(f'> Validating Vocab Files', 1)
    bpe_files = Sf.make_files_dict(args.vocabs)
    Sf.validate_vocab_files(bpe_files, args.shared)
    bpes = load_vocabs(bpe_files)

    log(f'> Validating Word Vocab Files', 1)
    word_files = Sf.make_files_dict(args.word_vocabs)
    Sf.validate_vocab_files(word_files, args.shared)
    words = load_vocabs(word_files)

    log(f'> Validating Trigram Files', 1)
    trigram_files = Sf.make_files_dict(args.trigrams)
    Sf.validate_vocab_files(trigram_files, args.shared)
    trigrams = load_vocabs(trigram_files)

    log(f'> Validating Bigram Files', 1)
    bigram_files = Sf.make_files_dict(args.bigrams)
    Sf.validate_vocab_files(bigram_files, args.shared)
    bigrams = load_vocabs(bigram_files)

    analyze(args.data_file, bpes, words, bigrams, trigrams, args.sorter, wdir, args.shared)
 
    log('Analysis Completed')
    log('Writing Meta')
    save_meta(args, wdir, trigram_files, bigram_files)
    

if __name__ == "__main__":
    main()
