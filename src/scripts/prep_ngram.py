import argparse
from pathlib import Path
from itertools import zip_longest
from typing import Union, List, Dict, Set, Tuple
from nlcodec import load_scheme, EncoderScheme, Type
import numpy as np
from tqdm import tqdm

from dataset import Dataset
from misc import FileWriter

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.prep', description='Prepare data for NMT experiment given the vocabs')
    parser.add_argument('-t', '--data_file', type=Path, help='Path of the src file')
    parser.add_argument('-w', '--work_dir', type=Path, help='Path to the working directory')
    parser.add_argument('-v', '--vocab', type=str, nargs='+', help='Enter vocabs as pairs : type [shared, src, tgt] and path to the file')
    parser.add_argument('-b', '--bpe_words', type=str, nargs='+', help='Similar to vocab arg')
    parser.add_argument('-m', '--min_freq', default=50, type=int, help='Minimum frequency upto which bi-grams are to be considered')
    parser.add_argument('-s', '--shared', type=bool, default=False)
    return parser.parse_args()

def args_validation(args):
    assert args.data_file.exists()
    # assert args.bpe_words.exists()
    assert args.work_dir is not None
    assert len(args.vocab) % 2 == 0

def validate_vocab_files(vocab_files:Dict[str,Union[Path,str]], shared):
    if shared:
        assert 'shared' in vocab_files.keys()
        assert vocab_files['shared'].exists()
    else:
        assert 'src' in vocab_files.keys()
        assert 'tgt' in vocab_files.keys()
        assert vocab_files['src'].exists()
        assert vocab_files['tgt'].exists()

def read_parallel(parallel_file:Path):
    ds = Dataset(['src', 'tgt'])
    with open(parallel_file, 'r') as fr:
        print('\t\t > Reading parallel data file ...')
        for line in tqdm(fr):
            x, y = line.strip().split('\t')
            src = list(map(int, x.split()))
            tgt = list(map(int, y.split()))
            ds.append([src, tgt], keys=['src', 'tgt'])
    return ds


def load_indexes(bpe_words_file:Path):
    indexes = set()
    with open(bpe_words_file, 'r') as fr:
        for line in fr:
            line = line.strip()
            if line.startswith('#'):
                continue
            cols = line.split('\t')
            idx = cols[0]
            indexes.add(int(idx))
    return indexes

def get_bigrams(bpe_word_file:Path, corps:List[List[Union[str, int]]], min_freq:int=50):
    bigrams = dict()
    bpe_indexes = load_indexes(bpe_word_file)
    temp = max(bpe_indexes)
    # print(temp, print(type(temp)))
    max_idx = max(bpe_indexes)+1
    for corp in corps:
        for sent in tqdm(corp):
            words = [False] * len(sent)
            for i, token in enumerate(sent):
                if token in bpe_indexes:
                    words[i] = True
            for i in range(len(words)-1):
                if words[i] and words[i+1]:
                    pos = sent[i]*max_idx + sent[i+1]
                    if pos not in bigrams.keys():
                        bigrams[pos] = 0
                    bigrams[pos] += 1
    bigram_list = [(key,val) for key,val in bigrams.items()]
    bigram_list.sort(key=lambda x : x[1], reverse=True)
    bigram_toks = [((x//max_idx, x%max_idx), freq) for x, freq in bigram_list if freq >= min_freq]
    return bigram_toks

def add_bigrams(vocab_file:Path, bigrams:List[Tuple[Tuple[int,int], int]], save_dir:Path):
    scheme = load_scheme(vocab_file)
    start_idx = len(scheme.table)
    offset = 0
    for tok_pair, freq in tqdm(bigrams):
        name = scheme.table[tok_pair[0]].name + scheme.table[tok_pair[1]].name
        scheme.table.append(Type(name, 1, start_idx+offset, freq, [scheme.table[x] for x in tok_pair]))
        offset += 1
    Type.write_out(scheme.table, save_dir / Path(f'{vocab_file.stem}.mod.model'))

def modify(data_file:Path, work_dir:Path, vocab_files:Dict[str,Union[str,Path]], 
            bpe_word_files:Dict[str,Union[str,Path]], min_freq:int=50, shared:bool=False):    
    dataset = read_parallel(data_file)
    bigrams = dict()
    if shared:
        bigrams['shared'] = get_bigrams(bpe_word_files['shared'], [dataset.lists['src'], dataset.lists['tgt']], min_freq=min_freq)
        add_bigrams(vocab_files['shared'], bigrams['shared'], work_dir / Path('_vocabs/'))
    else:
        bigrams['src'] = get_bigrams(bpe_word_files['src'], [dataset.lists['src']], min_freq=min_freq)
        bigrams['tgt'] = get_bigrams(bpe_word_files['tgt'], [dataset.lists['tgt']], min_freq=min_freq)
        add_bigrams(vocab_files['src'], bigrams['src'], work_dir / Path('_vocabs/'))
        add_bigrams(vocab_files['tgt'], bigrams['tgt'], work_dir / Path('_vocabs/'))
    # write_parallel(dataset, work_dir /  Path(f'{data_file.stem}.mod.tsv'))

def main():
    print('Running Scripts ...')
    args = parse_args()
    args_validation(args)
    print('\t > Loaded Args')

    if not args.work_dir.exists():
        args.work_dir.mkdir()
        print('\t > Making work dir ...')

    vocab_files = dict()
    for i in range(0,len(args.vocab),2):
        vocab_files[args.vocab[i]] = Path(args.vocab[i+1])
    # print(vocab_files)

    bpe_word_files = dict()
    # print(args)
    for i in range(0,len(args.bpe_words),2):
        bpe_word_files[args.bpe_words[i]] = Path(args.bpe_words[i+1])
    # print(bpe_word_files)

    print('\t > Validating vocab files ...')    
    validate_vocab_files(vocab_files, args.shared)

    print('\t > Modifying the dataset ...')
    modify(args.data_file, args.work_dir, vocab_files, bpe_word_files, min_freq=args.min_freq, shared=args.shared)
    print('Completed.')
    
if __name__ == "__main__":
    main()