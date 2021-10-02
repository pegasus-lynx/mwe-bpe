import argparse

from pathlib import Path 
import collections as coll
from typing import Union, List, Dict, Any, Tuple, Iterator
from nlcodec.utils import IO
from nlcodec import Type

from lib.misc import read_conf, make_dir
from lib.schemes import load_scheme, NgramScheme, SkipScheme, WordScheme

from make_single import uniq_reader_func

def _ngrams(configs:Dict, wdir:Path, get_flag:str, head:int=0, tail:int=0, save:bool=True):

    shared = configs['shared']
    min_freq = configs.get('min_freq', 100)
    keys = ['shared'] if shared else ['src', 'tgt']

    ngram_lists = dict()
    for key in keys:
        ngram_lists[key] = dict()

        files = [configs['train_src'], configs['train_tgt']]
        if key != 'shared':
            files = [configs[f'train_{key}']]
        corp = uniq_reader_func(*files)
        term_freqs, nlines = WordScheme.term_frequencies(corp)

        if get_flag == 'ngram_freqs':
            configs['ngram_sorter'] = 'freq'

        bigram_freqs = None
        if 'pmi' in configs['ngram_sorter']:
            bigram_freqs = NgramScheme.ngram_frequencies(corp, 2)

        for ngram in configs['include_ngrams']:
            freqs = NgramScheme.ngram_frequencies(corp, ngram)
            freqs = { k:v for k,v in freqs.items() if v > min_freq }
            sorted = NgramScheme.sorted_ngrams(freqs, term_freqs, 
                        nlines, configs['ngram_sorter'], 
                        bigram_freqs=bigram_freqs,
                        min_freq=min_freq)
            ngram_lists[ngram] = sorted

            prefix = 'freqs.'
            if get_flag == 'sorted_ngrams':
                prefix = f'sorted.{configs["ngram_sorter"]}.'

            if save:
                file_name = ''.join([
                    f'{prefix}.ngram.{ngram}.',
                    f'head.{head}.' if head > 0 else '',
                    f'tail.{tail}' if tail > 0 else '',
                    f'{key}.list'
                ])
                save_list(sorted, wdir / Path(file_name))
    return ngram_lists

def _sgrams(configs:Dict, wdir:Path, get_flag:str, head:int=0, tail:int=0, save:bool=True):

    shared = configs['shared']
    min_freq = configs.get('min_freq', 100)
    keys = ['shared'] if shared else ['src', 'tgt']

    skipgram_lists = dict()
    for key in keys:
        skipgram_lists[key] = dict()

        files = [configs['train_src'], configs['train_tgt']]
        if key != 'shared':
            files = [configs[f'train_{key}']]
        corp = uniq_reader_func(*files)
        term_freqs, nlines = WordScheme.term_frequencies(corp)

        if get_flag == 'skipgram_freqs':
            configs['skipgram_sorter'] = 'freq'

        bigram_freqs = None
        if 'pmi' in configs['skipgram_sorter']:
            bigram_freqs = NgramScheme.ngram_frequencies(corp, 2)

        for sgram in configs['include_skipgrams']:
            freqs = SkipScheme.skipgram_frequencies(corp, sgram)
            freqs = { k:v for k,v in freqs.items() if sum(v.values()) > min_freq }
            sorted = SkipScheme.sorted_sgrams(freqs, term_freqs, 
                        nlines, configs['skipgram_sorter'],
                        min_freq=min_freq)
            hash = (sgram[0]*SkipScheme.hash_prime) + sgram[1]
            skipgram_lists[hash] = sorted

            prefix = 'freqs.'
            if get_flag == 'sorted_ngrams':
                prefix = f'sorted.{configs["ngram_sorter"]}.'

            if save:
                file_name = ''.join([
                    f'{prefix}.sgram.{"-".join(list(map(str,sgram)))}.',
                    f'head.{head}.' if head > 0 else '',
                    f'tail.{tail}' if tail > 0 else '',
                    f'{key}.list'
                ])
                save_list(sorted, wdir / Path(file_name))
    return skipgram_lists

def save_list(mwe_list, save_file):
    with open(save_file, 'w') as fw:
        for ix, tup in enumerate(mwe_list):
            tok = tup[0] 
            val = tup[1]
            fw.write('\t'.join([
                str(tok.idx), tok.name, 
                str(val), str(tok.freq), 
                ' '.join(list(map(str,tok.kids))) if tok.kids else '',
                '\n' 
            ]))

def driver(args, configs:Dict, wdir:Path):

    # As only one of these should be active at a time.
    assert not (args.head > 0 and args.tail > 0)

    if args.ngram_freqs:
        get_flag = 'ngram_freqs'
        _ngrams(configs, wdir, get_flag,  head=args.head, tail=args.tail)        

    if args.sorted_ngrams:
        get_flag = 'sorted_ngrams'
        _ngrams(configs, wdir, get_flag, head=args.head, tail=args.tail)

    if args.skipgram_freqs:
        get_flag = 'skipgram_freqs'
        _sgrams(configs, wdir, get_flag, head=args.head, tail=args.tail)

    if args.sorted_skipgrams:
        get_flag = 'sorted_skipgrams'
        _sgrams(configs, wdir, get_flag, head=args.head, tail=args.tail)

    return

def parse_args():
    parser = argparse.ArgumentParser(prog="getter", description="Use the script to get different things in a file.")

    ## Parameters : Conf and Dir --------------------------------------------------------------------------------------
    parser.add_argument('-c', '--conf_file', type=Path, 
                            help='Config file for preparation of the experiment')
    parser.add_argument('-w', '--work_dir', type=Path, 
                            help='Path to the working directory for storing the run')

    ## Use Cases
    parser.add_argument('--ngram_freqs', action='store_true')
    parser.add_argument('--sorted_ngrams', action='store_true')
    parser.add_argument('--skipgram_freqs', action='store_true')
    parser.add_argument('--sorted_skipgrams', action='store_true')

    ## Other Parameters
    parser.add_argument('--head', type=int, default=-1)
    parser.add_argument('--tail', type=int, default=-1)

    return parser.parse_args()

def make_configs(conf_file):
    configs = read_conf(conf_file)

    if 'include_skipgrams' in configs.keys():
        raw_skips = configs['include_skipgrams']
        skips = [list(map(int, x.split())) for x in raw_skips]
        configs['include_skipgrams'] = skips

    return configs

def main():
    args = parse_args()
    configs = make_configs(args.conf_file)
    work_dir = make_dir(args.work_dir)
    driver(args, configs, work_dir)


if __name__ == "__main__":
    main()