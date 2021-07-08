import argparse

from pathlib import Path 
import collections as coll
from typing import Union, List, Dict, Any, Tuple, Iterator
from nlcodec.utils import IO
from nlcodec import Type

from lib.misc import read_conf, make_dir
from lib.schemes import load_scheme, NgramScheme, SkipScheme

from make_single import uniq_reader_func

def _ngram_freqs(configs:Dict, wdir:Path, head:int=0, tail:int=0):

    assert 'shared' in configs.keys()

    shared = configs['shared']
    min_freq = configs.get('min_freq', 100)
    keys = ['shared'] if shared else ['src', 'tgt']

    # Just to avoid error in naming
    if head > 0:
        tail = 0

    ngram_freqs = dict()
    for key in keys:
        if key == 'shared':
            corp = uniq_reader_func([configs['train_src'], configs['train_tgt']])
        else:
            corp = uniq_reader_func(configs[f'train_{key}'])

        for ngram in configs['include_ngrams']:
            freqs = NgramScheme.ngram_frequencies(corp, ngram)
            freqs = [ (k,v) for k,v in freqs.items() if v > min_freq ]
            freqs.sort(lambda x: x[1], reverse=True)
            
            if head > 0:
                freqs = freqs[:head]    
            elif tail > 0:
                freqs = freqs[len(freqs)-tail:]
            
            ngram_freqs[ngram] = freqs

        for ngram in ngram_freqs.keys():
            file_name = ''.join([
                f'freqs.ngram.{ngram}.',
                f'head.{head}.' if head > 0 else '',
                f'tail.{tail}' if tail > 0 else '',
                'list'
            ])
            save_file = wdir / Path(file_name)

            with open(save_file, 'r') as fw:
                for pair in ngram_freqs[ngram]:
                    fw.write('\t'.join(pair))
                    fw.write('\n')



def driver(args, configs:Dict, wdir:Path):

    if args.ngram_freqs:
        _ngram_freqs(configs, wdir)        

    if args.sorted_ngrams:
        pass

    if args.skpgram_freqs:
        pass

    if args.sorted_skipgrams:
        pass

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

def make_configs(args):
    configs = read_conf(args.conf_file)

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