import argparse
from os import read

from pathlib import Path
import collections as coll
from typing import Union, List, Dict, Any, Tuple, Iterator
from nlcodec.utils import IO
from nlcodec import Type

from make_single import uniq_reader_func
from lib.misc import read_conf, make_dir
from lib.schemes import load_scheme

ds_keys = ['src', 'tgt']

# -------------------------------------------------------------------------------------------------
# | Purpose of the script :                                                                       |
# | Prepare a batch of experiments for running.                                                   |
# | Batch in the respects :                                                                       |   
# |    1. Multiple vocab sizes                                                                    |
# |    2. Different numbers of ngrams / skipgrams to be added for each vocab size                 |
# |       [ Here we can add two formats : one percent based (original) | absolute fixed numbers]  |
# -------------------------------------------------------------------------------------------------

def prep_vocabs(train_files:Dict[str,Path]):
    pass

def prep_data():
    pass

def prep(configs, run_dirs):
    make_dir(run_dirs / 'base')
    
    diff_vocab_sizes = configs['max_types_list']

    train_files = { 'src':configs['train_src'], 'tgt':configs['train_tgt'] }
    val_files = { 'src':configs['valid_src'], 'tgt':configs['valid_tgt'] }

    
    pass

def parse_args():
    parser = argparse.ArgumentParser(prog="make_batch", 
                description="Prepares a batch of experiments given the parameters.")
    
    ## Parameters : Conf and Dir --------------------------------------------------------------------------------------
    parser.add_argument('-c', '--conf_file', type=Path, 
                            help='Config file for preparation of the experiment')
    parser.add_argument('-w', '--runs_dir', type=Path, 
                            help='Path to the directory for storing the runs')
    
    ## Parameters : General -------------------------------------------------------------------------------------------
    parser.add_argument('--train_src', type=Path, help='Path to the train src file')
    parser.add_argument('--train_tgt', type=Path, help='Path to the train tgt file')
    parser.add_argument('--valid_src', type=Path, help='Path to the validation src file')
    parser.add_argument('--valid_tgt', type=Path, help='Path to the validation tgt file')
    parser.add_argument('--truncate', type=bool, default=True, 
                            help='If true, truncates the sequence with larger lengths')
    parser.add_argument('--src_len', type=int, help='Maximum src sequence length', default=512)
    parser.add_argument('--tgt_len', type=int, help='Maximum tgt sequence length', default=512)
    parser.add_argument('--shared', type=bool, default=False, 
                            help='Flag representing if the vocabulary is shared or not')

    ## Parameters : Schemes -------------------------------------------------------------------------------------------
    parser.add_argument('--pieces', type=str, 
                            choices=['char', 'word', 'bpe', 'ngram', 'skipgram', 'mwe'], 
                            default='bpe', help='Scheme for preparing the vocabulary')
    parser.add_argument('--include_ngrams', type=int, nargs='+', 
                            help='List of ngrams to be included in the vocabulary')
    parser.add_argument('--include_skipgrams', type=int, nargs='+', 
                            help='List of skipgrams to be included in the vocab')
    parser.add_argument('--ngram_sorter', type=str, default='freq',
                            choices=['freq', 'naive_pmi', 'avg_pmi', 'min_pmi'],  
                            help='Sorter function to be used for ngrams')
    parser.add_argument('--skipgram_sorter', type=str, 
                            default='freq', choices=['freq', 'skip_pmi'],  
                            help='Sorter function to be used for skipgrams')

    ## Parameters : Batch Runs ----------------------------------------------------------------------------------------
    parser.add_argument('--max_types_list', type=int, nargs='+', 
                            help="Different vocab sizes for the batch of runs.")
    parser.add_argument('--max_ngrams_list', type=int, nargs='+', 
                            help='Maximum ngrams list to be included in the vocab')
    parser.add_argument('--max_skipgrams_list', type=int, nargs='+', 
                            help='Maximum skipgrams list to be included in the vocab')

    ## Parameters : Filtering ngrams / skipgrams ----------------------------------------------------------------------
    parser.add_argument('--min_freq', type=int, 
                            help='Minimum frequency for ngrams / sgrams / mwes')
    parser.add_argument('--min_instances', type=int, 
                            help='Minimum number of instances', default=0)
    parser.add_argument('--max_instance_probs', type=float, 
                            help='Maximum prob for an sgram instance', default=0.1)

    return parser.parse_args()

def make_configs(args):
    if args.conf_file is not None:
        configs = read_conf(args.conf_file)
    else:
        configs = validated_args(args)

    if 'include_skipgrams' in configs.keys():
        raw_skips = configs['include_skipgrams']
        skips = [raw_skips[2*i:2*(i+1)] for i in range(len(raw_skips)//2)]
        configs['include_skipgrams'] = skips
    return configs

def validated_args(args):
    configs = coll.OrderedDict()

    assert args.train_src.exists() and args.train_tgt.exists()
    configs['train_src'] = args.train_src    
    configs['train_tgt'] = args.train_tgt    

    assert args.valid_src.exists() and args.valid_tgt.exists()
    configs['valid_src'] = args.valid_src    
    configs['valid_tgt'] = args.valid_tgt

    configs['src_len'] = args.src_len
    configs['tgt_len'] = args.tgt_len
    configs['truncate'] = args.truncate

    configs['shared'] = args.shared
    configs['max_types_list'] = args.max_types_list

    configs['pieces'] = args.pieces
    configs['ngram_sorter'] = args.ngram_sorter
    configs['sgram_sorter'] = args.sgram_sorter

    if args.pieces in ['ngram', 'mwe']:
        assert args.max_ngrams is not None
        assert args.include_ngrams is not None
        configs['max_ngrams_list'] = args.max_ngrams_list
        configs['include_ngrams'] = args.include_ngrams
    if args.pieces in ['skipgram', 'mwe']:
        assert args.max_skipgrams is not None
        assert args.include_skipgrams is not None
        configs['max_skipgrams_list'] = args.max_skipgrams_list
        configs['include_skipgrams'] = args.include_skipgrams

    assert args.min_freq >= 0 and args.min_instances >= 0
    assert args.max_instance_probs <= 1.0
    configs['min_freq'] = args.min_freq
    configs['min_instances'] = args.min_instances
    configs['max_instance_probs'] = args.max_instance_probs

    return configs

def main():
    args = parse_args()
    configs = make_configs(args)
    runs_dir = make_dir(args.runs_dir)
    prep(configs, runs_dir)

if __name__ == "__main__":
    main()