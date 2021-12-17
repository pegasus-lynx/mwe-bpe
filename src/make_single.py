import argparse
import os
from json import load

from pathlib import Path 
import collections as coll
from typing import List, Dict, Tuple


from nlcodec import Type
from rtg.data.dataset import TSVData, SqliteFile

from lib.misc import read_conf, make_dir, make_file, uniq_reader_func
from lib.schemes import load_scheme, MWE_MIN_FREQ

ds_keys = ['src','tgt']

def prep_vocabs(train_files:Dict[str,Path], 
        vocab_files:Dict[str,Path], pieces:str, 
        shared:bool, vocab_sizes:Dict[str,int], 
        ngram_sorter:str, skipgram_sorter:str,
        max_ngrams:int=0, include_ngrams:List[int]=None, 
        max_skipgrams:int=0, include_skipgrams:List[Tuple[int,int]]=None, 
        min_freq:int=MWE_MIN_FREQ, min_instances:int=0, max_instance_probs:float=1.0):
    
    scheme = load_scheme(pieces)
    keys = ['shared'] if shared else ['src', 'tgt']
    for key in keys:
        if key == 'shared':
            corp = uniq_reader_func(*train_files)
        else:
            corp = uniq_reader_func(train_files[key])

        vocab = scheme.learn(corp, vocab_size=vocab_sizes[key],
                            ngrams=include_ngrams, max_ngrams=max_ngrams, 
                            sgrams=include_skipgrams, max_sgrams=max_skipgrams,
                            ngram_sorter=ngram_sorter, min_freq=min_freq, 
                            min_instances=min_instances,
                            skipgram_sorter=skipgram_sorter,
                            max_instance_probs=max_instance_probs)
        Type.write_out(vocab, vocab_files[key])

def prep_data(train_files:Dict[str, Path], val_files:Dict[str, Path], 
            vocab_files:Dict[str, Path], pieces:str, shared:bool, 
            src_len:int, tgt_len:int, truncate:bool, work_dir:Path):
    scheme = load_scheme(pieces)
    
    codecs = {}
    for key, fpath in vocab_files.items():
        if fpath.exists():
            table, _ = Type.read_vocab(fpath)
            codecs[key] = scheme(table)
    src_codec = codecs['shared' if 'shared' in codecs.keys() else 'src']
    tgt_codec = codecs['shared' if 'shared' in codecs.keys() else 'tgt']

    ## For train files
    recs = TSVData.read_raw_parallel_recs(train_files['src'], 
                train_files['tgt'], truncate, src_len, tgt_len,
                src_codec.encode, tgt_codec.encode)
    # TSVData.write_parallel_recs(recs, work_dir / Path('train.tsv'))
    SqliteFile.write(work_dir / Path('train.db'), recs)

    ## For validation files
    recs = TSVData.read_raw_parallel_recs(val_files['src'], 
                val_files['tgt'], truncate, src_len, tgt_len,
                src_codec.encode, tgt_codec.encode)
    TSVData.write_parallel_recs(recs, work_dir / Path('valid.tsv.gz'))

    return

def prep(configs, work_dir, rtg_config_file=None):
    shared = configs['shared']
    data_dir = make_dir(work_dir / 'data')

    vocab_sizes = {
        'src' : configs.get('max_src_types', 0),
        'tgt' : configs.get('max_tgt_types', 0),
        'shared' : configs.get('max_types', 0)
    }

    train_files = { 'src' : Path(configs['train_src']), 
                    'tgt' : Path(configs['train_tgt']) }
    val_files = {   'src' : Path(configs['valid_src']), 
                    'tgt' : Path(configs['valid_tgt']) }

    vocab_files = {
        'src' : data_dir / 'nlcodec.src.model',
        'tgt' : data_dir / 'nlcodec.tgt.model',
        'shared': data_dir / 'nlcodec.shared.model'
    }

    vcb_flag = work_dir / Path('_VOCABS')
    if not vcb_flag.exists():
        prep_vocabs(train_files, vocab_files, 
                configs['pieces'], shared, vocab_sizes, 
                configs.get('ngram_sorter', 'freq'),
                configs.get('skipgram_sorter', 'freq'),
                configs.get('max_ngrams', 0), 
                configs.get('include_ngrams', None), 
                configs.get('max_skipgrams', 0), 
                configs.get('include_skipgrams', None),
                configs.get('min_freq', 0),
                configs.get('min_instances', 0),
                configs.get('max_instance_probs', 1))
        make_file(vcb_flag)

    data_flag = work_dir / Path('_DATA')
    if not data_flag.exists():
        prep_data(train_files, val_files, vocab_files, configs['pieces'],
            shared, configs['src_len'], configs['tgt_len'],
            configs['truncate'], data_dir)
        make_file(data_flag)

    make_file(work_dir / Path('_PREPARED'))
    
    if rtg_config_file is not None:
        make_file(work_dir / Path('conf.yml'), rtg_config_file)

    if vcb_flag.exists():
        os.remove(vcb_flag)
    if data_flag.exists():
        os.remove(data_flag)

def parse_args():
    parser = argparse.ArgumentParser(prog="make_single", description="Prepares a single experiment using the conf file")

    ## Parameters : Conf and Dir --------------------------------------------------------------------------------------
    parser.add_argument('-c', '--conf_file', type=Path, 
                            help='Config file for preparation of the experiment')
    parser.add_argument('-w', '--work_dir', type=Path, 
                            help='Path to the working directory for storing the run')
    parser.add_argument('-r', '--rtg_config', type=Path, help='Path to the RTG config file to be copied to the prepared run dir.')

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

    ## Parameters : Single Runs ---------------------------------------------------------------------------------------
    parser.add_argument('--max_types', type=int, help='Maximum shared types ( if shared == True )')
    parser.add_argument('--max_src_types', type=int, help='Maximum src vocab types')
    parser.add_argument('--max_tgt_types', type=int, help='Maximum tgt vocab types')

    ## Parameters : Schemes -------------------------------------------------------------------------------------------
    parser.add_argument('--pieces', type=str, 
                            choices=['char', 'word', 'bpe', 'ngram', 'skipgram', 'mwe'], 
                            default='bpe', help='Scheme for preparing the vocabulary')
    parser.add_argument('--ngram_sorter', type=str, default='freq',
                            choices=['freq', 'naive_pmi', 'avg_pmi', 'min_pmi'],  
                            help='Sorter function to be used for ngrams')
    parser.add_argument('--include_ngrams', type=int, nargs='+', 
                            help='List of ngrams to be included in the vocabulary')
    parser.add_argument('--max_ngrams', type=int, 
                            help='Maximum ngrams to be included in the vocab')
    parser.add_argument('--skipgram_sorter', type=str, 
                            default='freq', choices=['freq', 'skip_pmi'],  
                            help='Sorter function to be used for skipgrams')
    parser.add_argument('--include_skipgrams', type=int, nargs='+', 
                            help='List of skipgrams to be included in the vocab')
    parser.add_argument('--max_skipgrams', type=int, 
                            help='Maximum skipgrams to be included in the vocab')

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
        from_conf_file = True
    else:
        configs = validated_args(args)
        from_conf_file = False

    if 'include_skipgrams' in configs.keys():
        raw_skips = configs['include_skipgrams']
        if from_conf_file:
            skips = [list(map(int, x.split())) for x in raw_skips]
        else:
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
    if args.shared:
        assert args.max_types is not None
        configs['max_types'] = args.max_types
    else:
        assert args.max_src_types is not None
        assert args.mac_tgt_types is not None
        configs['max_src_types'] = args.max_src_types
        configs['max_tgt_types'] = args.max_tgt_types

    configs['pieces'] = args.pieces
    configs['ngram_sorter'] = args.ngram_sorter
    configs['sgram_sorter'] = args.sgram_sorter

    if args.pieces in ['ngram', 'mwe']:
        assert args.max_ngrams is not None
        assert args.include_ngrams is not None
        configs['max_ngrams'] = args.max_ngrams
        configs['include_ngrams'] = args.include_ngrams
    if args.pieces in ['skipgram', 'mwe']:
        assert args.max_skipgrams is not None
        assert args.include_skipgrams is not None
        configs['max_skipgrams'] = args.max_skipgrams
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
    work_dir = make_dir(args.work_dir)
    
    if args.conf_file is not None:
        make_file(work_dir / Path('prep.yml'), args.conf_file)
    
    prep(configs, work_dir, args.rtg_config)

if __name__ == "__main__":
    main()