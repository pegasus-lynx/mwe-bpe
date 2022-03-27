import argparse
import os
import json

from pathlib import Path 
import collections as coll
from typing import List, Dict, Tuple


from nlcodec import Type
from rtg.data.dataset import TSVData, SqliteFile

from lib.misc import read_conf, make_dir, make_file, uniq_reader_func
from nlcodec.codec import MWE_MIN_FREQ, get_scheme


ds_keys = ['src','tgt']

def prep_vocabs(train_files:Dict[str,Path], 
        vocab_files:Dict[str,Path], pieces:Dict[str,str], 
        shared:bool, vocab_sizes:Dict[str,int], 
        global_list_files:Dict[str, Path], 
        mwe_tokens=['bi', 'tri', 'ski'],
        max_mwes=0, change_mode='replace'):
    
    keys = ['shared'] if shared else ['src', 'tgt']
    for key in keys:
        scheme = get_scheme(pieces[key])
        global_lists = json.loads(global_list_files[key].read_text())
        mwe_lists = { x:global_lists[x] for x in mwe_tokens if x in global_lists.keys()}
        print("Tokens from following lists will be included : ", mwe_lists.keys())
        if key == 'shared':
            corp = uniq_reader_func(*train_files.values())
        else:
            corp = uniq_reader_func(train_files[key])

        vocab = scheme.learn(corp, vocab_sizes[key],
                                mwe_lists, max_mwes, change_mode)
        Type.write_out(vocab, vocab_files[key])

def prep_data(train_files:Dict[str, Path], val_files:Dict[str, Path], 
            vocab_files:Dict[str, Path], pieces:Dict[str, str], shared:bool, 
            src_len:int, tgt_len:int, truncate:bool, work_dir:Path, split_ratio:float=0.0):
    
    codecs = {}
    for key, fpath in vocab_files.items():
        scheme = get_scheme(pieces[key])
        if fpath.exists():
            table, _ = Type.read_vocab(fpath)
            codecs[key] = scheme(table)
    src_codec = codecs['shared' if 'shared' in codecs.keys() else 'src']
    tgt_codec = codecs['shared' if 'shared' in codecs.keys() else 'tgt']

    ## For train files
    recs = TSVData.read_raw_parallel_recs(train_files['src'], 
                train_files['tgt'], truncate, src_len, tgt_len,
                lambda x : src_codec.encode(x, split_ratio), lambda y : tgt_codec.encode(y, split_ratio))
    # TSVData.write_parallel_recs(recs, work_dir / Path('train.tsv'))
    SqliteFile.write(work_dir / Path('train.db'), recs)

    ## For validation files
    recs = TSVData.read_raw_parallel_recs(val_files['src'], 
                val_files['tgt'], truncate, src_len, tgt_len,
                src_codec.encode, tgt_codec.encode)
    TSVData.write_parallel_recs(recs, work_dir / Path('valid.tsv.gz'))

    return

def prep(configs, work_dir):
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

    global_list_files = {
        'src' : Path(configs.get('src_global_list', ".")),
        'tgt' : Path(configs.get('tgt_global_list', ".")),
        'shared' : Path(configs.get('shared_global_list', ".")),
    }

    default_scheme = configs.get('pieces', "bpe")
    pieces = { 
        'src':default_scheme, 
        'tgt':default_scheme, 
        'shared':default_scheme 
    }
    if configs.get('src_pieces') is not None:
        pieces['src'] = configs.get('src_pieces')
    if configs.get('tgt_pieces') is not None:
        pieces['tgt'] = configs.get('tgt_pieces')

    vcb_flag = work_dir / Path('_VOCABS')
    if not vcb_flag.exists():
        prep_vocabs(train_files, vocab_files, 
                pieces, shared, vocab_sizes, global_list_files, configs['mwe_tokens'])
        make_file(vcb_flag)

    data_flag = work_dir / Path('_DATA')
    if not data_flag.exists():
        prep_data(train_files, val_files, vocab_files, pieces,
            shared, configs['src_len'], configs['tgt_len'],
            configs['truncate'], data_dir, split_ratio=configs['split_ratio'])
        make_file(data_flag)

    make_file(work_dir / Path('_PREPARED'))

    if data_flag.exists():
        os.remove(data_flag)
    if vcb_flag.exists():
        os.remove(vcb_flag)

def parse_args():
    parser = argparse.ArgumentParser(prog="make_single", description="Prepares a single experiment using the conf file")

    parser.add_argument('-c', '--conf_file', type=Path, 
                            help='Config file for preparation of the experiment')
    parser.add_argument('-w', '--work_dir', type=Path, 
                            help='Path to the working directory for storing the run')
    return parser.parse_args()

def make_configs(args):
    configs = read_conf(args.conf_file)

    if 'include_skipgrams' in configs.keys():
        raw_skips = configs['include_skipgrams']
        skips = [list(map(int, x.split())) for x in raw_skips]
        configs['include_skipgrams'] = skips
    
    return configs

def validate_configs(configs):

    def assert_key(key, configs):
            assert key in configs.keys(), str.format("Key : {} is not defined in configs", key)
            assert configs[key] is not None, str.format("Value can not be none. [ Key : {} ]", key)

    def assert_keys(keys, configs):
        for key in keys:
            assert_key(key, configs)

    for key in ["train_src", "train_tgt", "valid_src", "valid_tgt"]:
        assert_key(key, configs)
        assert configs[key].exists(), str.format("File : {} does not exist. [ Key : {} ]", configs[key], key)

    assert_keys(["shared", "min_freq", "pieces"], configs)

    if configs["shared"]:
        assert_key("max_types", configs)
    else:
        assert_keys(["max_src_types", "max_tgt_types"], configs)

    assert configs["min_freq"] >= 0, str.format("Value of min_freq must be >=0")

    allowed_pieces = ['ngram', 'skipgram', 'mwe', 'bpe', 'word', 'char']
    assert configs["pieces"] in allowed_pieces, str.format(
                    "Value {} is not supported. Allowed values are : {}", 
                    configs['pieces'], str(allowed_pieces))

    if configs["pieces"] in ['ngram', 'mwe']:
        assert_keys(["max_ngrams", "include_ngrams"], configs)
    if configs["pieces"] in ['skipgram', 'mwe']:
        assert_keys(["max_skipgrams", "include_skipgrams", "min_instances", "max_instance_probs"], configs)  
        assert configs["min_instances"] >= 0, str.format("Min instances of distinct skipgrams should be >= 0")
        assert configs["max_instance_probs"] <= 1.0, str.format("Max allowed prob of any instance of skipgram, word pair must be less than <= 1.0")

def main():
    args = parse_args()

    assert args.conf_file is not None, "Conf file is required for this"

    configs = make_configs(args)
    work_dir = make_dir(args.work_dir)
    
    # The work of this block has been moved to make_conf file.
    # Keeping this block active just for now.
    if args.conf_file is not None:
        prep_file = work_dir / Path('prep.yml')
        if not prep_file.exists():
            make_file(prep_file , args.conf_file)
    
    prep(configs, work_dir)

if __name__ == "__main__":
    main()
