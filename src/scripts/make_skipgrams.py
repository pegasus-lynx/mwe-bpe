import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

from lib.misc import FileWriter, log, Filepath, make_dir
from lib.vocabs import Vocabs
from lib.grams import SkipGrams
from lib.dataset import Dataset, read_parallel
from nlcodec import Reseved, Type

#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.make_skipgrams', description='Prepares skip gram vocabs for the datassets')
    parser.add_argument('-d', '--data_files', type=Path, nargs='+', help='List of processed dataset files')
    parser.add_argument('-s', '--shared', type=bool, default=False, help='True if shared vocabs')
    parser.add_argument('-m', '--match_files', type=str, nargs='+', help='List of pairs : [(shared, src, tgt), Path of the file]')
    parser.add_argument('-b', '--bpe_files', type=str, nargs='+', help='List of pairs : [(shared, src, tgt), Path of the file]')    
    parser.add_argument('-v', '--word_files', type=str, nargs='+', help='List of pairs : [(shared, src, tgt), Path of the file]')
    parser.add_argument('-w', '--work_dir', type=Path, help='Working Experiment Directory')
    
    parser.add_argument('-a', '--max_sgrams', type=int, default=0, help='Max skip grams to be considered')
    parser.add_argument('-f', '--min_freq', type=int, default=100, help='Min frequency of the skip grams to be considered')
    parser.add_argument('-c', '--max_corr', type=float, default=1.0, help='Max correlation alloed for skip token')
    parser.add_argument('-cw', '--min_center_words', type=int, default=0, help='Atleast "cw" different words must appear in the skip space')
    parser.add_argument('-x', '--sorter', type=str, choices=['freq', 'pmi', 'ngdf', 'ngd'], 
                        default='freq', help='NGram Sorter Function to be used.')
    # parser.add_argument('-x', '--save_file', type=str)
    return parser.parse_args()

def save_meta(args, work_dir):
    mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')

    mw.close(add_dashline=True)

def args_validation(args):
    for data_file in args.data_files:
        assert data_file.exists()
    assert len(args.match_files) % 2 == 0
    assert len(args.bpe_files) % 2 == 0

# ----------------------------------------------------------------------------

def make_files_dict(file_str:List[str]):
    files = dict()
    for i in range(0, len(file_str), 2):
        files[file_str[i]] = Path(file_str[i+1])
    return files

def validate_vocab_files(vocab_files:Dict[str,Union[Path,str]], shared):
    if shared:
        assert 'shared' in vocab_files.keys()
        assert vocab_files['shared'].exists()
    else:
        assert 'src' in vocab_files.keys()
        assert 'tgt' in vocab_files.keys()
        assert vocab_files['src'].exists()
        assert vocab_files['tgt'].exists()

def make_sgrams(data_files:List[Filepath], bpe_files:Dict[str,Path], match_files:Dict[str,Path],  
                word_files:Dict[str,Path], work_dir:Filepath, shared:bool=False, 
                max_corr:float=1.0, min_center_words:int=0, min_freq:int=0, 
                max_sgrams:int=0, sorter:str='freq'):
    ds = Dataset(['src', 'tgt'])
    for data_file in data_files:
        ds.add(read_parallel(data_file))
    if shared:
        shared_vcb, _ = SkipGrams.get_skipgrams(ds.lists.values(), 
                                    match_files['shared'], bpe_files['shared'],
                                    word_files['shared'], min_freq=min_freq, 
                                    max_corr=max_corr, max_sgrams=max_sgrams, 
                                    min_center_words=min_center_words, sorter=sorter)
        shared_vcb._write_out(work_dir / Path(f'sgrams.{sorter}.corr{max_corr}.{bpe_files["shared"].name}'))
    else:
        src_vcb, _ = SkipGrams.get_skipgrams([ds.lists['src']], 
                                match_files['src'], bpe_files['src'],
                                word_files['src'], min_freq=min_freq, 
                                max_corr=max_corr, max_sgrams=max_sgrams,
                                min_center_words=min_center_words, sorter=sorter)
        src_vcb._write_out(work_dir / Path(f'sgrams.{sorter}.corr{max_corr}.cw{min_center_words}.{bpe_files["src"].name}'))
        
        tgt_vcb, _ = SkipGrams.get_skipgrams([ds.lists['tgt']], 
                                match_files['tgt'], bpe_files['tgt'],
                                word_files['tgt'], min_freq=min_freq, 
                                max_corr=max_corr, max_sgrams=max_sgrams, 
                                min_center_words=min_center_words, sorter=sorter)
        tgt_vcb._write_out(work_dir / Path(f'sgrams.{sorter}.corr{max_corr}.cw{min_center_words}.{bpe_files["tgt"].name}'))

def main():
    log('Starting script : make_sgrams')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    bpe_files = make_files_dict(args.bpe_files)
    match_files = make_files_dict(args.match_files)
    word_files = make_files_dict(args.word_files)
    log('> Validating match, bpe and word files', 1)
    validate_vocab_files(bpe_files, args.shared)
    validate_vocab_files(match_files, args.shared)
    validate_vocab_files(word_files, args.shared)

    wdir = make_dir(args.work_dir)
    ndir = make_dir(wdir / Path('sgrams/'))

    log(f'Preparing skip grams ...',1)
    make_sgrams(args.data_files, bpe_files, match_files, word_files, ndir,  
                shared=args.shared, min_freq=args.min_freq, max_corr=args.max_corr,
                max_sgrams=args.max_sgrams, min_center_words=args.min_center_words, 
                sorter=args.sorter)
    log('Process completed')

if __name__ == "__main__":
    main()
