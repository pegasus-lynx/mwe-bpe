import argparse
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from lib.dataset import SqliteFile, TSVData
from lib.misc import FileWriter, log, Filepath, make_dir
from nlcodec import EncoderScheme, Type, load_scheme
from tqdm import tqdm

#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.prep', description='Prepare data for NMT experiment given the vocabs')
    parser.add_argument('-s', '--src_path', type=Path, help='Path of the src file')
    parser.add_argument('-t', '--tgt_path', type=Path, help='Path of the tgt file')
    parser.add_argument('-w', '--work_dir', type=Path, help='Path to the working directory')
    parser.add_argument('--shared_vocab', type=Path)
    parser.add_argument('--src_vocab', type=Path)
    parser.add_argument('--tgt_vocab', type=Path)
    parser.add_argument('--src_len', type=int, default=0)
    parser.add_argument('--tgt_len', type=int, default=0)
    parser.add_argument('--truncate', type=bool, default=False)
    parser.add_argument('-m', '--save_mode', type=str, default='101', help='Binary string of \
                        length 3, on bit saving file as .db, .tsv, .tsv.gz respectively')
    parser.add_argument('-x', '--save_file', type=str, help='Name of the file to store the processed data files')
    return parser.parse_args()

def save_meta(args, work_dir, work_file):
    mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')
    mw.heading('prep_data')
    mw.section('Work File :', [work_file.name])
    mw.section('Files :', [
        f'Soruce Data File  : {args.src_path}',
        f'Target Data File  : {args.tgt_path}',
        f'Source Vocab File : {args.src_vocab}',
        f'Target Vocab File : {args.tgt_vocab}',
        f'Shared Vocab File : {args.shared_vocab}'
    ])

    mw.section('Arguments :', [
        f'Source Len : {args.src_len}',
        f'Target Len : {args.tgt_len}',
        f'Truncate : {args.truncate}',
        f'Save Mode : {args.save_mode} [ {" ".join([x for x, b in zip(suffixes,list(args.save_mode)) if b=="1"])} ]'
    ])
    mw.close(add_dashline=True)

def args_validation(args):
    assert args.src_path.exists()
    assert args.tgt_path.exists()
    assert args.save_file is not None
    assert len(args.save_mode) == 3
    if args.shared_vocab is not None:
        assert args.shared_vocab.exists()
    else:
        assert args.src_vocab.exists()
        assert args.tgt_vocab.exists()

# ----------------------------------------------------------------------------

suffixes = ['.db', '.tsv', '.tsv.gz']

def pre_process(src_path:Union[Path, str], tgt_path:Union[Path, str], vocab_files:Dict[str,Path],
                truncate:bool, src_len:int, tgt_len:int, work_file:Filepath, save_mode:str='101'):
    codecs = {key: load_scheme(value) for (key, value) in vocab_files.items() if value is not None}
    save_flags = [ False if x=='0' else True for x in list(save_mode)]

    for save_flag, suffix in zip(save_flags, suffixes):
        if save_flag:
            log('Reading parallel recs', 2)
            recs = read_parallel_recs(codecs, src_path, tgt_path, truncate, 
                                src_len, tgt_len, _encode, _encode)
            save_file = work_file.with_suffix(suffix)
            log(f'Writing : {save_file.name}', 2)
            write_parallel_recs(recs, save_file)

def _encode(codec, text:str):
    ids = codec.encode(text)
    return np.array(ids, dtype=np.int32)

def read_parallel_recs(codecs: Dict[str,Path], src_path:Path, tgt_path:Path, truncate:bool, 
                        src_len:int, tgt_len:int, src_tokenizer, tgt_tokenizer):    
    src_codec = codecs['shared' if 'shared' in codecs.keys() else 'src']
    tgt_codec = codecs['shared' if 'shared' in codecs.keys() else 'tgt']
    return TSVData.read_raw_parallel_recs(src_path, tgt_path, truncate, src_len, tgt_len, 
                                lambda x: src_tokenizer(src_codec, x), lambda y: tgt_tokenizer(tgt_codec, y))

def write_parallel_recs(records, path:Union[str, Path]):
    if path.name.endswith('.db'):
        SqliteFile.write(path, records)
    else:
        TSVData.write_parallel_recs(records, path)

def main():
    log('Starting Script : prep_data')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    wdir = make_dir(args.work_dir)

    vocab_files = dict()
    if args.shared_vocab is not None:
        vocab_files['shared'] = args.shared_vocab
    else:
        vocab_files['src'] = args.src_vocab
        vocab_files['tgt'] = args.tgt_vocab

    work_file = wdir / Path(f'{str(args.save_file)}.model')

    log('Processing data files', 1)
    pre_process(args.src_path, args.tgt_path, vocab_files, args.truncate, 
                args.src_len, args.tgt_len, work_file, args.save_mode)
    log('Process completed')
    save_meta(args, wdir, work_file)
    log('Writing meta')
    
if __name__ == "__main__":
    main()
