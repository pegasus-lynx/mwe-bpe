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
    # parser.add_argument('-s', '--src_path', type=Path, help='Path of the src file')
    # parser.add_argument('-t', '--tgt_path', type=Path, help='Path of the tgt file')
    parser.add_argument('-w', '--work_dir', type=Path, help='Path to the working directory')
    parser.add_argument('--shared_vocab', type=Path)
    parser.add_argument('--src_vocab', type=Path)
    parser.add_argument('--tgt_vocab', type=Path)
    # parser.add_argument('--src_len', type=int, default=0)
    # parser.add_argument('--tgt_len', type=int, default=0)
    # parser.add_argument('--truncate', type=bool, default=False)
    # parser.add_argument('-m', '--save_mode', type=str, default='101', help='Binary string of \
                        # length 3, on bit saving file as .db, .tsv, .tsv.gz respectively')
    # parser.add_argument('-x', '--save_file', type=str, help='Name of the file to store the processed data files')
    parser.add_argument('-d', '--data_file', type=Path)
    return parser.parse_args()

def args_validation(args):
    # assert args.src_path.exists()
    # assert args.tgt_path.exists()
    # assert args.save_file is not None
    # assert len(args.save_mode) == 3
    if args.shared_vocab is not None:
        assert args.shared_vocab.exists()
    else:
        assert args.src_vocab.exists()
        assert args.tgt_vocab.exists()

# ----------------------------------------------------------------------------

def decode_data_file(datafile, vocab_files:Dict[str,Path], work_dir:Filepath):
    codecs = {key: load_scheme(value) for (key, value) in vocab_files.items() if value is not None}
    src_codec = codecs['shared'] if 'shared' in codecs.keys() else codecs['src']
    tgt_codec = codecs['shared'] if 'shared' in codecs.keys() else codecs['tgt']
    
    src_seqs = []
    tgt_seqs = []
    with open(datafile, 'r') as fr:
        for line in fr:
            src, tgt = line.strip().split('\t')
            src, tgt = src.split(), tgt.split()
            
            src_str = _decode(src_codec, src)
            tgt_str = _decode(tgt_codec, tgt)

            src_seqs.append(src_str)
            tgt_seqs.append(tgt_str)

    save_dec_file(src_seqs, work_dir / Path(f'src.dec.{datafile.name}'))
    save_dec_file(tgt_seqs, work_dir / Path(f'tgt.dec.{datafile.name}'))

def save_dec_file(seqs, filepath):
    with open(filepath, 'w') as fw:
        for seq in seqs:
            fw.write(f'{seq}\n')

def _decode(codec, indices):
    indices = [int(x) for x in indices]
    text = codec.decode(indices)
    return text

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

    # work_file = wdir / Path(f'{str(args.save_file)}.model')

    log('Processing data files', 1)
    decode_data_file(args.data_file, vocab_files, wdir)
    log('Process completed')
    

if __name__ == "__main__":
    main()
