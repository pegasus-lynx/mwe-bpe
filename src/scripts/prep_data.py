import argparse
from pathlib import Path
from itertools import zip_longest
from typing import Union, List, Dict
from nlcodec import load_scheme, EncoderScheme, Type
import numpy as np
from lib.misc import FileWriter
from tqdm import tqdm

from lib.dataset import SqliteFile, TSVData

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
    return parser.parse_args()

def args_validation(args):
    assert args.src_path.exists()
    assert args.tgt_path.exists()
    
    if args.shared_vocab is not None:
        assert args.shared_vocab.exists()
    else:
        # assert args.get('src_vocab') is not None 
        # assert args.get('tgt_vocab') is not None 
        assert args.src_vocab.exists()
        assert args.tgt_vocab.exists()

    # assert args.src_len != 0
    # assert args.tgt_len != 0

def pre_process(src_path:Union[Path, str], tgt_path:Union[Path, str], vocab_files:Dict[str,Path],
                truncate:bool, src_len:int, tgt_len:int, work_file:Path):
    codecs = {key: load_scheme(value) for (key, value) in vocab_files.items() if value is not None}

    print(f'> Writing {work_file.name} ...')
    recs = read_parallel_recs(codecs, src_path, tgt_path, truncate, 
                                src_len, tgt_len, encode_as_ids, encode_as_ids)
    write_parallel_recs(recs, work_file)


    if work_file.name.endswith('.db'):
        work_file = work_file.with_suffix('.tsv')
    elif work_file.name.endswith('.tsv'):
        work_file = work_file.with_suffix('.tsv.gz')

    print(f'> Writing {work_file.name} ...')
    recs = read_parallel_recs(codecs, src_path, tgt_path, truncate, 
                                src_len, tgt_len, encode_as_ids, encode_as_ids)
    write_parallel_recs(recs, work_file)

    if work_file.name.endswith('.gz'):
        return

    work_file = work_file.with_suffix('.tsv.gz')
    print(f'> Writing {work_file.name} ...')
    recs = read_parallel_recs(codecs, src_path, tgt_path, truncate, 
                                src_len, tgt_len, encode_as_ids, encode_as_ids)
    write_parallel_recs(recs, work_file)

def encode_as_ids(codec, text:str):
    ids = codec.encode(text)
    return np.array(ids, dtype=np.int32)

def read_parallel_lines(src_path:Union[Path, str], tgt_path:Union[Path, str]):
    src_lines = open(src_path, 'r')
    tgt_lines = open(tgt_path, 'r')
    recs = ((src.strip(), tgt.strip()) for src, tgt in tqdm(zip_longest(src_lines, tgt_lines)))
    recs = ((src, tgt) for src, tgt in tqdm(recs) if src and tgt)
    yield from recs

def read_parallel_recs(codecs: Dict[str,Path], src_path:Path, tgt_path:Path, truncate:bool, 
                        src_len:int, tgt_len:int, src_tokenizer, tgt_tokenizer):
    # recs = read_parallel_lines(src_path, tgt_path)
    
    if 'shared' in codecs.keys():
        src_codec = codecs['shared']
        tgt_codec = codecs['shared']
    else:
        src_codec = codecs['src']
        tgt_codec = codecs['tgt']

    return TSVData.read_raw_parallel_recs(src_path, tgt_path, truncate, src_len, tgt_len, 
                                lambda x: src_tokenizer(src_codec, x), lambda y: tgt_tokenizer(tgt_codec, y))

    # recs = ((src_tokenizer(src_codec, x), tgt_tokenizer(tgt_codec, y)) for x, y in tqdm(recs))
    # if truncate:
    #     recs = ((src[:src_len], tgt[:tgt_len]) for src, tgt in recs)
    # else:  # Filter out longer sentences
    #     recs = ((src, tgt) for src, tgt in tqdm(recs) if len(src) <= src_len and len(tgt) <= tgt_len)
    # return recs    

def write_lines(lines, path):
    fw = FileWriter(path)
    fw.textlines(lines)
    fw.close()

def write_parallel_recs(records, path:Union[str, Path]):
    if path.name.endswith('.db'):
        SqliteFile.write(path, records)
    else:
        TSVData.write_parallel_recs(records, path)
    # seqs = ((' '.join(map(str, x)), ' '.join(map(str, y))) for x, y in records)
    # lines = (f'{x}\t{y}' for x,y in seqs)
    # write_lines(lines, path)
    # path = Path(str(path).replace('.tsv', '.tsv.gz'))
    # write_lines(lines, path)

def main():
    print('Running Scripts ...')
    args = parse_args()
    args_validation(args)
    print('\t > Loaded Args')

    if not args.work_dir.exists():
        args.work_dir.mkdir()
        print('\t > Making work dir ...')

    vocab_files = dict()
    if args.shared_vocab is not None:
        vocab_files['shared'] = args.shared_vocab
    else:
        vocab_files['src'] = args.src_vocab
        vocab_files['tgt'] = args.tgt_vocab

    pre_process(args.src_path, args.tgt_path, vocab_files, args.truncate, 
                args.src_len, args.tgt_len, args.work_dir / Path('train.db'))
    
if __name__ == "__main__":
    main()