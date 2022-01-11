#from .schemes import load_scheme, SkipScheme
import argparse
from nlcodec import Type, load_scheme
from typing import List, Dict
from pathlib import Path


def cprint(*args):

    def _dash():
        print('-'*80)

    def _space():
        print()

    for arg in args:
        if type(arg) == str:
            if arg == 'dash':
                _dash()
            elif arg == 'space':
                _space()
            else:
                print(arg)
            continue
        print(arg)


def read_validation_file(filepath):
    import gzip
    with gzip.open(filepath, 'rb') as fp:
        content = fp.read().decode('utf-8')
    lines = content.split('\n')
    src_lines, tgt_lines = [], []
    for line in lines:
        src, tgt = line.split('\t')
        src_lines.append( [ int(x) for x in src.split() ] )
        tgt_lines.append( [ int(x) for x in tgt.split() ] )
    return (src_lines, tgt_lines)


def decode_random_lines(corp, vocab):

    print("Printing Random Lines")

    lines = [5, 13, 72, 12, 108, 357, 123, 49]
    for idx in lines:
        line = corp[idx]
        tokens = [vocab[x].name for x in line]
        text = vocab.decode(line)
        print("Tokens : ", tokens)
        print("Text : ", text)
        print()

def decode_scrambled_lines(corp, vocab):
    skip_a, skip_b, skip_c = len(vocab)-1 , len(vocab)-2, len(vocab)-3
    bpe_a, bpe_b, bpe_c = 10, 12, 14
    lines = [
        [skip_a, bpe_a, skip_b, bpe_b, bpe_b, skip_c],
        [skip_a, skip_b, 7, 7, 10, 12, 15, skip_c, skip_b, skip_a]
    ]

    print("Decoding Hard Coded Lines")

    for line in lines:
        tokens = [ vocab[x].name for x in line ]
        text = vocab.decode(line)
        print("Tokens : ", tokens)
        print("Text : ", text)
        print()

def check_decoding_using_validation_set(data_dir):
    shared_vocab_file = data_dir / "nlcodec.shared.model"
    src_lines, tgt_lines = read_validation_file(data_dir / 'valid.tsv.gz')
    
    if shared_vocab_file.exists():
        corp = []
        corp.extend(src_lines)
        corp.extend(tgt_lines)

        shared_vocab = load_scheme(shared_vocab_file)
        decode_random_lines(corp, shared_vocab)
        decode_scrambled_lines(corp, shared_vocab)
    else:
        src_vocab = load_scheme(src_vocab_file)
        tgt_vocab = load_scheme(tgt_vocab_file)

        decode_random_lines(src_lines, src_vocab, 5)
        decode_scrambled_lines(src_lines, src_vocab, 5)

        decode_random_lines(tgt_lines, tgt_vocab, 5)
        decode_scrambled_lines(tgt_lines, tgt_vocab, 5)


def parse_args():
    parser = argparse.ArgumentParser(prog="test.enc-dec")
    parser.add_argument('-w', '--work_dir', type=Path)
    return parser.parse_args()

def validate(args):
    assert args.work_dir.exists()
    
    data_dir = args.work_dir / 'data'
    assert data_dir.exists()

    src_vocab = data_dir / 'nlcodec.src.model'
    tgt_vocab = data_dir / 'nlcodec.tgt.model'
    shared_vocab = data_dir / 'nlcodec.shared.model'

    vocab_exists = src_vocab.exists() and tgt_vocab.exists()
    vocab_exists = vocab_exists or shared_vocab.exists()
    assert vocab_exists

def main():
    args = parse_args()
    data_dir = args.work_dir / Path("data")
    validate(args)

    valid_zip_file = data_dir / 'valid.tsv.gz'
    if valid_zip_file.exists():
        check_decoding_using_validation_set(data_dir)

    decoded = SkipScheme.decode_str(decode_seqs[0])
    cprint('space', 'dash', decode_seqs[0], 'dash', 'space', decoded, 'dash', 'space')


if __name__ == "__main__":
    main()
