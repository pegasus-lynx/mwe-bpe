from lib.schemes import load_scheme, SkipScheme
import argparse
from nlcodec import Type, codec
from typing import List, Dict
from pathlib import Path

skip_texts = [
    'She ( her ) ( things here )'
]

ngram_texts = [
    'India is located in South Asia .',
    'e. g. is the short hand notation of example',
    'Shri Pranab Mukherjee is the president of India'
]

decode_seqs = [
    ['along▁', 'along▁', 'krä', 'ment▁', 'hol', 'zustellen▁', '„▁▂”▁', 'al', 'ate▁', 
     'al', '„▁▂”▁', 'il▁', 'al', '„▁▂”▁', 'al', 'al', 'al', 'benötigen▁', 'benötigen▁', 
     'benötigen▁', 'benötigen▁', 'Wach', 'benötigen▁', 'benötigen▁', 'Wach', 'Wach', 'Wach', 
     'Wach', 'il▁', 'il▁', 'il▁', 'lives▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 
     'benötigen▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'benötigen▁'], ['benötigen▁', 'benötigen▁', 
     'benötigen▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'hal', 'lives▁', 'lives▁', 'lives▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 
     'benötigen▁', 'Wach', 'Wach', 'Wach', 'Wach', 'Wach', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'solchen▁', 'solchen▁', 'Wach', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'benötigen▁', 'Wach', 'Wach', 'Wach', 'Wach', 'benötigen▁', 'benötigen▁', 'Wach', 'Wach', 'Wach', 'Wach', 'lexi', 'lexi', 'lexi', 'lexi', 'Wach', 'Wach', 'Wach', 'Wach', 'along▁', 'along▁', 'along▁', 'along▁', 'along▁', 'along▁', 'Wach', 'Wach', 'Wach']
]

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

def enc_dec(text:str, codec:List['Type']):
    encoded = codec.encode(text)
    parts   = [codec.table[x].name for x in encoded]
    decoded = codec.decode_str(parts)
    cprint('space', 'dash', text, 'dash', 'space', 
            encoded, 'space', parts, 'space', 'dash', 
            'space', decoded, 'dash', 'space')

def decode_only(seq, codec:List['Type']):
    decoded = codec.decode_str(seq)
    cprint('space', 'dash', seq, 'dash', 'space', decoded, 'dash', 'space')

def parse_args():
    parser = argparse.ArgumentParser(prog="test.enc-dec")
    parser.add_argument('-w', '--work_dir', type=Path)
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--ngram', action='store_true')
    parser.add_argument('--nlcodec_load', action='store_true')
    return parser.parse_args()

def validate(args):
    assert args.work_dir.exists()
    
    data_dir = args.work_dir / 'data'
    assert data_dir.exists()

    src_vocab = data_dir / 'nlcodec.src.model'
    tgt_vocab = data_dir / 'nlcodec.tgt.model'
    assert src_vocab.exists()
    assert tgt_vocab.exists()

def main():
    # args = parse_args()
    # validate(args)

    # vocab_files = {
    #     'src': args.work_dir / Path('data/nlcodec.src.model'),
    #     'tgt': args.work_dir / Path('data/nlcodec.tgt.model')
    # }

    from nlcodec.codec import SkipScheme

    decoded = SkipScheme.decode_str(decode_seqs[0])
    cprint('space', 'dash', decode_seqs[0], 'dash', 'space', decoded, 'dash', 'space')

    # vocab_file = {

    # }

    # keys = ['src', 'tgt']
    # codecs = {}

    # if args.nlcodec_load:
    #     print('Check Nlcodec Load ...')
    #     from nlcodec import load_scheme
    #     for key in keys:
    #         codecs[key] = load_scheme(vocab_files[key])
    # else:
    #     from lib.schemes import load_scheme
    #     scheme = load_scheme('ngram' if args.ngram else 'skipgram')
    #     for key in keys:
    #         table, _ = Type.read_vocab(vocab_files[key])
    #         codecs[key] = scheme(table)

    # if args.skip:
    #     cprint('space', 'Testing Skipgram Texts : ', 'dash')
    #     for text in skip_texts:
    #         enc_dec(text, codecs['tgt'])

    # if args.ngram:
    #     cprint('space', 'Testing Ngram Texts : ', 'dash')
    #     for text in ngram_texts:
    #         enc_dec(text, codecs['tgt'])

if __name__ == "__main__":
    main()