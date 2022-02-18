import argparse

from pathlib import Path 
import collections as coll
from typing import Union, List, Dict, Any, Tuple, Iterator
from nlcodec.utils import IO
from nlcodec import Type

from lib.misc import eval_file

def parse_args():
    parser = argparse.ArgumentParser(prog='get_analysis')
    parser.add_argument('-at', '--analysis_type', type=str, choices=['coverage'], default='coverage')
    
    parser.add_argument('-sv', '--src_vocab', type=Path)
    parser.add_argument('-tv', '--tgt_vocab', type=Path)
    
    parser.add_argument('-w', '--work_dir', type=Path)
    
    parser.add_argument('-s', '--src', type=Path)
    parser.add_argument('-t', '--tgt', type=Path)

    parser.add_argument('-d', '--detok', type=Path)
    parser.add_argument('-r', '--ref', type=Path)
    return parser.parse_args()

def main():
    args = parse_args()
    bleu_str, chrf2 = eval_file(args.detok, args.ref)
    print(bleu_str)
    print(chrf2)

if __name__ == "__main__":
    main()
