import argparse
from pathlib import Path
from typing import Dict, List, Union
from collections import Counter

from nlcodec import Reseved

def parse_args():
    parser = argparse.ArgumentParser(prog='analysis.make_stats')
    parser.add_argument('-f', '--file', type=Path)
    return parser.parse_args()

def args_validation(args):
    assert args.file.exists()

def remake(lfile):
    sfile = lfile.parent / Path(f'mwe.{lfile.name}')
    with open(sfile, 'w') as fw:
        with open(lfile, 'r') as fr:
            for ix, line in enumerate(fr):
                line = line.strip()
                cnts = Counter(line)
                if cnts[Reseved.SPACE_TOK[0]] > 1:
                    fw.write('\t'.join([str(ix), line, '\n']))

def main():
    args = parse_args()
    remake(args.file)

if __name__ == "__main__":
    main()