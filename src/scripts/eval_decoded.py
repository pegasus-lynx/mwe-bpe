import argparse
import json
from pathlib import Path
from typing import Dict, List, Union
from indicnlp.tokenize.indic_detokenize import trivial_detokenize
from lib.misc import eval_file, log

#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.eval_decoded', description='Evaluate decoded file')
    parser.add_argument('-t', '--tsv_file', type=Path)
    parser.add_argument('-r', '--ref_file', type=Path)
    return parser.parse_args()

def save_meta(args, work_dir):
    mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')
    mw.close(add_dashline=True)

def args_validation(args):
    assert args.tsv_file.exists()
    assert args.ref_file.exists()

# ----------------------------------------------------------------------------

def make_detok(tsv_file:Path):
    detok_file = tsv_file.with_suffix('.detok')
    fr = open(tsv_file, 'r')
    fw = open(detok_file, 'w')
    for line in fr:
        line = line.strip()
        text = line.split('\t')[0]
        text = text.strip()
        fw.write(f'{trivial_detokenize(text)}\n')
    return detok_file

def main():
    log('Starting script : eval_decoded')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    detok_file = make_detok(args.tsv_file)
    eval_file(detok_file, args.ref_file)

if __name__ == "__main__":
    main()
