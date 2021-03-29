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
    detok_file = tsv_file.parent / Path('detoks') / Path(tsv_file.with_suffix('.detok').name)
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

    score_file = args.tsv_file.parent / Path('scores.txt')
    detok_file = make_detok(args.tsv_file)
    bleu_str = eval_file(detok_file, args.ref_file)
    parts = bleu_str.split(':')

    with open(score_file, 'a') as fr:
        fr.write(f'{"-"*60}\n')
        fr.write(f'{args.tsv_file.name}\n')
        fr.write(f'{parts[0]}\n')
        fr.write(f'{parts[1]}\n')
        fr.write('\n')

if __name__ == "__main__":
    main()
