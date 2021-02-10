import argparse
from pathlib import Path
from indicnlp.tokenize.indic_detokenize import trivial_detokenize

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.detok', description='Detokenizes hin tok files')
    parser.add_argument('-f', '--file', type=Path, help='File for detokenization')
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.file.exists()
    fr = open(args.file, 'r')
    write_file = args.file.with_suffix('.raw.txt')
    fw = open(write_file, 'w')
    for line in fr:
        line = line.strip()
        fw.write(f'{trivial_detokenize(line)}\n')

if __name__ == '__main__':
    main()