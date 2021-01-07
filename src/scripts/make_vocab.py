import argparse
from pathlib import Path
from typing import Union, List, Dict
from nlcodec import learn_vocab, term_freq

from vocabs import Vocabs
from misc import load_conf, FileReader, FileWriter

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.vocab', description='Prepare data for NMT experiment.')
    parser.add_argument('-f', '--files', nargs='+', type=Path, help='List of files to be processed')
    parser.add_argument('-w', '--workdir', type=Path)
    parser.add_argument('-v', '--vocab_size', type=int, default=30000)
    parser.add_argument('-t', '--type', type=str, choices=['char', 'word', 'bpe'], default='bpe')
    return parser.parse_args()

def args_validation(args):
    assert args.workdir is not None
    for filepath in args.files:
        assert filepath.exists()

def save_meta(args, metapath:Union[Path, str]):
    pass

def make_vocabs(corpus:List[Union[str, Path]], model_path, vocab_size:int=30000, level:str='bpe'):
    fr = FileReader(corpus, tokenized=False)
    corp = fr.unique()
    learn_vocab(inp=corp, level=level, model=model_path, vocab_size=vocab_size)

def main():
    print('Starting Script ...')
    args = parse_args()
    args_validation(args)
    print('\t > Loaded Args')

    if not args.workdir.exists():
        args.workdir.mkdir()
        print('\t > Making working dir')
    
    vdir = workdir / Path('_vocabs/')
    if not vdir.exists():
        vdir.mkdir()
        print('\t > Making vocab dir ...')

    print('\t > Making and saving vocabs')
    make_vocabs(args.files, vdir/Path(f'vocab.{args.type}.{args.vocab_size}.model'), vocab_size=args.vocab_size, level=args.type)
    print('Process Completed')

if __name__ == "__main__":
    main()