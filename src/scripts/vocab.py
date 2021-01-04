import argparse
from pathlib import Path
from typing import Union, List, Dict
from nlcodec import learn_vocab, term_freq

from vocabs import Vocabs
from misc import load_conf, FileReader, MetaWriter

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.prep', description='Prepare data for NMT experiment.')
    parser.add_argument('-f', '--files', nargs='+', type=Path, help='List of files to be processed')
    parser.add_argument('-w', '--workdir', type=Path)
    parser.add_argument('-v', '--vocab_size', type=int, default=30000)
    # parser.add_argument('-c', '--configs', type=Path, default='conf.json')
    return parser.parse_args()

def args_validation(args):
    # assert args.configs.exists()
    assert args.workdir is not None
    for filepath in args.files:
        assert filepath.exists()

def save_meta(args, metapath:Union[Path, str]):
    pass

def make_vocabs(corpus:List[Union[str, Path]], model_path, vocab_size:int):
    fr = FileReader(corpus, tokenized=False)
    corp = fr.unique()
    learn_vocab(inp=corp, level='bpe', model=model_path, vocab_size=vocab_size)

def main():
    args = parse_args()
    args_validation(args)

    if not args.workdir.exists():
        args.workdir.mkdir()
    make_vocabs(args.files, args.workdir/Path(f'vocab.{args.vocab_size}.model'), args.vocab_size)

if __name__ == "__main__":
    main()