import json
import argparse
from pathlib import Path
from typing import Union, List, Dict
from nlcodec import load_scheme, Reseved, Type
from collections import Counter
from lib.misc import FileWriter, get_now


def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.full_token', description='Find all tokens from the bpe vocab that are complete words.')
    parser.add_argument('-b', '--bpe_vocab', type=Path, help='Bpe Vocab file from where to find the full tokens')
    parser.add_argument('-v', '--word_vocab', type=Path, help='Word vocab file to match the full tokens')
    parser.add_argument('-w', '--work_dir', type=Path, help='Working Experiment Directory')
    return parser.parse_args()

def args_validation(args):
    assert args.word_vocab.exists()
    assert args.bpe_vocab.exists()

def write_out(bpe_words:List['Type'], work_file:Path):
    fw = FileWriter(work_file)
    levels = Counter(v.level for v in bpe_words)
    max_level = max(levels.keys())
    meta = dict(total=len(bpe_words), levels=levels, max_level=max_level, create=get_now())
    meta = json.dumps(meta)
    fw.textline(f'#{meta}')
    for i, item in enumerate(bpe_words):
        fw.textline(item.format())
    fw.close()
    print(f'Wrote {len(bpe_words)} to the file.')

def bpe_words(bpe_vocab:Path, word_vocab:Path, work_file:Path):
    bpe_toks = load_scheme(bpe_vocab)
    word_toks = load_scheme(word_vocab)

    reserved_toks = [x[0] for x in Reseved.ALL]

    words_list = set()
    for token in word_toks.table:
        if token.name in reserved_toks:
            continue
        words_list.add(token.name)

    bpe_words = []
    for token in bpe_toks.table:
        if token.name in words_list:
            bpe_words.append(token)
            continue
        if token.name.endswith(Reseved.SPACE_TOK[0]):
            name = token.name[:-1]
            if name in words_list:
                bpe_words.append(token)

    write_out(bpe_words, work_file)

def main():
    print('Starting script ...')
    args = parse_args()
    args_validation(args)
    print('\t > Loaded Args')

    if not args.work_dir.exists():
        args.work_dir.mkdir()
        print('\t > Making work dir ...')

    vdir = args.work_dir / Path('_vocabs/')
    if not vdir.exists():
        vdir.mkdir()
        print('\t > Making vocab dir ...')

    bpe_words(args.bpe_vocab, args.word_vocab, vdir / Path('bpe.word.model'))

if __name__ == "__main__":
    main()