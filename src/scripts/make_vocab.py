import argparse
from pathlib import Path
from typing import Dict, List, Union

from lib.misc import FileReader, FileWriter, log
from lib.vocabs import Vocabs

#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='scripts.make_vocab', description='Vocab operations for the preparation of data.')
    parser.add_argument('-f', '--files', nargs='+', type=Path, help='List of files to be processed')
    parser.add_argument('-w', '--work_dir', type=Path, help='Path to the working directory')
    parser.add_argument('-v', '--vocab_size', type=int, default=8000)
    parser.add_argument('-t', '--type', type=str, choices=['char', 'word', 'bpe'], default='bpe')
    parser.add_argument('-x', '--save_file', type=str)
    return parser.parse_args()

def save_meta(args, work_dir, vocab_file):
    mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')
    mw.heading('make_vocab')
    mw.section('Work File :', [vocab_file.name])
    mw.section('Arguments :', [
        f'Vocab Size : {args.vocab_size}',
        f'Vocab Type : {args.type}',
    ])
    mw.section('Files :', args.files)
    mw.close()

def args_validation(args):
    assert args.work_dir is not None
    for filepath in args.files:
        assert filepath.exists()

# ----------------------------------------------------------------------------

def main():
    log('Starting Script : make_vocab')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    wdir = args.work_dir
    vsize = args.vocab_size
    if not wdir.exists():
        wdir.mkdir()
        print('> Making working dir', 1)

    log('> Making and saving vocabs', 1)
    work_file = wdir/Path(f'{args.type}.{vsize//1000}k.model')
    if args.save_file is not None:
        work_file = wdir/Path(args.save_file)
    Vocabs.make(args.files, work_file, 
                vocab_size=vsize, level=args.type)
    log('Process Completed')
    log('Writing Meta')
    save_meta(args, wdir, work_file)
    

if __name__ == "__main__":
    main()
