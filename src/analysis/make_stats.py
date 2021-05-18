import argparse
from pathlib import Path
from typing import Dict, List, Union
from collections import Counter

from nlcodec import load_scheme, Reseved

from lib.misc import FileReader, FileWriter, log, make_dir
from lib.misc import ScriptFuncs as Sf
from lib.vocabs import Vocabs
from lib.stat import StatLib
from lib.stat import BaseFuncs as Bf
from lib.grams import GramsBase as Gb
from lib.grams import GramsNormalizer as Gn
from lib.dataset import read_parallel, Dataset

def parse_args():
    parser = argparse.ArgumentParser(prog='analysis.make_stats')
    parser.add_argument('-d', '--data_file', type=Path)
    parser.add_argument('-b', '--bpe_files', type=str, nargs='+')
    parser.add_argument('-w', '--work_dir', type=Path)
    parser.add_argument('--shared', type=bool, default=False)
    return parser.parse_args()

def args_validation(args):
    assert args.data_file.exists()
    # assert args.save_file.parent.exists()

def load_vocabs(vocab_files):
    vocabs = dict()
    for k,v in vocab_files.items():
        vcb = Vocabs()
        vcb._read_in(v)
        vocabs[k] = vcb
    return vocabs

def make_stats(corps, bpe):
    bpe_stats = { t.idx : {'freq':0, 'nsents':0} for t in bpe }
    for corp in corps:
        for sent in corp:
            toks = Counter(sent)
            for x in toks.keys():
                bpe_stats[x]['freq'] += toks[x]
                bpe_stats[x]['nsents'] += 1
    return bpe_stats

def save_stats(stats, nsents, bpe, save_file):
    with open(save_file, 'w') as fw:
        ntokens = sum([stats[x]['freq'] for x in stats.keys()])
        for token in bpe:
            ix = token.idx
            parts = [token.name, str(stats[ix]['freq']), str((stats[ix]['nsents'] * 100) / nsents)]
            fw.write('\t'.join(parts))
            fw.write('\n')

def prepare_stats(data_file, bpe_files, work_dir, shared):
    ds = Dataset(['src', 'tgt'])
    ds.add(read_parallel(data_file))
    nsents = len(ds.lists['src'])
    bpes = {k:Vocabs(v) for k,v in bpe_files.items()}

    if shared:
        shared_stats = make_stats(ds.lists.values(), bpes['shared'])
        save_stats(shared_stats, 2*nsents, bpes['shared'], work_dir / Path(f'stats.{bpe_files["shared"].name}'))
    else:
        src_stats = make_stats([ds.lists['src']], bpes['src'])
        save_stats(src_stats, nsents, bpes['src'], work_dir / Path(f'stats.{bpe_files["src"].name}'))
        tgt_stats = make_stats([ds.lists['tgt']], bpes['tgt'])
        save_stats(tgt_stats, nsents, bpes['tgt'], work_dir / Path(f'stats.{bpe_files["tgt"].name}'))

def main():
    log('Starting Script : merge_list')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    wdir = args.work_dir
    make_dir(wdir)
    log(f'> Work Dir : {wdir}', 1)

    log(f'> Validating Vocab Files', 1)
    bpe_files = Sf.make_files_dict(args.bpe_files)
    Sf.validate_vocab_files(bpe_files, args.shared)

    prepare_stats(args.data_file, bpe_files, wdir, args.shared)

if __name__ == "__main__":
    main()