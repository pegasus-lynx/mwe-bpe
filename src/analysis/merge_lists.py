import argparse
from pathlib import Path
from typing import Dict, List, Union

from nlcodec import load_scheme, Reseved

from lib.misc import FileReader, FileWriter, log, make_dir
from lib.misc import ScriptFuncs as Sf
from lib.vocabs import Vocabs
from lib.stat import StatLib
from lib.stat import BaseFuncs as Bf
from lib.grams import GramsBase as Gb
from lib.grams import GramsNormalizer as Gn
from lib.dataset import read_parallel, Dataset


#  Default script functions --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(prog='analysis.prep_list')
    parser.add_argument('-l', '--list_files', type=Path, nargs='+', help='List of data files to be used for comparision.')
    parser.add_argument('-s', '--save_file', type=Path, help='Path to the save merged list')
    parser.add_argument('-m', '--max_types', type=int)
    # parser.add_argument('-b', '--bpe_file', type=Path)
    parser.add_argument('-r', '--reverse', type=bool, default=False)
    return parser.parse_args()

# def save_meta(args, work_dir, trigram_files):
#     mw = FileWriter(work_dir / Path('meta.txt'), mode='a+')
#     mw.heading('trigram_bigram_comparision')
#     mw.section('Work Dir :', [work_dir])
#     mw.section('Data Files :', [args.data_file])
#     mw.section('Trigram Files :', [f'{k} : {v}' for k,v in trigram_files.items()])
#     mw.section('Arguments :', [
#         f'Shared : {args.shared}',
#         f'Norm : {args.norm}'
#     ])
#     mw.close(add_dashline=True)

def args_validation(args):
    for list_file in args.list_files:
        assert list_file.exists()
    assert args.save_file.parent.exists()

# ----------------------------------------------------------------------------

def load_list(list_file):
    vals = list()
    name_set = set()
    with open(list_file, 'r') as fr:
        for line in fr:
            name, val = line.strip().split('\t')
            name_set.add(name)
            vals.append((name, float(val)))
    return vals, name_set

def analyze(sorted_list, names_set, list_files):
    counts = [0] * len(list_files)
    print_list = [100, 200, 500, 1000]
    for pair in sorted_list:
        name, val = pair
        for ix, nset in enumerate(names_set):
            if name in nset:
                counts[ix] += 1
            total_count = sum(counts)
            if total_count in print_list:
                print_list.remove(total_count)
                print(f'Total Count : {total_count}')
                for list_file, count in zip(list_files, counts):
                    print('\t', f'{list_file.name} : {count}')
                print()

def save_list(sorted_list, save_file):
    with open(save_file, 'w') as fw:
        for pair in sorted_list:
            name, val = pair
            fw.write('\t'.join([name, str(val), '\n']))

def merge_lists(list_files, max_types, save_file, reverse=False):
    
    lists, names_set = [], []
    for list_file in list_files:
        vals, names = load_list(list_file)
        lists.append(vals)  
        names_set.append(names)
    
    all_list = []
    for val_list in lists:
        all_list.extend(val_list)

    all_list.sort(key= lambda x: x[1], reverse=reverse)
    
    sorted_list = all_list
    if max_types > 0:
        sorted_list = all_list[:max_types]

    analyze(sorted_list, names_set, list_files)
    save_list(sorted_list, save_file)

def main():
    log('Starting Script : merge_list')
    args = parse_args()
    args_validation(args)
    log('> Loaded Args', 1)

    merge_lists(args.list_files, args.max_types, args.save_file, args.reverse)
    log('Process Completed')
    log('Writing Meta')

if __name__ == "__main__":
    main()