import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(prog='misc.merge_detoks')
    parser.add_argument('-d', '--data_files', type=Path, nargs='+')
    parser.add_argument('-s', '--save_file', type=Path)
    return parser.parse_args()

def args_validation(args):
    for f in args.data_files:
        assert f.exists()
    assert args.save_file.parent.exists()

def merge(data_files, save_file):

    readers = [ open(f, 'r') for f in data_files]

    print('Reading Files')
    liness = []
    for r in readers:
        lines = []
        for line in r:
            lines.append(line.strip())
        liness.append(lines)

    nfiles = len(liness)
    lens = [len(x) for x in liness]
    print(lens)
    assert max(lens) == min(lens)
    length = max(lens)
    print('Starting Merge')
    with open(save_file, 'w') as fw:
        for s in range(length):
            for x in range(nfiles):
                fw.write(liness[x][s])
                fw.write('\n')
            fw.write('\n')
    for r in readers:
        r.close()


def main():
    args = parse_args()
    args_validation(args)
    merge(args.data_files, args.save_file)

if __name__ == "__main__":
    main()