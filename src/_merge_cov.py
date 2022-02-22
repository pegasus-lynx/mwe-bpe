import argparse
from pathlib import Path


def load_file(filep):
    fdict = dict()
    with open(filep, 'r') as fp:
        for line in fp:
            parts = line.split('\t')
            name = parts[0]
            scov = parts[1]
            freq = parts[2]
            fdict[name] = [scov, freq]
    return fdict


def merge_dicts(dict1, dict2):
    mdict = dict()
    for key in dict1.keys():
        if key not in mdict.keys():
            mdict[key] = []

        mdict[key].extend(dict1[key])
        if key not in dict2.keys():
            mdict[key].extend(["0", "0"])
        else:
            mdict[key].extend(dict2[key])

    for key in dict2.keys():
        if key not in mdict.keys():
            mdict[key] = ["0", "0"]
            mdict[key].extend(dict2[key])

    return mdict


def save_dict(merged, filep):
    with open(filep, 'w') as fw:
        for key in merged.keys():
            fw.write("\t".join([key, *merged[key]]))
            fw.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(prog="Merge Lists", description="Merge coverage lists")
    parser.add_argument('-f1', '--file1', type=Path)
    parser.add_argument('-f2', '--file2', type=Path)
    parser.add_argument('-sf', '--save_file',  type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    file1_dict = load_file(args.file1)
    file2_dict = load_file(args.file2)
    merged_dict = merge_dicts(file1_dict, file2_dict)
    save_dict(merged_dict, args.save_file)


if __name__ == "__main__":
    main()
