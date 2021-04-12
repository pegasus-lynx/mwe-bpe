from ruamel.yaml import YAML
from pathlib import Path
import argparse

def load_conf(yaml, conf_file):
    with open(conf_file, 'r') as fr:
        config = yaml.load(fr)
    return config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runs_dir', type=Path)
    parser.add_argument('-c', '--conf_file', type=Path)
    return parser.parse_args()

def main():
    args = parse_args()
    yaml = YAML()
    config = load_conf(args.conf_file)


if __name__ == '__main__':;
    main()