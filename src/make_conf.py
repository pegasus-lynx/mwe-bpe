import argparse
import os
import copy
from json import load

import re
from pathlib import Path 
import collections as coll
from typing import List, Dict, Tuple

from nlcodec import Type
from rtg.data.dataset import TSVData, SqliteFile

from lib.misc import read_conf, make_dir, make_file, write_conf

def convert_type(value):
    if value == '':
        return None
    if value in ['True', 'true']:
        return True
    if value in ['False', 'false']:
        return False
    if re.fullmatch("-?\d+", value):
        return int(value)
    if re.fullmatch("-?\d+\.\d*", value):
        return float(value)
    return value


# create a args2dict_action class
class args2dict_action(argparse.Action):
    # Constructor calling
    def __call__( self , parser, namespace,
                 values, option_string = None):
        setattr(namespace, self.dest, dict())
          
        for value in values:
            # split it into key and value
            key, value = value.split('=')
            if ',' in value:
                value = [convert_type(x) for x in value.split(',')]
            else:
                value = convert_type(value)

            # assign into dictionary
            getattr(namespace, self.dest)[key] = value

def remove_none_values(configs, trimmed_configs=None):
    if trimmed_configs is None:
        trimmed_configs = copy.deepcopy(configs)
    for key in configs.keys():
        value = configs[key]
        if type(value) == dict and len(value) != 0: 
            trimmed_configs[key] = remove_none_values(value)
        
        value = trimmed_configs[key]
        if type(value) == dict or type(value) == list:
            if len(value)==0:
                del trimmed_configs[key]
        else:
            if value is None:
                del trimmed_configs[key]
    return trimmed_configs

def dfs_update(configs, kwargs, keys):
    # print(configs.keys())
    for key in configs.keys():
        if type(configs[key]) == dict:
            configs[key] = dfs_update(configs[key], kwargs, keys)
        elif key in keys:
            if type(configs[key]) == dict:
                configs[key] = dfs_update(configs[keys], kwargs, keys)
            else:
                configs[key] = kwargs[key]
    return configs

def update_configs(configs, kwargs):
    keys = set(kwargs.keys())
    updated_configs = dfs_update(configs, kwargs, keys)

    # If we use a general base for every type of experiment, we will need trimming
    # If we get a custom base for different language pairs depending on shared vocab and other 
    # parameters,  trimming part can be taken care of before hand.
    # Trimmings function can have different bugs.
    trimmed_configs = remove_none_values(updated_configs)
    return trimmed_configs

def parse_args():
    parser = argparse.ArgumentParser(prog='make_conf', description="Prepares custom config files from base config files")

    parser.add_argument('-c', '--base_config_file', type=Path, 
                            help='Base config file for preparation of new config files.')
    parser.add_argument('-w', '--work_dir', type=Path, 
                            help='Path to the working directory for storing the prepared config files.')
    parser.add_argument('-n', '--output_filename', type=str)

    parser.add_argument('--kwargs', nargs='*', action=args2dict_action)

    return parser.parse_args()
    
def main():

    print("Parsing args ...")
    args = parse_args()

    print("Parameters to be updated ...")
    print(args.kwargs)

    # Read Base Confs
    print("Reading configs ...")
    base_configs = read_conf(args.base_config_file, 'yaml')
    
    # Update Confs
    print("Updating configs ...")
    updated_configs = update_configs(base_configs, args.kwargs)

    # Save Confs
    print("Making Directory ...")
    make_dir(args.work_dir)

    output_file = args.work_dir / Path(args.output_filename)

    print("Writing configs to : {}".format(str(output_file)))
    write_conf(updated_configs, output_file)

if __name__ == "__main__":
    main()