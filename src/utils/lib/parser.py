""" Utils for processing datasets """

__author__ = "Dipesh Kumar"

import xml.etree.ElementTree as ET 
from .dataset import Dataset, Functionals
from .tokenizer import Tokenizer
from .lang import  CharTokens

from tqdm import tqdm
from typing import Union, Dict
from pathlib import Path
from abc import ABC, abstractmethod, abstractstaticmethod
import re

class Parser(ABC):
    @abstractstaticmethod
    def processor():
        """
            Processes the raw dataset in the form from where it can be directly
            read and be used
        """
        pass

    @abstractstaticmethod
    def writer():
        """
            Helps in writing the processed dataset into seperate files.
        """
        pass

    @abstractstaticmethod
    def reader():
        """
            Reads the processed dataset.
        """
        pass

    @abstractstaticmethod
    def filter():
        """
            Filters out the text that is not to be included
            True : Include the text in corpus
            False: Don't include the text in corpus
        """
        pass

class Parallel(Parser):
    @staticmethod
    def processor(filepaths:dict, keys:list=["src", "tgt"], langs:dict={"src":"en", "tgt":"hi"}, filter=True):
        ds = Dataset(keys)
        tok = Tokenizer()
        src_fs = open(filepaths["src"], "r+")
        tgt_fs = open(filepaths["tgt"], "r+")
        for src, tgt in tqdm(zip(src_fs, tgt_fs)):
            if filter and not Parallel.filter([src, tgt], langs=langs):
                continue
            src_tokens = tok.tokenize(src.strip(), langs["src"])
            tgt_tokens = tok.tokenize(tgt.strip(), langs["tgt"])
            ds.append([src_tokens, tgt_tokens], keys)
        return ds

    @staticmethod
    def writer(dataset:Dataset, files:dict):
        Dataset.save(dataset, files)
        return None

    @staticmethod
    def reader(files:dict, keys:list):
        dataset = Dataset.load(keys, files)
        for ix, row in enumerate(dataset):
            for key, val in zip(keys, row):
                dataset.lists[key][ix] = val.split()
        return dataset

    @staticmethod
    def filter(row, langs={'src':'en', 'tgt':'hi'}):
        src, tgt = row
        if not CharTokens.eng(src):
            return False 
        if not CharTokens.hin(tgt):
            return False       
        return True
