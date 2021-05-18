""" Utils for processing datasets """

__author__ = ""

from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path
from typing import Dict, Union

from tqdm import tqdm

from .dataset import Dataset
from .tokenizer import CharTokens, Tokenizer


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
    def filter(row, langs=['en', 'hi']):
        filter_in = True
        for text, lang in zip(row, langs):
            if lang == 'en':
                filter_in = filter_in and CharTokens.eng(text)
            elif lang == 'hi':
                filter_in = filter_in and CharTokens.hin(text)
        return filter_in
