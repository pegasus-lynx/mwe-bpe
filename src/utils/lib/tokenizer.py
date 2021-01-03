""" Utils for tokenizing the text """

__author__ = "Dipesh Kumar"

import re
import json
from collections import Counter

from sacremoses import MosesTokenizer, MosesDetokenizer

from indicnlp import loader, common
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

config_file =  open("../../config.json", "r")
configs = json.load(config_file)

class Reserved(object):
    PAD_TOK = "<pad>", 0
    OOV_TOK = "<unk>", 1
    BRK_TOK = "<brk>", 2
    SOS_TOK = "<sos>", 3
    EOS_TOK = "<eos>", 4

    TAG_TOKS = [ PAD_TOK[0], OOV_TOK[0], BRK_TOK[0], SOS_TOK[0], EOS_TOK[0] ]
    ENT_TOKS = [ ("<num{}>", 10), ("<alf{}>", 5), ("<date{}>", 5)]

    @classmethod
    def all(cls, vcb_type = 'word'):
        tokens = cls.TAG_TOKS
        if vcb_type == 'char':
            return tokens
        for tok, freq in cls.ENT_TOKS:
            for pos in range(freq):
                tokens.append(tok.format(pos))
        return tokens

    @classmethod
    def validate(cls, word):
        if word in cls.TAG_TOKS:
            return True

        for ent, freq in cls.ENT_TOKS:
            for i in range(freq):
                if word == ent.format(i):
                    return True

        return False

class Tokenizer(object):

    def __init__(self):
        common.set_resources_path(configs["INDIC_NLP_RESOURCES"])
        loader.load()

        # For indic languages
        factory = IndicNormalizerFactory()
        self.normalizer = factory.get_normalizer("hi")

        # Moses Tokenizer
        self.mts = dict()
        self.mts["en"] = MosesTokenizer(lang="en")
        self.mts["ru"] = MosesTokenizer(lang="ru")

    def tokenize(self, text: str, lang: str = "en"):
        text = text.strip()
        if len(text) == 0:
            return []

        if lang in ["en", "ru"]:
            return self.moses(text, lang)
        elif lang == "hi":
            return self.hindi(text)
        else:
            raise(NotImplementedError("The tokenizer works for : English, Hindi and Russian"))

    def moses(self, text:str, lang:str):
        return self.mts[lang].tokenize(text, )
        
    def hindi(self, text:str, normalize=False):
        if normalize:
            text = self.normalizer.normalize(text)
        return indic_tokenize.trivial_tokenize(text)

class EntityMasker(object):

    mask_regs = {
        "num" : "^-?[0-9]+(\\.[0-9]+)?$",
        "date": "^[0-9]{4} ?[-/.] ?[0-9]{1,2} ?[-/.] ?[0-9]{1,2}$",
        "date2": "^[0-9]{1,2} ?[-/.] ?[0-9]{1,2} ?[-/.] ?[0-9]{4}$",
        "alf" : ".*[0-9]+.*[a-zA-Z]*.*",
        "alf2": ".*[a-zA-Z]*.*[0-9]+.*"
    }

    @classmethod
    def match(cls, word):
        """ 
            Matches the word for the regex pattern
            Returns None if not found.
        """
        if Reserved.validate(word):
            return 'res'

        for key in ['num', 'date', 'date2', 'alf', 'alf2']:
            if re.search(cls.mask_regs[key], word) is not None:
                return key
        return None

    @staticmethod
    def get_mask_dict(dataset, keys):
        """
            Returns a mask dict for each sequence.
        """
        pos = []
        for ix, key in enumerate(dataset.lists.keys()):
            if key in keys:
                pos.append(ix)
        mask_dict = []
        rev_mask_dict = []
        for row in dataset:
            counts = { key:0 for key in ["num", "date", "alf"] }
            mask = dict()
            rev_mask = dict()
            for ix in pos:
                for tok in row[ix]:
                    key = EntityMasker.match(tok)
                    if key is not None and tok not in mask.values():
                        if key in ["num"]:
                            mask["<{}{}>".format(key,str(counts[key]))] = tok
                            rev_mask[tok] = "<{}{}>".format(key,str(counts[key]))
                            counts[key] += 1
                        elif key in ["date", "date2"]:
                            mask["<date{}>".format(str(counts["date"]))] = tok 
                            rev_mask[tok] = "<date{}>".format(str(counts["date"]))
                            counts["date"] += 1
                        elif key in ["alf", "alf2"]:
                            mask["<alf{}>".format(str(counts["alf"]))] = tok 
                            rev_mask[tok] = "<alf{}>".format(str(counts["alf"]))
                            counts["alf"] += 1

            mask_dict.append(mask)
            rev_mask_dict.append(rev_mask)

        return mask_dict, rev_mask_dict

    @staticmethod
    def mask_ents(dataset, keys, rev_mask_dict):
        for key, dim in keys:
            for ix, row in enumerate(dataset.lists[key]):
                dataset.lists[key][ix] = EntityMasker.mask(row, dim-1, rev_mask_dict[ix])

        return dataset

    @staticmethod
    def unmask_ents(dataset, keys, mask_dict):
        for key in keys:
            for ix, row in enumerate(dataset.lists[key]):
                dataset.lists[key][ix] = EntityMasker.unmask_seq(row, mask_dict[ix])

        return dataset

    @staticmethod
    def unmask_rows(rows, mask_dict):
        for ix, row in enumerate(rows):
            rows[ix] = EntityMasker.unmask_seq(row, mask_dict[ix])
        
        return rows

    @staticmethod
    def mask(mat, dim, mask_dict):
        if dim == 1:
            return EntityMasker.mask_seq(mat, mask_dict)
        
        for ix, row in enumerate(mat):
            mat[ix] = EntityMasker.mask(row, dim-1, mask_dict)

        return mat 

    @staticmethod
    def mask_seq(seq, rev_mask):
        if len(rev_mask) == 0:
            return seq

        nseq = []
        for tok in seq:
            if tok in rev_mask.keys():
                nseq.append(rev_mask[tok])
            else:
                nseq.append(tok)

        return nseq

    @staticmethod
    def unmask_seq(seq, mask):
        if len(mask) == 0:
            return seq

        nseq = []
        for tok in seq:
            if tok in mask.keys():
                nseq.append(mask[tok])
            else:
                nseq.append(tok)

        return nseq
