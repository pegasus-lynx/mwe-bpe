""" Utils for tokenizing the text """

__author__ = "Dipesh Kumar"

import re
import json
from collections import Counter

from sacremoses import MosesTokenizer, MosesDetokenizer

# from indicnlp import loader, common
# from indicnlp.tokenize import sentence_tokenize, indic_tokenize
# from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# config_file =  open("../../config.json", "r")
# configs = json.load(config_file)

class CharTokens(object):
    symbols = [
        '.', ',', '(', ')', '[', ']', '{', '}', '!',
        ':', '-', '"', "'", ';', '<', '>', '?', '&',
        '–', '@', ' ', '\t', '\n'
    ]
    en_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    hi_digits = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
    en_chars  = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    hi_chars  = [
        'ऀ', 'ँ', 'ं', 'ः', 'ऄ', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ऌ', 
        'ऍ', 'ऎ', 'ए', 'ऐ', 'ऑ', 'ऒ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 
        'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 
        'ध', 'न', 'ऩ', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल', 'ळ', 
        'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'ऺ', 'ऻ', '़', 'ऽ', 'ा', 'ि', 'ी', 'ु', 
        'ू', 'ृ', 'ॄ', 'ॅ', 'ॆ', 'े', 'ै', 'ॉ', 'ॊ', 'ो', 'ौ', '्', 'ॎ', 'ॏ', 'ॐ', 
        '॑', '॒', '॓', '॔', 'ॕ', 'ॖ', 'ॗ', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 
        'य़', 'ॠ', 'ॡ', 'ॢ', 'ॣ', '।', '॥', '॰', 'ॱ', 'ॲ', 'ॳ', 'ॴ', 'ॵ', 
        'ॶ', 'ॷ', 'ॸ', 'ॹ', 'ॺ', 'ॻ', 'ॼ', 'ॽ', 'ॾ', 'ॿ'
    ]

    @classmethod
    def eng(cls, text:str):
        for x in text:
            digit = x in cls.en_digits
            letter = x in cls.en_chars
            symbol = x in cls.symbols
            valid = (digit or letter or symbol)
            if not valid:
                return False
        return True

    @classmethod
    def hin(cls, text:str):
        for x in text:
            digit = x in cls.hi_digits
            digit = digit or (x in cls.en_digits)
            letter = x in cls.hi_chars
            symbol = x in cls.symbols
            valid = (digit or letter or symbol)
            if not valid:
                return False
        return True

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
        # common.set_resources_path(configs["INDIC_NLP_RESOURCES"])
        # loader.load()

        # For indic languages
        # factory = IndicNormalizerFactory()
        # self.normalizer = factory.get_normalizer("hi")

        # Moses Tokenizer
        self.mts = dict()
        self.mts["en"] = MosesTokenizer(lang="en")

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
