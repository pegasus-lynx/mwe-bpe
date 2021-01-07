import json
import uuid
from datetime import datetime
from typing import List, Union
from pathlib import Path

def load_conf(filepath):
    config_file =  open(filepath, "r")
    configs = json.load(config_file)
    return configs

def get_now():
    datetime_str = datetime.now().isoformat()
    datetime_str = datetime_str.replace('T', ' ')
    pos = datetime_str.index('.')
    return datetime_str[:pos]

def make_dir(bdir:Path, sdir:Path = None):
    """
        @param bdir: Path to the base dir
        @param sdir: Stem name for the working dir
    """

    if sdir is None:
        sdir = Path(f'dir-{uuid.uuid4()}')

    workdir = bdir / sdir
    if not workdir.exists():
        try:
            print(f'> Making working directory : {workdir.relative_to(bdir)} ...')
            workdir.mkdir()
        except Exception as e:
            print(f'> Could not make working directory : {workdir.relative_to(bdir)}. Trying again ...')
            sdir = Path(f'dir-{uuid.uuid4()}')
            workdir = bdir / sdir
            print(f'> Making working directory : {workdir.relative_to(bdir)} ...')
            workdir.mkdir()
    else:
        print(f'> Directory {workdir.relative_to(bdir)} exists.')

    return workdir

class FileReader(object):
    def __init__(self, filepaths:List[Path], tokenized=True):
        self.filepaths = filepaths
        self.tokenized = tokenized
        self.length = None

    def __iter__(self):
        for filepath in self.filepaths:
            fs = open(filepath, 'r')
            for line in fs:
                line = line.strip()
                if self.tokenized:
                    line = line.split()
                yield line

    def __len__(self):
        if self.length is not None:
            return self.length
        self.length = 0
        for filepath in self.filepaths:
            fs = open(filepath, 'r')
            self.length += len(fs.readlines())
        return self.length

    def unique(self):
        flat = set()
        for line in self:
            line = line.strip()
            flat.add(line)
        return flat

class FileWriter(object):

    def __init__(self, filepath):
        self.fw = open(filepath, 'w')
        self.tablevel = 0

    def close(self):
        self.fw.close()

    def textlines(self, texts:List[Union[str or Path]]):
        for text in texts:
            self.textline(text)

    def heading(self, text):
        self.newline()
        self.textline(text)
        self.textline(f'TIME : {get_now()}')
        self.newline()
        self.dashline()
        self.newline()

    def time(self):
        text = get_now()
        self.textline(text)

    def textline(self, text):
        self.fw.write(f'{"    "*self.tablevel}{text}\n')

    def dashline(self):
        self.fw.write(f'{"-"*50} \n')

    def sectionstart(self, text):
        self.newline()
        self.textline(text)
        self.dashline()
        self.tablevel += 1

    def sectionclose(self):
        self.dashline()
        self.tablevel = max(0, self.tablevel-1)

    def section(self, heading:str, lines:List):
        self.sectionstart(heading)
        self.textlines(lines)
        self.sectionclose()

    def newline(self):
        self.fw.write('\n')

