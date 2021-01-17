import json
import uuid
from datetime import datetime
from typing import List, Union
from pathlib import Path
import gzip

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

class IO:
    """File opener and automatic closer"""

    def __init__(self, path, mode='r', encoding=None, errors=None):
        self.path = path if type(path) is Path else Path(path)
        self.mode = mode
        self.fd = None
        self.encoding = encoding if encoding else 'utf-8' if 't' in mode else None
        self.errors = errors if errors else 'replace'

    def __enter__(self):

        if self.path.name.endswith(".gz"):  # gzip mode
            self.fd = gzip.open(self.path, self.mode, encoding=self.encoding, errors=self.errors)
        else:
            if 'b' in self.mode:  # binary mode doesnt take encoding or errors
                self.fd = self.path.open(self.mode)
            else:
                self.fd = self.path.open(self.mode, encoding=self.encoding, errors=self.errors,
                                         newline='\n')
        return self.fd

    def __exit__(self, _type, value, traceback):
        self.fd.close()

    @classmethod
    def reader(cls, path, text=True):
        return cls(path, 'rt' if text else 'rb')

    @classmethod
    def writer(cls, path, text=True, append=False):
        return cls(path, ('a' if append else 'w') + ('t' if text else 'b'))

    @classmethod
    def get_lines(cls, path, col=0, delim='\t', line_mapper=None, newline_fix=True):
        with cls.reader(path) as inp:
            if newline_fix and delim != '\r':
                inp = (line.replace('\r', '') for line in inp)
            if col >= 0:
                inp = (line.split(delim)[col].strip() for line in inp)
            if line_mapper:
                inp = (line_mapper(line) for line in inp)
            yield from inp

    @classmethod
    def get_liness(cls, *paths, **kwargs):
        for path in paths:
            yield from cls.get_lines(path, **kwargs)

    @classmethod
    def write_lines(cls, path: Path, text):
        if isinstance(text, str):
            text = [text]
        with cls.writer(path) as out:
            for line in text:
                out.write(line)
                out.write('\n')

    @classmethod
    def copy_file(cls, src: Path, dest: Path, follow_symlinks=True):
        log.info(f"Copy {src} → {dest}")
        assert src.resolve() != dest.resolve()
        shutil.copy2(str(src), str(dest), follow_symlinks=follow_symlinks)

    @classmethod
    def maybe_backup(cls, file: Path):
        if file.exists():
            time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            dest = file.with_suffix(f'.{time}')
            log.info(f"Backup {file} → {dest}")
            file.rename(dest)

    @classmethod
    def safe_delete(cls, path: Path):
        try:
            if path.exists():
                if path.is_file():
                    log.info(f"Delete file {path}")
                    path.unlink()
                elif path.is_dir():
                    log.info(f"Delete dir {path}")
                    path.rmdir()
                else:
                    log.warning(f"Coould not delete {path}")
        except:
            log.exception(f"Error while clearning up {path}")
