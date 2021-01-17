import os
import numpy as np
from tqdm import tqdm
from itertools import zip_longest
from pathlib import Path
import gzip
import random
import sqlite3
from collections import OrderedDict
from typing import Iterable, Iterator, Tuple, List, Union

from lib.misc import IO

Array = np.ndarray
RawRecord = Tuple[str, str]
TokRawRecord = Tuple[List[str], List[str]]
MonoSeqRecord = List[Union[int, str]]
ParallelSeqRecord = Tuple[MonoSeqRecord, MonoSeqRecord]
TokStream = Union[Iterator[Iterator[str]], Iterator[str]]

def read_parallel(parallel_file:Path):
    ds = Dataset(['src', 'tgt'])
    with open(parallel_file, 'r') as fr:
        print('\t\t > Reading parallel data file ...')
        for line in tqdm(fr):
            x, y = line.strip().split('\t')
            src = list(map(int, x.split()))
            tgt = list(map(int, y.split()))
            ds.append([src, tgt], keys=['src', 'tgt'])
    return ds

class Dataset(object):
    """
        Generic class to handle the
        text datasets and operations
    """

    def __init__(self, list_keys):
        self.lists = OrderedDict()
        self.shuffled = None
        for key in list_keys:
            self.lists[key] = []

    @staticmethod
    def merge(keys, *args):
        ''' Merge multiple datasets together '''
        ds = Dataset(keys)
        for dataset in args:
            ds.add(dataset)
        return ds

    @staticmethod
    def load(keys, files: dict):
        ''' Loads dataset from files '''
        ds = Dataset(keys)
        for key, filepath in files.items():
            with open(filepath, "r+") as f:
                for line in f:
                    line = line.strip()
                    ds.lists[key].append(line)
        return ds

    @staticmethod
    def save(dataset, files: dict):
        """
            Saves the dataset in the corresponding files
        """
        keys = files.keys()
        for key in keys:
            with open(files[key], "w+") as f:
                for row in dataset.lists[key]:
                    if type(row) == list:
                        f.write('{}\n'.format(" ".join(row)))
                    if type(row) == str:
                        f.write('{}\n'.format(row))

    @staticmethod
    def analyze(dataset):
        pass

    def add(self, dataset):
        ''' Add a dataset to the current dataset '''
        keys = dataset.lists.keys()
        for row in dataset:
            self.append(row, keys)

    def append(self, datarow, keys=None):
        ''' Appends a datarow to the dataset '''
        if keys is not None:
            for key, val in zip(keys, datarow):
                self.lists[key].append(val)
        else:
            for key, val in datarow.items():
                self.lists[key].append(val)

    def shuffle(self):
        ''' Shuffling datasets '''
        if self.shuffled is None or len(self.shuffled) != len(self):
            nrows = len(self)
            self.shuffled = [x for x in range(nrows)]
        self.shuffled = random.sample(self.shuffled, len(self.shuffled))

    def __len__(self):
        for key in self.lists:
            return len(self.lists[key])        

    def __iter__(self):
        curr = 0
        while curr < len(self):
            ret = []
            for key in self.lists.keys():
                ret.append(self.lists[key][curr])
            curr += 1
            yield tuple(ret)

    def minibatches(self, size):
        ''' Returns batches from the dataset '''
        if self.shuffled is None:
            self.shuffled = [ x for x in range(len(self))]
        batch = [[] for x in range(len(self.lists))]
        curr = 0
        for ix in self.shuffled:
            if curr == size:
                curr = 0
                yield tuple(batch)
                batch = [[] for x in range(len(self.lists))]
            for p, key in enumerate(self.lists.keys()):
                batch[p].append(self.lists[key][ix])
            curr += 1

    # Implement later
    def find(self, word: str):
        pass

    def find_all(self, word: str):
        pass

class IdExample:
    __slots__ = 'x', 'y', 'id', 'x_raw', 'y_raw', 'x_len', 'y_len'

    def __init__(self, x, y, id):
        self.x: Array = x
        self.y: Array = y
        self.id = id
        self.x_raw: Optional[str] = None
        self.y_raw: Optional[str] = None

    def val_exists_at(self, side, pos: int, exist: bool, val:int):
        assert side == 'x' or side == 'y'
        assert pos == 0 or pos == -1
        seq = self.x if side == 'x' else self.y
        if exist:
            if seq[pos] != val:
                if pos == 0:
                    seq = np.append(np.int32(val), seq)
                else: # pos = -1
                    seq = np.append(seq, np.int32(val))
                # update
                if side == 'x':
                    self.x = seq
                else:
                    self.y = seq
        else:  # should not have val at pos
            assert seq[pos] != val

    def eos_check(self, side, exist):
        raise

    def __getitem__(self, key):
        if key == 'x_len':
            return len(self.x)
        elif key == 'y_len':
            return len(self.y)
        else:
            return getattr(self, key)

class SqliteFile(Iterable[IdExample]):
    """
    Change log::
    VERSION 0: (unset)
        x_seq and y_seq were list of integers, picked using pickle.dumps
        very inefficient
    VERSION 1:
        x_seq and y_seq were np.array(, dtyp=np.int32).tobytes()

    """
    CUR_VERSION = 1

    TABLE_STATEMENT = f"""CREATE TABLE IF NOT EXISTS data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        x BLOB NOT NULL,
        y BLOB,
        x_len INTEGER,
        y_len INTEGER);"""
    INDEX_X_LEN = "CREATE INDEX IF NOT EXISTS  idx_x_len ON data (x_len);"
    INDEX_Y_LEN = "CREATE INDEX IF NOT EXISTS  idx_y_len ON data (y_len);"

    INSERT_STMT = "INSERT INTO data (x, y, x_len, y_len) VALUES (?, ?, ?, ?)"
    READ_RANDOM = "SELECT * from data ORDER BY RANDOM()"
    COUNT_ROWS = "SELECT COUNT(*) as COUNT from data"

    @classmethod
    def make_query(cls, sort_by: str, len_rand: int):
        assert len_rand >= 1
        select_no_sort = 'SELECT * from data'
        template = f"{select_no_sort} ORDER BY %s + (RANDOM() %% %d) %s"
        known_queries = dict(y_len_asc=template % ('y_len', len_rand, 'ASC'),
                             y_len_desc=template % ('y_len', len_rand, 'DESC'),
                             x_len_asc=template % ('x_len', len_rand, 'ASC'),
                             x_len_desc=template % ('x_len', len_rand, 'DESC'),
                             random=cls.READ_RANDOM,
                             eq_len_rand_batch=template % ('y_len', len_rand, 'DESC'))
        known_queries[None] = known_queries['none'] = select_no_sort
        assert sort_by in known_queries, ('sort_by must be one of ' + str(known_queries.keys()))
        return known_queries[sort_by]

    @classmethod
    def write(cls, path, records: Iterator[ParallelSeqRecord]):
        if path.exists():
            # log.warning(f"Overwriting {path} with new records")
            os.remove(str(path))
        # maybe_tmp = IO.maybe_tmpfs(path)
        # log.info(f'Creating {maybe_tmp}')
        maybe_tmp = path
        conn = sqlite3.connect(str(maybe_tmp))
        cur = conn.cursor()
        cur.execute(cls.TABLE_STATEMENT)
        cur.execute(cls.INDEX_X_LEN)
        cur.execute(cls.INDEX_Y_LEN)
        cur.execute(f"PRAGMA user_version = {cls.CUR_VERSION};")

        count = 0
        for x_seq, y_seq in records:
            # use numpy. its a lot efficient
            if not isinstance(x_seq, np.ndarray):
                x_seq = np.array(x_seq, dtype=np.int32)
            if y_seq is not None and not isinstance(y_seq, np.ndarray):
                y_seq = np.array(y_seq, dtype=np.int32)
            values = (x_seq.tobytes(),
                      None if y_seq is None else y_seq.tobytes(),
                      len(x_seq), len(y_seq) if y_seq is not None else -1)
            cur.execute(cls.INSERT_STMT, values)
            count += 1
        cur.close()
        conn.commit()
        # if maybe_tmp != path:
            # bring the file back to original location where it should be
            # copy_file(maybe_tmp, path)
        # log.info(f"stored {count} rows in {path}")

class TSVData(Iterable[IdExample]):

    def __init__(self, path: Union[str, Path], in_mem=False, shuffle=False, longest_first=True,
                 max_src_len: int = 512, max_tgt_len: int = 512, truncate: bool = False):
        """
        :param path: path to TSV file have parallel sequences
        :param in_mem: hold data in memory instead of reading from file for subsequent pass.
         Don't use in_mem for large data_sets.
        :param shuffle: shuffle data between the reads
        :param longest_first: On the first read, get the longest sequence first by sorting by length
        """
        self.path = path
        self.in_mem = in_mem or shuffle or longest_first
        self.longest_first = longest_first
        self.shuffle = shuffle
        self.truncate = truncate
        self.max_src_len, self.max_tgt_len = max_src_len, max_tgt_len
        self.mem = list(self.read_all()) if self.in_mem else None
        self._len = len(self.mem) if self.in_mem else line_count(path)
        self.read_counter = 0

    @staticmethod
    def _parse(line: str):
        return [int(t) for t in line.split()]

    def read_all(self) -> Iterator[IdExample]:
        with IO.reader(self.path) as lines:
            recs = (line.split('\t') for line in lines)
            for idx, rec in enumerate(recs):
                x = self._parse(rec[0].strip())
                y = self._parse(rec[1].strip()) if len(rec) > 1 else None
                if self.truncate:  # truncate long recs
                    x = x[:self.max_src_len]
                    y = y if y is None else y[:self.max_tgt_len]
                elif len(x) > self.max_src_len or (0 if y is None else len(y)) > self.max_tgt_len:
                    continue  # skip long recs
                if not x or (y is not None and len(y) == 0):  # empty on one side
                    log.warning(f"Ignoring an empty record  x:{len(x)}    y:{len(y)}")
                    continue
                yield IdExample(x, y, id=idx)

    def __len__(self):
        return self._len

    def __iter__(self) -> Iterator[IdExample]:
        if self.shuffle:
            if self.read_counter == 0:
                log.info("shuffling the data...")
            random.shuffle(self.mem)
        if self.longest_first:
            if self.read_counter == 0:
                log.info("Sorting the dataset by length of target sequence")
            sort_key = lambda ex: len(ex.y) if ex.y is not None else len(ex.x)
            self.mem = sorted(self.mem, key=sort_key, reverse=True)
            if self.read_counter == 0:
                log.info(f"Longest source seq length: {len(self.mem[0].x)}")

        yield from self.mem if self.mem else self.read_all()
        self.read_counter += 1

    @staticmethod
    def write_lines(lines, path):
        # log.info(f"Storing data at {path}")
        with IO.writer(path) as f:
            for line in lines:
                f.write(line)
                f.write('\n')

    @staticmethod
    def write_parallel_recs(records: Iterator[ParallelSeqRecord], path: Union[str, Path]):
        seqs = ((' '.join(map(str, x)), ' '.join(map(str, y))) for x, y in records)
        lines = (f'{x}\t{y}' for x, y in seqs)
        TSVData.write_lines(lines, path)

    @staticmethod
    def write_mono_recs(records: Iterator[MonoSeqRecord], path: Union[str, Path]):
        lines = (' '.join(map(str, rec)) for rec in records)
        TSVData.write_lines(lines, path)

    @staticmethod
    def read_raw_parallel_lines(src_path: Union[str, Path], tgt_path: Union[str, Path]) \
            -> Iterator[RawRecord]:
        with IO.reader(src_path) as src_lines, IO.reader(tgt_path) as tgt_lines:
            # if you get an exception here --> files have un equal number of lines
            recs = ((src.strip(), tgt.strip()) for src, tgt in zip_longest(src_lines, tgt_lines))
            recs = ((src, tgt) for src, tgt in recs if src and tgt)
            yield from recs

    @staticmethod
    def read_raw_parallel_recs(src_path: Union[str, Path], tgt_path: Union[str, Path],
                               truncate: bool, src_len: int, tgt_len: int, src_tokenizer,
                               tgt_tokenizer) \
            -> Iterator[ParallelSeqRecord]:
        recs = TSVData.read_raw_parallel_lines(src_path, tgt_path)

        recs = ((src_tokenizer(x), tgt_tokenizer(y)) for x, y in recs)
        if truncate:
            recs = ((src[:src_len], tgt[:tgt_len]) for src, tgt in recs)
        else:  # Filter out longer sentences
            recs = ((src, tgt) for src, tgt in recs if len(src) <= src_len and len(tgt) <= tgt_len)
        return recs

    @staticmethod
    def read_raw_mono_recs(path: Union[str, Path], truncate: bool, max_len: int, tokenizer):
        with IO.reader(path) as lines:
            recs = (tokenizer(line.strip()) for line in lines if line.strip())
            if truncate:
                recs = (rec[:max_len] for rec in recs)
            else:  # Filter out longer sentences
                recs = (rec for rec in recs if 0 < len(rec) <= max_len)
            yield from recs
