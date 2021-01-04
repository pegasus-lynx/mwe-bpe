from collections import OrderedDict
import random

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
