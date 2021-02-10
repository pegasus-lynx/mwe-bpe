from lib.dataset import Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    files = {
        'src': Path('/home/parzival/Repos/bigram-bpe/data/proc/parallel/filtered/train.en.txt'),
        'tgt': Path('/home/parzival/Repos/bigram-bpe/data/proc/parallel/filtered/train.hi.txt')
    }

    keys = ['src', 'tgt']

    ds = Dataset.load(keys, files, split=' ')
    lens = dict()
    for ix, row in tqdm(enumerate(ds)):
        src, tgt = row
        if len(src) not in lens.keys():
            lens[len(src)] = []
        lens[len(src)].append(ix)
    total = len(ds)

    lens_freq = {key:len(value) for key, value in lens.items()}    
    lens_pick = {key:int(value*10000//total)+(1 if value > 2 else 0) for key,value in lens_freq.items()} # int will round down the value so added +1 
    # print(lens_pick)
    picks = sum(lens_pick.values())
    if picks != 10000:
        lens_pick[1] += 10000-picks

    dev_set = set()
    test_set = set()
    for k in lens.keys():
        pick = lens_pick[k]
        dev_set.update(lens[k][:pick])
        test_set.update(lens[k][pick:2*pick])

    train_ds = Dataset(keys)
    dev_ds = Dataset(keys)
    test_ds = Dataset(keys)

    for ix, row in tqdm(enumerate(ds)):
        if ix in dev_set:
            dev_ds.append(row, keys=keys)
        elif ix in test_set:
            test_ds.append(row, keys=keys)
        else:
            train_ds.append(row, keys=keys)

    # print(len(train_ds))
    # print(len(test_ds))
    # print(len(dev_ds))

    print('Writing train files')
    Dataset.save(train_ds, {
        'src': Path('/home/parzival/Repos/bigram-bpe/data/proc/parallel/split/train.en.txt'),
        'tgt': Path('/home/parzival/Repos/bigram-bpe/data/proc/parallel/split/train.hi.txt')
    })

    print('Writing dev files')
    Dataset.save(dev_ds, {
        'src': Path('/home/parzival/Repos/bigram-bpe/data/proc/parallel/split/dev.en.txt'),
        'tgt': Path('/home/parzival/Repos/bigram-bpe/data/proc/parallel/split/dev.hi.txt')
    })

    print('Writing test files')
    Dataset.save(test_ds, {
        'src': Path('/home/parzival/Repos/bigram-bpe/data/proc/parallel/split/test.en.txt'),
        'tgt': Path('/home/parzival/Repos/bigram-bpe/data/proc/parallel/split/test.hi.txt')
    })
