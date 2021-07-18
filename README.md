# BPE with N-Grams and Skip-Grams

This repository consists of scripts and files used for preparing experiments for the paper.

## Requirements

- Python 3.7
- PyTorch 1.7.1
- nlcodec 0.3.0
- RTG 0.4.0
- indicnlp 0.0.1 (Used for tokenization of data in hi)


## Code 

The code is basically divided into modules and scripts that can be used to :

- Prepare single runnable experiment ( to be directly used by using RTG ) given the config files / arguments.
- Prepare a batch of runnable experiments from the given config file / arguments.
- Get the list of ngrams / skipgrams / mwes , sorted by metric.
- Simple test scripts : check the encoding / decoding process for a scheme.


### Structure

```
src ------- lib -------- schemes.py
        |           |--- misc.py
        |
        |-- tests ------ enc-dec-skip.py
        |
        |-- misc
        |
        |-- getter.py
        |-- make_single.py
        |-- make_batch.py ( Incomplete )

```
 The directory consists of sub-modules that are required for the data preparation task.

### Scripts


#### make_single.py
---
This script creates a single experiment run that can be directly used in RTG ( currently addition of _PREPARED and con.yml is manual). 

The script can be provided with either a config file or the other way is pass them as arguments to this script.

Here are simple example use cases of the script:

```shell
python -m make_single -c ../data/confs/sample.single.yml -w ../data/runs.single/    
```

Here the argument -c (--conf_file) refers to the config file to be used for preparing the run and argument -w (--work_dir) refers to the working directory in which we need to save the prepared vocabs and data.
 

#### make_batch.py
---

The script creates a batch of experiment given the right parameters. 

The parameters that can vary to create range of experiments are :
- vocab_size
- max_ngrams / max_skipgrams

Configs / Parameters can be provided in the same way as mentioned above for make_single.py script.

Here are some sample use cases of the following script:
```
TBD
```

#### get_lists.py
---
This script is mainly to be used for analysis / testing purpose. The main objective is to get and save a list of ngrams / skipgrams based on different sorting metrics, etc.

Here is a sample usage of the above script:
```
python -m get_lists -c ../data/confs/sample.single.yml -w ../data/runs.single/ --sorted_ngrams
```


### Config File : prep.yml (default name)
---
All the scripts mentioned above can work with the same config script. This is an example of a sample script file:
```yml

train_src: ../data.simp/ds/enhi/train.hi.tok
train_tgt: ../data.simp/ds/enhi/train.en.tok
valid_src: ../data.simp/ds/enhi/dev.hi.tok
valid_tgt: ../data.simp/ds/enhi/dev.en.tok
truncate: True
src_len: 512
tgt_len: 512
shared: False
max_src_types: 4000
max_tgt_types: 4000
pieces: ngram
min_freq: 100
ngram_sorter: naive_pmi
include_ngrams:
  - 2
  - 3
max_ngrams: 400

```

Here are all the different parameters that can be specified for creating the experiments:
- **train_src** : type=Path, help='Path to the train src file'
- **train_tgt** : type=Path, help='Path to the train tgt file'
- **valid_src** : type=Path, help='Path to the validation src file'
- **valid_tgt** : type=Path, help='Path to the validation tgt file'
- **truncate** : type=bool, default=True, help='If true, truncates the sequence with larger lengths'
- **src_len** : type=int, help='Maximum src sequence length', default=512
- **tgt_len** : type=int, help='Maximum tgt sequence length', default=512
- **shared** : type=bool, default=False, help='Flag representing if the vocabulary is shared or not'

- **max_types** : type=int, help='Maximum shared types ( if shared == True )'
- **max_src_types** : type=int, help='Maximum src vocab types'
- **max_tgt_types** : type=int, help='Maximum tgt vocab types'

- **pieces** : type=str, choices=['char', 'word', 'bpe', 'ngram', 'skipgram', 'mwe'], default='bpe', help='Scheme for preparing the vocabulary'
- **ngram_sorter** : type=str, default='freq',choices=['freq', 'naive_pmi', 'avg_pmi', 'min_pmi'], help='Sorter function to be used for ngrams'
- **include_ngrams** : type=int, nargs='+', help='List of ngrams to be included in the vocabulary'
- **max_ngrams** : type=int, help='Maximum ngrams to be included in the vocab'
- **skipgram_sorter** : type=str, default='freq', choices=['freq', 'skip_pmi'], help='Sorter function to be used for skipgrams'
- **include_skipgrams** : type=int, nargs='+', help='List of skipgrams to be included in the vocab'
- **max_skipgrams** : type=int, help='Maximum skipgrams to be included in the vocab'

- **min_freq** : type=int, help='Minimum frequency for ngrams / sgrams / mwes'
- **min_instances** : type=int, help='Minimum number of instances', default=0
- **max_instance_probs** : type=float, help='Maximum prob for an sgram instance', default=0.1

