# Data Preparation Scripts

This directory consists of sub-modules that are required for the data preparation task. The directory consists of two sub-modules : 
- **lib** : Contains the modules and classes for Vocabs, Tokenizer, Dataset, File and other misc function.
- **scripts** : Contains the scripts that perform unit data preparation operations. It consists of the following scripts:
  - **make_vocab** : Makes char/bpe/word vocabs given the dataset files and parameters
  - **match_vocab** : For matching the bpe tokens to a work token file
  - **merge_vocab** : Used for merging vocab files to a bpe vocab file (merged by frequency)
  - **make_ngrams** : Used for preparing ngram vocabs using the bpe, match files and processed datafiles
  - **prep_data** : Prepares data in the indexed format which can be used for training


## Process 

For preparing the baseline data and the ngram data, the general flow of execution is as follows:
- Prepare bpe vocabs of desired vocab size
```
python -m scripts.make_vocab -w path/to/work/dir -f ../data/train.en.txt ../data/dev.en.txt -v 48000 -t bpe -x bpe.48k.en.model

python -m scripts.make_vocab -w path/to/work/dir -f ../data/train.hi.txt ../data/dev.hi.txt -v 48000 -t bpe -x bpe.48k.hi.model
```
- Prepare word vocabs for the corpus
```
python -m scripts.make_vocab -w path/to/work/dir -f ../data/train.en.txt ../data/dev.en.txt -v 48000 -t word -x word.max.en.model

python -m scripts.make_vocab -w path/to/work/dir -f ../data/train.hi.txt ../data/dev.hi.txt -v 48000 -t word -x word.max.hi.model
```
- Find the bpe tokens that are complete words
```
python -m scripts.match_vocab -w path/to/work/dir -v ../data/word.max.en.model -b ../data/bpe.48k.en.model
python -m scripts.match_vocab -w path/to/work/dir -v ../data/word.max.hi.model -b ../data/bpe.48k.hi.model
```
- Prepare the .tsv and .db data files using tokenized dataset files and bpe vocabs
```
python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/dev.en.txt -t ../data/dev.hi.txt -m 011 --src_vocab ../data/bpe.48k.en.model --tgt_vocab ../data/bpe.48k.hi.model -w path/to/work/dir -x valid

python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/dev.en.txt -t ../data/dev.hi.txt -m 110 --src_vocab ../data/bpe.48k.en.model --tgt_vocab ../data/bpe.48k.hi.model -w path/to/work/dir -x train
```
- Use prepared datafiles and the matched token files to generate the nram vocabs
```
python -m scripts.make_ngrams -d ../data/data.48k/train.tsv ../data/data.48k/valid.tsv -w ../data/data.48k -a 16000 -n 2 3 -m src ../data/match.bpe.48k.en.word.model tgt ../data/match.bpe.48k.hi.word.model -b src ../data/bpe.48k.en.model tgt ../data/bpe.48k.hi.model
```
- Merge ngram vocabs to the main bpe vocab file
```
python -m scripts.merge_vocab -w ../data/data.48k -b ../data/bpe.48k.en.model -d ../data/ngrams/ngrams.2.bpe.48k.en.model -s 48000 -x vocabs.b2.en.model

python -m scripts.merge_vocab -w ../data/data.48k -b ../data/bpe.48k.hi.model -d ../data/ngrams/ngrams.2.bpe.48k.hi.model -s 48000 -x vocabs.b2.hi.model
```
- Prepare new datafiles with the modified vocab files
```
python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/dev.en.txt -t ../data/dev.hi.txt -m 001 --src_vocab ../data/ngrams/ngrams.2.bpe.48k.en.model --tgt_vocab ../data/ngrams/ngrams.2.bpe.48k.hi.model -w path/to/work/dir -x valid.b2

python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/dev.en.txt -t ../data/dev.hi.txt -m 100 --src_vocab ../data/ngrams/ngrams.2.bpe.48k.en.model --tgt_vocab ../data/ngrams/ngrams.2.bpe.48k.hi.model -w path/to/work/dir -x train.b2
```

Once we are done with the processing of the data, we will need to structure the data as follows for training:
```
work_dir ---- _PREPARED
          |-- conf.yml
          |-- data ---- train.db                (Training File)
                    |-- valid.tsv.gz            (Validation File)
                    |-- nlcodec.src.model       (Src Vocab File)
                    |-- nlcodec.tgt.model       (Tgt Vocab File)
                    |-- [nlcodec.shared.model]  (Shared Vocab File)
```

## Description of the scripts

### make_vocabs

Prepares the vocabs for given dataset files.

**Arguments**
- -w, --work_dir, type=Path : Path to the working directory
- -x, --save_file, type=str : Name of the file in which the vocab is to be stored
- -f, --files, nargs='+, type=Path : List of files to be processed
- -t, --type, type=str, choices=['char, word, bpe'], default='bpe : Type of vocabulary to build

### match_vocab

Matches a bpe vocab with a word vocab to find the tokens bpe tokens that are complete words

**Arguments**
- -w, --work_dir, type=Path : Path to the working directory
- -x, --save_file, type=str : Name of the file in which the vocab is to be stored
- -b, --bpe_vocab, type=Path : Bpe Vocab file path
- -v, --word_vocab, type=Path : Word vocab file to match the bpe token

### prep_data

Prepares the data using the vocab files and the tokenized data files. Outputs files in the format directly usable for the experiments

**Arguments**
- -w, --work_dir, type=Path : Path to the working directory
- -x, --save_file, type=str : Name of the file in which the vocab is to be stored
- -s, --src_path, type=Path : Path of the src file
- -t, --tgt_path, type=Path : Path of the tgt file
- -w, --work_dir, type=Path : Path to the working directory
- --shared_vocab, type=Path, default=None : Path to the shared vocab file (if shared vocab is being used)
- --src_vocab, type=Path, default=None : Path to the src vocab file
- --tgt_vocab, type=Path, default=None : Path to the tgt vocab file
- --src_len, type=int, default=0 : Length of the source sentences
- --tgt_len, type=int, default=0 : Length of the target sentences
- --truncate, type=bool, default=False : If true, truncate longer sentences
- -m, --save_mode, type=str, default='101' : Binary string of length 3 [ 1 - Saves .db file, 0 - Does not save .tsv files, 1 - Saves .tsv.gz files ]

### make_ngrams

Prepares ngram vocabs from the prepared data files, bpe vocab files and match files

**Arguments**
- -w, --work_dir, type=Path : Path to the working directory
- -d, --data_files, type=Path, nargs='+' : List of processed dataset files [The file must be in .tsv format]
- -n, --ngrams, type=int, nargs='+, default=[2] : List of ngrams to be prepared
- -s, --shared, type=bool, default=False : True if shared vocabs
- -m, --match_files, type=str, nargs='+' : List of pairs : [(shared, src, tgt), Path of the file]
- -b, --bpe_files, type=str, nargs='+' : List of pairs : [(shared, src, tgt), Path of the file]    
- -f, --min_freq, type=int, default=0 : Min frequency of the ngrams to be considered
- -a, --max_ngrams, type=int, default=0 : Max ngrams to be considered

### merge_vocab

Merges ngram or other vocabs to a main bpe vocab file sorted in the order of decreasing frequency

**Arguments**
- -w, --work_dir, type=Path : Path to the working directory
- -x, --save_file, type=str : Name of the file in which the vocab is to be stored
- -d, --data_files, type=Path, nargs='+' : Path of the corpus files
- -b, --bpe_file, type=Path : Path to the base bpe vocab file
- -s, --vocab_size, type=int, default=8000 : Vocab size of the merged file
