# BPE with N-Grams and Skip-Grams

This repository consists of scripts and files used for preparing experiments for the paper.

## Requirements

- Python 3.7
- PyTorch 1.7.1
- nlcodec 0.3.0
- RTG 0.4.0
- indicnlp 0.0.1 (Used for tokenization of data in hi)


## Code 

The code is divided into python modules and scripts that can then be used to:

- Create char, word, BPE vocabulary from tokenized data files.
- Create ngrams and skipgrams using the vocab files.
- Merge different vocabs to a main vocabulary file.
- Prepare data for experiments ( indexed data ). The data can then be used with the RTG framework to perform the experiments.
- Analysis of ngrams and skipgrams.

### Structure

```
src ------- lib
        |-- analysis ------ prep_lists.py
        |               |-- norm_lists.py
        |               |-- merge_lists.py
        |               |-- make_stats.py
        |               |-- merge_detoks.py
        |               |-- tri_bi_eval.py
        |
        |-- scripts ------- make_ngrams.py
        |               |-- make_skipgrams.py
        |               |-- make_vocab.py
        |               |-- match_vocab.py
        |               |-- merge_vocab.py
        |               |-- prep_data.py
        |-- misc
        |-- other bash scripts (for preparation of multiple experiments together)
```
 The directory consists of sub-modules that are required for the data preparation task.

## Process

For preparing the baseline data and the ngram data, the general flow of execution is as follows:
- Prepare bpe vocabs of desired vocab size
- Prepare word vocabs for the corpus
- Find the bpe tokens that are complete words
- Prepare the .tsv and .db data files using tokenized dataset files and bpe vocabs
- Use prepared datafiles and the matched token files to generate the ngram / skipgram vocabs
- Merge ngram / skipgram vocabs to the main bpe vocab file
- Prepare new datafiles with the modified vocab files

Further details about how to use the scripts are mentioned [here](src/README.md)
