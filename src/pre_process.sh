# echo 'Running BPE Vocab Scripts'
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.en.txt ../data/proc/parallel/filtered/dev.en.txt -v 2000 -t bpe -x bpe.2k.en.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.en.txt ../data/proc/parallel/filtered/dev.en.txt -v 4000 -t bpe -x bpe.4k.en.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.en.txt ../data/proc/parallel/filtered/dev.en.txt -v 8000 -t bpe -x bpe.8k.en.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.en.txt ../data/proc/parallel/filtered/dev.en.txt -v 16000 -t bpe -x bpe.16k.en.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.en.txt ../data/proc/parallel/filtered/dev.en.txt -v 32000 -t bpe -x bpe.32k.en.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.en.txt ../data/proc/parallel/filtered/dev.en.txt -v 48000 -t bpe -x bpe.48k.en.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.en.txt ../data/proc/parallel/filtered/dev.en.txt -v 64000 -t bpe -x bpe.64k.en.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.hi.txt ../data/proc/parallel/filtered/dev.hi.txt -v 64000 -t bpe -x bpe.64k.hi.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.hi.txt ../data/proc/parallel/filtered/dev.hi.txt -v 48000 -t bpe -x bpe.48k.hi.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.hi.txt ../data/proc/parallel/filtered/dev.hi.txt -v 32000 -t bpe -x bpe.32k.hi.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.hi.txt ../data/proc/parallel/filtered/dev.hi.txt -v 16000 -t bpe -x bpe.16k.hi.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.hi.txt ../data/proc/parallel/filtered/dev.hi.txt -v 8000 -t bpe -x bpe.8k.hi.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.hi.txt ../data/proc/parallel/filtered/dev.hi.txt -v 4000 -t bpe -x bpe.4k.hi.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.hi.txt ../data/proc/parallel/filtered/dev.hi.txt -v 2000 -t bpe -x bpe.2k.hi.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.hi.txt ../data/proc/parallel/filtered/dev.hi.txt -v 300000 -t word -x word.max.hi.model
# python -m scripts.make_vocab -w ../data/exps_/raw-enhi -f ../data/proc/parallel/filtered/train.en.txt ../data/proc/parallel/filtered/dev.en.txt -v 300000 -t word -x word.max.en.model

# echo 'Running Match Vocab Scripts'
# python -m sripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.en.model -b ../data/exps_/raw-enhi/bpe.2k.en.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.en.model -b ../data/exps_/raw-enhi/bpe.2k.en.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.en.model -b ../data/exps_/raw-enhi/bpe.2k.en.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.en.model -b ../data/exps_/raw-enhi/bpe.4k.en.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.en.model -b ../data/exps_/raw-enhi/bpe.8k.en.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.en.model -b ../data/exps_/raw-enhi/bpe.16k.en.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.en.model -b ../data/exps_/raw-enhi/bpe.32k.en.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.en.model -b ../data/exps_/raw-enhi/bpe.48k.en.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.en.model -b ../data/exps_/raw-enhi/bpe.64k.en.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.hi.model -b ../data/exps_/raw-enhi/bpe.64k.hi.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.hi.model -b ../data/exps_/raw-enhi/bpe.48k.hi.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.hi.model -b ../data/exps_/raw-enhi/bpe.32k.hi.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.hi.model -b ../data/exps_/raw-enhi/bpe.16k.hi.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.hi.model -b ../data/exps_/raw-enhi/bpe.8k.hi.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.hi.model -b ../data/exps_/raw-enhi/bpe.4k.hi.model
# python -m scripts.match_vocab -w ../data/exps_/raw-enhi -v ../data/exps_/raw-enhi/word.max.hi.model -b ../data/exps_/raw-enhi/bpe.2k.hi.model

# echo 'Running Data Prep Script'
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/bpe.2k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.2k.hi.model -w ../data/exps_/raw-enhi/data.2k -x valid
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 110 --src_vocab ../data/exps_/raw-enhi/bpe.2k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.2k.hi.model -w ../data/exps_/raw-enhi/data.2k -x train
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/bpe.4k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.4k.hi.model -w ../data/exps_/raw-enhi/data.4k -x valid
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 110 --src_vocab ../data/exps_/raw-enhi/bpe.4k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.4k.hi.model -w ../data/exps_/raw-enhi/data.4k -x train
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/bpe.8k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.8k.hi.model -w ../data/exps_/raw-enhi/data.8k -x valid
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 110 --src_vocab ../data/exps_/raw-enhi/bpe.8k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.8k.hi.model -w ../data/exps_/raw-enhi/data.8k -x train
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/bpe.16k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.16k.hi.model -w ../data/exps_/raw-enhi/data.16k -x valid
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 110 --src_vocab ../data/exps_/raw-enhi/bpe.16k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.16k.hi.model -w ../data/exps_/raw-enhi/data.16k -x train
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/bpe.32k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.32k.hi.model -w ../data/exps_/raw-enhi/data.32k -x valid
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 110 --src_vocab ../data/exps_/raw-enhi/bpe.32k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.32k.hi.model -w ../data/exps_/raw-enhi/data.32k -x train
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/bpe.48k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.48k.hi.model -w ../data/exps_/raw-enhi/data.48k -x valid
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 110 --src_vocab ../data/exps_/raw-enhi/bpe.48k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.48k.hi.model -w ../data/exps_/raw-enhi/data.48k -x train
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/bpe.64k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.64k.hi.model -w ../data/exps_/raw-enhi/data.64k -x valid
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 110 --src_vocab ../data/exps_/raw-enhi/bpe.64k.en.model --tgt_vocab ../data/exps_/raw-enhi/bpe.64k.hi.model -w ../data/exps_/raw-enhi/data.64k -x train

# echo 'Preparing N-Gram Vocabs'
# python -m scripts.make_ngrams -d ../data/exps_/raw-enhi/data.2k/train.tsv ../data/exps_/raw-enhi/data.2k/valid.tsv -w ../data/exps_/raw-enhi/data.2k -a 2000 -n 2 3 -m src ../data/exps_/raw-enhi/match.bpe.2k.en.word.model tgt ../data/exps_/raw-enhi/match.bpe.2k.hi.word.model -b src ../data/exps_/raw-enhi/bpe.2k.en.model tgt ../data/exps_/raw-enhi/bpe.2k.hi.model
# python -m scripts.make_ngrams -d ../data/exps_/raw-enhi/data.4k/train.tsv ../data/exps_/raw-enhi/data.4k/valid.tsv -w ../data/exps_/raw-enhi/data.4k -a 4000 -n 2 3 -m src ../data/exps_/raw-enhi/match.bpe.4k.en.word.model tgt ../data/exps_/raw-enhi/match.bpe.4k.hi.word.model -b src ../data/exps_/raw-enhi/bpe.4k.en.model tgt ../data/exps_/raw-enhi/bpe.4k.hi.model
# python -m scripts.make_ngrams -d ../data/exps_/raw-enhi/data.8k/train.tsv ../data/exps_/raw-enhi/data.8k/valid.tsv -w ../data/exps_/raw-enhi/data.8k -a 8000 -n 2 3 -m src ../data/exps_/raw-enhi/match.bpe.8k.en.word.model tgt ../data/exps_/raw-enhi/match.bpe.8k.hi.word.model -b src ../data/exps_/raw-enhi/bpe.8k.en.model tgt ../data/exps_/raw-enhi/bpe.8k.hi.model
# python -m scripts.make_ngrams -d ../data/exps_/raw-enhi/data.16k/train.tsv ../data/exps_/raw-enhi/data.16k/valid.tsv -w ../data/exps_/raw-enhi/data.16k -a 16000 -n 2 3 -m src ../data/exps_/raw-enhi/match.bpe.16k.en.word.model tgt ../data/exps_/raw-enhi/match.bpe.16k.hi.word.model -b src ../data/exps_/raw-enhi/bpe.16k.en.model tgt ../data/exps_/raw-enhi/bpe.16k.hi.model
# python -m scripts.make_ngrams -d ../data/exps_/raw-enhi/data.32k/train.tsv ../data/exps_/raw-enhi/data.32k/valid.tsv -w ../data/exps_/raw-enhi/data.32k -a 32000 -n 2 3 -m src ../data/exps_/raw-enhi/match.bpe.32k.en.word.model tgt ../data/exps_/raw-enhi/match.bpe.32k.hi.word.model -b src ../data/exps_/raw-enhi/bpe.32k.en.model tgt ../data/exps_/raw-enhi/bpe.32k.hi.model
# python -m scripts.make_ngrams -d ../data/exps_/raw-enhi/data.48k/train.tsv ../data/exps_/raw-enhi/data.48k/valid.tsv -w ../data/exps_/raw-enhi/data.48k -a 48000 -n 2 3 -m src ../data/exps_/raw-enhi/match.bpe.48k.en.word.model tgt ../data/exps_/raw-enhi/match.bpe.48k.hi.word.model -b src ../data/exps_/raw-enhi/bpe.48k.en.model tgt ../data/exps_/raw-enhi/bpe.48k.hi.model
# python -m scripts.make_ngrams -d ../data/exps_/raw-enhi/data.64k/train.tsv ../data/exps_/raw-enhi/data.64k/valid.tsv -w ../data/exps_/raw-enhi/data.64k -a 64000 -n 2 3 -m src ../data/exps_/raw-enhi/match.bpe.64k.en.word.model tgt ../data/exps_/raw-enhi/match.bpe.64k.hi.word.model -b src ../data/exps_/raw-enhi/bpe.64k.en.model tgt ../data/exps_/raw-enhi/bpe.64k.hi.model

# echo 'Merging Vocabs'

# echo 'Merging bpe and bigrams : en'
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.2k -b ../data/exps_/raw-enhi/bpe.2k.en.model -d ../data/exps_/raw-enhi/data.2k/ngrams/ngrams.2.bpe.2k.en.model -s 2000 -x vocabs.b2.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.4k -b ../data/exps_/raw-enhi/bpe.4k.en.model -d ../data/exps_/raw-enhi/data.4k/ngrams/ngrams.2.bpe.4k.en.model -s 4000 -x vocabs.b2.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.8k -b ../data/exps_/raw-enhi/bpe.8k.en.model -d ../data/exps_/raw-enhi/data.8k/ngrams/ngrams.2.bpe.8k.en.model -s 8000 -x vocabs.b2.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.16k -b ../data/exps_/raw-enhi/bpe.16k.en.model -d ../data/exps_/raw-enhi/data.16k/ngrams/ngrams.2.bpe.16k.en.model -s 16000 -x vocabs.b2.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.32k -b ../data/exps_/raw-enhi/bpe.32k.en.model -d ../data/exps_/raw-enhi/data.32k/ngrams/ngrams.2.bpe.32k.en.model -s 32000 -x vocabs.b2.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.48k -b ../data/exps_/raw-enhi/bpe.48k.en.model -d ../data/exps_/raw-enhi/data.48k/ngrams/ngrams.2.bpe.48k.en.model -s 48000 -x vocabs.b2.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.64k -b ../data/exps_/raw-enhi/bpe.64k.en.model -d ../data/exps_/raw-enhi/data.64k/ngrams/ngrams.2.bpe.64k.en.model -s 64000 -x vocabs.b2.en.model

# echo 'Merging bpe and bigrams : hi'
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.2k -b ../data/exps_/raw-enhi/bpe.2k.hi.model -d ../data/exps_/raw-enhi/data.2k/ngrams/ngrams.2.bpe.2k.hi.model -s 2000 -x vocabs.b2.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.4k -b ../data/exps_/raw-enhi/bpe.4k.hi.model -d ../data/exps_/raw-enhi/data.4k/ngrams/ngrams.2.bpe.4k.hi.model -s 4000 -x vocabs.b2.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.8k -b ../data/exps_/raw-enhi/bpe.8k.hi.model -d ../data/exps_/raw-enhi/data.8k/ngrams/ngrams.2.bpe.8k.hi.model -s 8000 -x vocabs.b2.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.16k -b ../data/exps_/raw-enhi/bpe.16k.hi.model -d ../data/exps_/raw-enhi/data.16k/ngrams/ngrams.2.bpe.16k.hi.model -s 16000 -x vocabs.b2.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.32k -b ../data/exps_/raw-enhi/bpe.32k.hi.model -d ../data/exps_/raw-enhi/data.32k/ngrams/ngrams.2.bpe.32k.hi.model -s 32000 -x vocabs.b2.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.48k -b ../data/exps_/raw-enhi/bpe.48k.hi.model -d ../data/exps_/raw-enhi/data.48k/ngrams/ngrams.2.bpe.48k.hi.model -s 48000 -x vocabs.b2.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.64k -b ../data/exps_/raw-enhi/bpe.64k.hi.model -d ../data/exps_/raw-enhi/data.64k/ngrams/ngrams.2.bpe.64k.hi.model -s 64000 -x vocabs.b2.hi.model

# echo 'Merging bpe, bigrams and trigrams : en'
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.2k -b ../data/exps_/raw-enhi/bpe.2k.en.model -d ../data/exps_/raw-enhi/data.2k/ngrams/ngrams.2.bpe.2k.en.model ../data/exps_/raw-enhi/data.2k/ngrams/ngrams.3.bpe.2k.en.model -s 2000 -x vocabs.b23.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.4k -b ../data/exps_/raw-enhi/bpe.4k.en.model -d ../data/exps_/raw-enhi/data.4k/ngrams/ngrams.2.bpe.4k.en.model ../data/exps_/raw-enhi/data.4k/ngrams/ngrams.3.bpe.4k.en.model -s 4000 -x vocabs.b23.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.8k -b ../data/exps_/raw-enhi/bpe.8k.en.model -d ../data/exps_/raw-enhi/data.8k/ngrams/ngrams.2.bpe.8k.en.model ../data/exps_/raw-enhi/data.8k/ngrams/ngrams.3.bpe.8k.en.model -s 8000 -x vocabs.b23.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.16k -b ../data/exps_/raw-enhi/bpe.16k.en.model -d ../data/exps_/raw-enhi/data.16k/ngrams/ngrams.2.bpe.16k.en.model ../data/exps_/raw-enhi/data.16k/ngrams/ngrams.3.bpe.16k.en.model -s 16000 -x vocabs.b23.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.32k -b ../data/exps_/raw-enhi/bpe.32k.en.model -d ../data/exps_/raw-enhi/data.32k/ngrams/ngrams.2.bpe.32k.en.model ../data/exps_/raw-enhi/data.32k/ngrams/ngrams.3.bpe.32k.en.model -s 32000 -x vocabs.b23.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.48k -b ../data/exps_/raw-enhi/bpe.48k.en.model -d ../data/exps_/raw-enhi/data.48k/ngrams/ngrams.2.bpe.48k.en.model ../data/exps_/raw-enhi/data.48k/ngrams/ngrams.3.bpe.48k.en.model -s 48000 -x vocabs.b23.en.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.64k -b ../data/exps_/raw-enhi/bpe.64k.en.model -d ../data/exps_/raw-enhi/data.64k/ngrams/ngrams.2.bpe.64k.en.model ../data/exps_/raw-enhi/data.64k/ngrams/ngrams.3.bpe.64k.en.model -s 64000 -x vocabs.b23.en.model

# echo 'Merging bpe, bigrams and trigrams : hi'
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.2k -b ../data/exps_/raw-enhi/bpe.2k.hi.model -d ../data/exps_/raw-enhi/data.2k/ngrams/ngrams.2.bpe.2k.hi.model ../data/exps_/raw-enhi/data.2k/ngrams/ngrams.3.bpe.2k.hi.model -s 2000 -x vocabs.b23.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.4k -b ../data/exps_/raw-enhi/bpe.4k.hi.model -d ../data/exps_/raw-enhi/data.4k/ngrams/ngrams.2.bpe.4k.hi.model ../data/exps_/raw-enhi/data.4k/ngrams/ngrams.3.bpe.4k.hi.model -s 4000 -x vocabs.b23.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.8k -b ../data/exps_/raw-enhi/bpe.8k.hi.model -d ../data/exps_/raw-enhi/data.8k/ngrams/ngrams.2.bpe.8k.hi.model ../data/exps_/raw-enhi/data.8k/ngrams/ngrams.3.bpe.8k.hi.model -s 8000 -x vocabs.b23.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.16k -b ../data/exps_/raw-enhi/bpe.16k.hi.model -d ../data/exps_/raw-enhi/data.16k/ngrams/ngrams.2.bpe.16k.hi.model ../data/exps_/raw-enhi/data.16k/ngrams/ngrams.3.bpe.16k.hi.model -s 16000 -x vocabs.b23.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.32k -b ../data/exps_/raw-enhi/bpe.32k.hi.model -d ../data/exps_/raw-enhi/data.32k/ngrams/ngrams.2.bpe.32k.hi.model ../data/exps_/raw-enhi/data.32k/ngrams/ngrams.3.bpe.32k.hi.model -s 32000 -x vocabs.b23.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.48k -b ../data/exps_/raw-enhi/bpe.48k.hi.model -d ../data/exps_/raw-enhi/data.48k/ngrams/ngrams.2.bpe.48k.hi.model ../data/exps_/raw-enhi/data.48k/ngrams/ngrams.3.bpe.48k.hi.model -s 48000 -x vocabs.b23.hi.model
# python -m scripts.merge_vocab -w ../data/exps_/raw-enhi/data.64k -b ../data/exps_/raw-enhi/bpe.64k.hi.model -d ../data/exps_/raw-enhi/data.64k/ngrams/ngrams.2.bpe.64k.hi.model ../data/exps_/raw-enhi/data.64k/ngrams/ngrams.3.bpe.64k.hi.model -s 64000 -x vocabs.b23.hi.model

# echo 'Preparing the mod data : b2'
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.2k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.2k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.2k -x valid.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.2k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.2k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.2k -x train.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.4k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.4k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.4k -x valid.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.4k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.4k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.4k -x train.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.8k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.8k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.8k -x valid.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.8k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.8k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.8k -x train.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.16k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.16k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.16k -x valid.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.16k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.16k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.16k -x train.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.32k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.32k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.32k -x valid.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.32k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.32k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.32k -x train.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.48k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.48k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.48k -x valid.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.48k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.48k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.48k -x train.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.64k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.64k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.64k -x valid.b2
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.64k/vocabs.b2.en.model --tgt_vocab ../data/exps_/raw-enhi/data.64k/vocabs.b2.hi.model -w ../data/exps_/raw-enhi/data.64k -x train.b2

# echo 'Preparing the mod data : b23'
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.2k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.2k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.2k -x valid.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.2k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.2k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.2k -x train.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.4k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.4k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.4k -x valid.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.4k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.4k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.4k -x train.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.8k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.8k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.8k -x valid.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.8k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.8k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.8k -x train.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.16k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.16k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.16k -x valid.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.16k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.16k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.16k -x train.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.32k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.32k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.32k -x valid.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.32k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.32k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.32k -x train.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.48k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.48k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.48k -x valid.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.48k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.48k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.48k -x train.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/dev.en.txt -t ../data/proc/parallel/filtered/dev.hi.txt -m 001 --src_vocab ../data/exps_/raw-enhi/data.64k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.64k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.64k -x valid.b23
# python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True  -s ../data/proc/parallel/filtered/train.en.txt -t ../data/proc/parallel/filtered/train.hi.txt -m 100 --src_vocab ../data/exps_/raw-enhi/data.64k/vocabs.b23.en.model --tgt_vocab ../data/exps_/raw-enhi/data.64k/vocabs.b23.hi.model -w ../data/exps_/raw-enhi/data.64k -x train.b23

# echo 'Copying data files to the corresponding folders'

# echo 'Making data.2k'
# cd ~/Repos/bigram-bpe/data/exps_/raw-enhi/data.2k
# mkdir ../../data.2k/base/data | cp train.db valid.tsv.gz ../bpe.2k.en.model ../bpe.2k.hi.model ../../data.2k/base/data
# mkdir ../../data.2k/2gram/data | cp train.b2.db valid.b2.tsv.gz vocabs.b2.en.model vocabs.b2.hi.model ../../data.2k/2gram/data
# mkdir ../../data.2k/3gram/data | cp train.b23.db valid.b23.tsv.gz vocabs.b23.en.model vocabs.b23.hi.model ../../data.2k/3gram/data
# touch ../../data.2k/PREPARED_ ../../data.2k/conf.yml

# echo 'Making data.4k'
# cd ~/Repos/bigram-bpe/data/exps_/raw-enhi/data.4k
# mkdir ../../data.4k/base/data | cp train.db valid.tsv.gz ../bpe.4k.en.model ../bpe.4k.hi.model ../../data.4k/base/data
# mkdir ../../data.4k/2gram/data | cp train.b2.db valid.b2.tsv.gz vocabs.b2.en.model vocabs.b2.hi.model ../../data.4k/2gram/data
# mkdir ../../data.4k/3gram/data | cp train.b23.db valid.b23.tsv.gz vocabs.b23.en.model vocabs.b23.hi.model ../../data.4k/3gram/data
# touch ../../data.4k/PREPARED_ ../../data.4k/conf.yml

# echo 'Making data.8k'
# cd ~/Repos/bigram-bpe/data/exps_/raw-enhi/data.8k
# mkdir ../../data.8k/base/data | cp train.db valid.tsv.gz ../bpe.8k.en.model ../bpe.8k.hi.model ../../data.8k/base/data
# mkdir ../../data.8k/2gram/data | cp train.b2.db valid.b2.tsv.gz vocabs.b2.en.model vocabs.b2.hi.model ../../data.8k/2gram/data
# mkdir ../../data.8k/3gram/data | cp train.b23.db valid.b23.tsv.gz vocabs.b23.en.model vocabs.b23.hi.model ../../data.8k/3gram/data
# touch ../../data.8k/PREPARED_ ../../data.8k/conf.yml

# echo 'Making data.16k'
# cd ~/Repos/bigram-bpe/data/exps_/raw-enhi/data.16k
# mkdir ../../data.16k/base/data | cp train.db valid.tsv.gz ../bpe.16k.en.model ../bpe.16k.hi.model ../../data.16k/base/data
# mkdir ../../data.16k/2gram/data | cp train.b2.db valid.b2.tsv.gz vocabs.b2.en.model vocabs.b2.hi.model ../../data.16k/2gram/data
# mkdir ../../data.16k/3gram/data | cp train.b23.db valid.b23.tsv.gz vocabs.b23.en.model vocabs.b23.hi.model ../../data.16k/3gram/data
# touch ../../data.16k/PREPARED_ ../../data.16k/conf.yml

# echo 'Making data.32k'
# cd ~/Repos/bigram-bpe/data/exps_/raw-enhi/data.32k
# mkdir ../../data.32k/base/data | cp train.db valid.tsv.gz ../bpe.32k.en.model ../bpe.32k.hi.model ../../data.32k/base/data
# mkdir ../../data.32k/2gram/data | cp train.b2.db valid.b2.tsv.gz vocabs.b2.en.model vocabs.b2.hi.model ../../data.32k/2gram/data
# mkdir ../../data.32k/3gram/data | cp train.b23.db valid.b23.tsv.gz vocabs.b23.en.model vocabs.b23.hi.model ../../data.32k/3gram/data
# touch ../../data.32k/PREPARED_ ../../data.32k/conf.yml

# echo 'Making data.48k'
# cd ~/Repos/bigram-bpe/data/exps_/raw-enhi/data.48k
# mkdir ../../data.48k/base/data | cp train.db valid.tsv.gz ../bpe.48k.en.model ../bpe.48k.hi.model ../../data.48k/base/data
# mkdir ../../data.48k/2gram/data | cp train.b2.db valid.b2.tsv.gz vocabs.b2.en.model vocabs.b2.hi.model ../../data.48k/2gram/data
# mkdir ../../data.48k/3gram/data | cp train.b23.db valid.b23.tsv.gz vocabs.b23.en.model vocabs.b23.hi.model ../../data.48k/3gram/data
# touch ../../data.48k/PREPARED_ ../../data.48k/conf.yml

# echo 'Making data.64k'
# cd ~/Repos/bigram-bpe/data/exps_/raw-enhi/data.64k
# mkdir ../../data.64k/base/data | cp train.db valid.tsv.gz ../bpe.64k.en.model ../bpe.64k.hi.model ../../data.64k/base/data
# mkdir ../../data.64k/2gram/data | cp train.b2.db valid.b2.tsv.gz vocabs.b2.en.model vocabs.b2.hi.model ../../data.64k/2gram/data
# mkdir ../../data.64k/3gram/data | cp train.b23.db valid.b23.tsv.gz vocabs.b23.en.model vocabs.b23.hi.model ../../data.64k/3gram/data
# touch ../../data.64k/PREPARED_ ../../data.64k/conf.yml

# cd ~/Repos/bigram-bpe/data/exps_/data.2k
# touch base/PREPARED_ base/conf.yml
# touch 2gram/PREPARED_ 2gram/conf.yml
# touch 3gram/PREPARED_ 3gram/conf.yml
# rm -rf PREPARED_ conf.yml

# cd ~/Repos/bigram-bpe/data/exps_/data.4k
# touch base/PREPARED_ base/conf.yml
# touch 2gram/PREPARED_ 2gram/conf.yml
# touch 3gram/PREPARED_ 3gram/conf.yml
# rm -rf PREPARED_ conf.yml

# cd ~/Repos/bigram-bpe/data/exps_/data.8k
# touch base/PREPARED_ base/conf.yml
# touch 2gram/PREPARED_ 2gram/conf.yml
# touch 3gram/PREPARED_ 3gram/conf.yml
# rm -rf PREPARED_ conf.yml

# cd ~/Repos/bigram-bpe/data/exps_/data.16k
# touch base/PREPARED_ base/conf.yml
# touch 2gram/PREPARED_ 2gram/conf.yml
# touch 3gram/PREPARED_ 3gram/conf.yml
# rm -rf PREPARED_ conf.yml

# cd ~/Repos/bigram-bpe/data/exps_/data.32k
# touch base/PREPARED_ base/conf.yml
# touch 2gram/PREPARED_ 2gram/conf.yml
# touch 3gram/PREPARED_ 3gram/conf.yml
# rm -rf PREPARED_ conf.yml

# cd ~/Repos/bigram-bpe/data/exps_/data.48k
# touch base/PREPARED_ base/conf.yml
# touch 2gram/PREPARED_ 2gram/conf.yml
# touch 3gram/PREPARED_ 3gram/conf.yml
# rm -rf PREPARED_ conf.yml

# cd ~/Repos/bigram-bpe/data/exps_/data.64k
# touch base/PREPARED_ base/conf.yml
# touch 2gram/PREPARED_ 2gram/conf.yml
# touch 3gram/PREPARED_ 3gram/conf.yml
# rm -rf PREPARED_ conf.yml

# echo 'Renaming files for training'

# cd ~/Repos/bigram-bpe/data/exps_/data.2k/base/data
# mv bpe.2k.en.model nlcodec.src.model
# mv bpe.2k.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.2k/2gram/data
# mv train.b2.db train.db
# mv valid.b2.tsv.gz valid.tsv.gz
# mv vocabs.b2.en.model nlcodec.src.model
# mv vocabs.b2.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.2k/3gram/data
# mv train.b23.db train.db
# mv valid.b23.tsv.gz valid.tsv.gz
# mv vocabs.b23.en.model nlcodec.src.model
# mv vocabs.b23.hi.model nlcodec.tgt.model

# cd ~/Repos/bigram-bpe/data/exps_/data.4k/base/data
# mv bpe.4k.en.model nlcodec.src.model
# mv bpe.4k.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.4k/2gram/data
# mv train.b2.db train.db
# mv valid.b2.tsv.gz valid.tsv.gz
# mv vocabs.b2.en.model nlcodec.src.model
# mv vocabs.b2.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.4k/3gram/data
# mv train.b23.db train.db
# mv valid.b23.tsv.gz valid.tsv.gz
# mv vocabs.b23.en.model nlcodec.src.model
# mv vocabs.b23.hi.model nlcodec.tgt.model

# cd ~/Repos/bigram-bpe/data/exps_/data.8k/base/data
# mv bpe.8k.en.model nlcodec.src.model
# mv bpe.8k.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.8k/2gram/data
# mv train.b2.db train.db
# mv valid.b2.tsv.gz valid.tsv.gz
# mv vocabs.b2.en.model nlcodec.src.model
# mv vocabs.b2.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.8k/3gram/data
# mv train.b23.db train.db
# mv valid.b23.tsv.gz valid.tsv.gz
# mv vocabs.b23.en.model nlcodec.src.model
# mv vocabs.b23.hi.model nlcodec.tgt.model

# cd ~/Repos/bigram-bpe/data/exps_/data.16k/base/data
# mv bpe.16k.en.model nlcodec.src.model
# mv bpe.16k.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.16k/2gram/data
# mv train.b2.db train.db
# mv valid.b2.tsv.gz valid.tsv.gz
# mv vocabs.b2.en.model nlcodec.src.model
# mv vocabs.b2.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.16k/3gram/data
# mv train.b23.db train.db
# mv valid.b23.tsv.gz valid.tsv.gz
# mv vocabs.b23.en.model nlcodec.src.model
# mv vocabs.b23.hi.model nlcodec.tgt.model

# cd ~/Repos/bigram-bpe/data/exps_/data.32k/base/data
# mv bpe.32k.en.model nlcodec.src.model
# mv bpe.32k.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.32k/2gram/data
# mv train.b2.db train.db
# mv valid.b2.tsv.gz valid.tsv.gz
# mv vocabs.b2.en.model nlcodec.src.model
# mv vocabs.b2.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.32k/3gram/data
# mv train.b23.db train.db
# mv valid.b23.tsv.gz valid.tsv.gz
# mv vocabs.b23.en.model nlcodec.src.model
# mv vocabs.b23.hi.model nlcodec.tgt.model

# cd ~/Repos/bigram-bpe/data/exps_/data.48k/base/data
# mv bpe.48k.en.model nlcodec.src.model
# mv bpe.48k.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.48k/2gram/data
# mv train.b2.db train.db
# mv valid.b2.tsv.gz valid.tsv.gz
# mv vocabs.b2.en.model nlcodec.src.model
# mv vocabs.b2.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.48k/3gram/data
# mv train.b23.db train.db
# mv valid.b23.tsv.gz valid.tsv.gz
# mv vocabs.b23.en.model nlcodec.src.model
# mv vocabs.b23.hi.model nlcodec.tgt.model

# cd ~/Repos/bigram-bpe/data/exps_/data.64k/base/data
# mv bpe.64k.en.model nlcodec.src.model
# mv bpe.64k.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.64k/2gram/data
# mv train.b2.db train.db
# mv valid.b2.tsv.gz valid.tsv.gz
# mv vocabs.b2.en.model nlcodec.src.model
# mv vocabs.b2.hi.model nlcodec.tgt.model
# cd ~/Repos/bigram-bpe/data/exps_/data.64k/3gram/data
# mv train.b23.db train.db
# mv valid.b23.tsv.gz valid.tsv.gz
# mv vocabs.b23.en.model nlcodec.src.model
# mv vocabs.b23.hi.model nlcodec.tgt.model