#!/bin/bash
#PBS -l select=ncpus=6:mem=16gb:ngpus=1
#PBS -q gpu
module load cuda
module load anaconda/3
source activate bpe
cd bigram-bpe/src
python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True -m 100 --shared_vocab ../data/combo/vocabs_/bi_tri/vocabs.bt.pmi.r800.shared.model -w ../data/combo/runs_/bi_tri/data -x train -s ../data/deen/train.deu.tok -t ../data/deen/train.eng.tok
python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True -m 100 --shared_vocab ../data/combo/vocabs_/all/vocabs.bt.pmi.r800.shared.model -w ../data/combo/runs_/all/data -x train -s ../data/deen/train.deu.tok -t ../data/deen/train.eng.tok
python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True -m 001 --shared_vocab ../data/combo/vocabs_/all/vocabs.bt.pmi.r800.shared.model -w ../data/combo/runs_/all/data -x valid -s ../data/deen/tests/newstest2018_deen.deu.tok -t ../data/deen/tests/newstest2018_deen.eng.tok
python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True -m 001 --shared_vocab ../data/combo/vocabs_/bi_tri/vocabs.bt.pmi.r800.shared.model -w ../data/combo/runs_/bi_tri/data -x valid -s ../data/deen/tests/newstest2018_deen.deu.tok -t ../data/deen/tests/newstest2018_deen.eng.tok
source deactivate