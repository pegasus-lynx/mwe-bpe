# Initial Variabes
sizes=(8)
# ngram_modes=('freq' 'pmi' 'ngdf')
ngram_modes=('pmi')
modes=('r')

# Base Dir
base_dir='../data/exps_/data_v8'
vdir="${base_dir}/vocabs_"
rdir="${base_dir}/runs_"

data_dir='../data/exps_/data_v3/'
vcb_dir="${data_dir}/vocabs_"

# Languages
sl='hi'
tl='en'

vsl='nlcodec.src.model'
vtl='nlcodec.tgt.model'

train_src='../data/proc/parallel/split/train.hi.txt'
train_tgt='../data/proc/parallel/split/train.en.txt'
val_src='../data/proc/parallel/split/dev.hi.orig.txt'
val_tgt='../data/proc/parallel/split/dev.en.orig.txt'
test_src='../data/proc/parallel/split/test.hi.orig.txt'
test_tgt='../data/proc/parallel/split/test.en.orig.txt'

# Initial Functions --------------------------------
make_dirs(){
    for d in "$@"; do
        if [[ ! -d $d ]]; then mkdir $d; fi
    done
}

make_val(){
    python -m scripts.prep_skipgram_data -x valid -m 001 --truncate True --src_len 512 --tgt_len 512 -w ${1} --vocab src ${2} tgt ${3} --skipgram_vocab src ${4} tgt ${5} -t ${6} -tm r -d ${7} -v so
}

make_train(){
    python -m scripts.prep_skipgram_data -x train -m 100 --truncate True --src_len 512 --tgt_len 512 -w ${1} --vocab src ${2} tgt ${3} --skipgram_vocab src ${4} tgt ${5} -t ${6} -tm r -d ${7} -v so
}

make_test(){
    python -m scripts.prep_skipgram_data -x test_ -m 010 --truncate True --src_len 512 --tgt_len 512 -w ${1} --vocab src ${2} tgt ${3} --skipgram_vocab src ${4} tgt ${5} -t ${6} -tm r -d ${7} -v so
}


# --------------------------------------------------

# Starting main script

echo 'Preparing the skip-gram data'
for sz in ${sizes[@]}; do
    ddir="${data_dir}/data.${sz}k"
    sdir="${ddir}/sgrams"
    vpr="bpe.${sz}k" 

    # python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate False -m 010 --src_vocab ${vcb_dir}/${vpr}.${sl}.model --tgt_vocab ${vcb_dir}/${vpr}.${tl}.model -w ${ddir}/untrimmed -x valid -s ${val_src} -t ${val_tgt}
    # python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate False -m 010 --src_vocab ${vcb_dir}/${vpr}.${sl}.model --tgt_vocab ${vcb_dir}/${vpr}.${tl}.model -w ${ddir}/untrimmed -x train -s ${train_src} -t ${train_tgt}
    # python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate False -m 010 --src_vocab ${vcb_dir}/${vpr}.${sl}.model --tgt_vocab ${vcb_dir}/${vpr}.${tl}.model -w ${ddir}/untrimmed -x test_ -s ${test_src} -t ${test_tgt}

    for mode in ${ngram_modes[@]}; do
        spr="sgrams.${mode}.corr0.1.cw15.bpe.${sz}k"
        # rep_tokens=($((25*$sz/2)) $(($sz*25)) $(($sz*50)))
        rep_tokens=($((25*$sz/2)) $(($sz*25)) $(($sz*50)))
        for tokens in ${rep_tokens[@]}; do
            wdir="${rdir}/${sz}k_${mode}_r${tokens}_so"
            make_dirs $wdir ${wdir}/data
            # make_val $wdir/data ${vcb_dir}/${vpr}.${sl}.model ${vcb_dir}/${vpr}.${tl}.model ${sdir}/${spr}.${sl}.model ${sdir}/${spr}.${tl}.model $tokens ${ddir}/untrimmed/valid.tsv
            # make_train $wdir/data ${vcb_dir}/${vpr}.${sl}.model ${vcb_dir}/${vpr}.${tl}.model ${sdir}/${spr}.${sl}.model ${sdir}/${spr}.${tl}.model $tokens ${ddir}/untrimmed/train.tsv
            # make_test $wdir/data ${vcb_dir}/${vpr}.${sl}.model ${vcb_dir}/${vpr}.${tl}.model ${sdir}/${spr}.${sl}.model ${sdir}/${spr}.${tl}.model $tokens ${ddir}/untrimmed/test_.tsv
            # cp ${vcb_dir}/${vpr}.${tl}.model ${wdir}/data/nlcodec.tgt.model
            # cut -f 1 $wdir/data/test_.tsv > $wdir/data/test.tsv
        done
    done
done
