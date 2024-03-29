# Initial Variabes
sizes=(1 2 4 8 16 48 64)
ngram_modes=('pmi' 'ngdf')
modes=('r')

# Base Dir
base_dir='../data/exps_/data_v5'
vdir="${base_dir}/vocabs_"
rdir="${base_dir}/runs_"

# Languages
sl='de'
tl='en'
vl='nlcodec.shared.model'

# Train and Validation Source and Target Files
train_src='../data/deen/train.deu.tok'
train_tgt='../data/deen/train.eng.tok'
val_src='../data/deen/tests/newstest2018_deen.deu.tok'
val_tgt='../data/deen/tests/newstest2018_deen.eng.tok'

# Initial Functions --------------------------------
make_dirs(){
    for d in "$@"; do
        if [[ ! -d $d ]]; then mkdir $d; fi
    done
}

prep_run(){
    python -m scripts.prep_data -w $1 -x $3 -s $4 -t $5 --src_len 512 --tgt_len 512 --truncate True -m $2 --shared_vocab $1/${vl}
}

prep_exp(){
    make_dirs $1 $1/data
    echo "Preparing experiment : ${1}"
    if [[ ! -f $1/_PREPARED ]]; then
        touch $1/conf.yml
        cp $2 $1/data/${vl}
        prep_run $1/data 001 valid $val_src $val_tgt
        prep_run $1/data 100 train $train_src $train_tgt
        touch $1/_PREPARED
    fi
    echo "Prepared Experiment."
}
# --------------------------------------------------

# Starting main script
make_dirs ${base_dir} ${vdir} ${rdir}

echo 'Building wrd vocabs'
word_vocab="word.max.shared.model"
python -m scripts.make_vocab -w $vdir -f $train_src $train_tgt -v 10000000 -t word -x $word_vocab

echo 'Making BPE Vocabs / Matching bpe vocabs with words'
for sz in ${sizes[@]}; do
    vsz=$((1000*$sz))
    vpr="bpe.${sz}k"

    echo "Building bpe vocab ${vsz}"
    python -m scripts.make_vocab -t bpe -v $vsz -w $vdir -f $train_src $train_tgt -x ${vpr}.shared.model
    echo "Matching bpe vocab ${vsz}"
    python -m scripts.match_vocab -w $vdir -v ${vdir}/${word_vocab} -b ${vdir}/${vpr}.shared.model
done

echo 'Preprocessing dataset files'
for sz in ${sizes[@]}; do
    vsz=$((1000*$sz))
    vpr="bpe.${sz}k"
    
    cdir="${base_dir}/data.${sz}k"
    bcdir="${cdir}/base"
    vcdir="${cdir}/vocabs_"

    make_dirs $cdir $bcdir $vcdir

    python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True -m 001 --shared_vocab ${vdir}/${vpr}.shared.model -x valid -s $val_src -t $val_tgt -w $bcdir
    python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True -m 110 --shared_vocab ${vdir}/${vpr}.shared.model -x train -s $train_src -t $train_tgt -w $bcdir
    # python -m scripts.make_ngrams -w $cdir -n 2 -a $vsz -x freq -d ${bcdir}/train.tsv -b src ${vdir}/${vpr}.${sl}.model tgt ${vdir}/${vpr}.${tl}.model -m src ${vdir}/match.${vpr}.${sl}.word.model tgt ${vdir}/match.${vpr}.${tl}.word.model  

    tokens=($((25*$sz/2)) $(($sz*25)) $(($sz*50)))
    for nmode in ${ngram_modes[@]}; do
        python -m scripts.make_ngrams -w $cdir -a $vsz -x $nmode -n 2 -d ${bcdir}/train.tsv -v shared ${vdir}/word.max.shared.model -b shared ${vdir}/${vpr}.shared.model -m shared ${vdir}/match.${vpr}.shared.word.model
        for ns in ${tokens[@]}; do 
            for mode in ${modes[@]}; do
                python -m scripts.merge_vocab -w $vcdir -b ${vdir}/${vpr}.shared.model -t $ns -m replace -d ${cdir}/ngrams/ngrams.2.${nmode}.${vpr}.shared.model -s $vsz -x vocabs.b2.${nmode}.${mode}${ns}.shared.model 
            done
        done
    done
done


echo 'Preparing the runs'
for sz in ${sizes[@]}; do
    vpr="bpe.${sz}k"
    vsz=$((1000*$sz))
    
    cdir="${rdir}/${sz}k_base"
    cddir="${cdir}/data"
    ddir="${base_dir}/data.${sz}k/base"
    
    make_dirs $cdir $cddir $ddir

    if [[ ! -f ${cdir}/_PREPARED ]]; then
        touch ${cdir}/conf.yml
        cp ${vdir}/${vpr}.shared.model ${cddir}/${vl}
        cp ${ddir}/train.db ${ddir}/valid.tsv.gz ${cddir}
        touch ${cdir}/_PREPARED
    fi

    tokens=($((25*$sz/2)) $(($sz*25)) $(($sz*50)))

    # for ns in ${tokens[@]}; do
    #     for mode in ${modes[@]}; do
    #         for nmode in ${ngram_modes[@]}; do
    #             vpre="vocabs.b2.${nmode}.${mode}${ns}" 
    #             bdir="${base_dir}/data.${sz}k/vocabs_"
    #             prep_exp ${rdir}/${sz}k_${nmode}_${mode}${ns} ${bdir}/${vpre}.${sl}.model ${bdir}/${vpre}.${tl}.model
    #             prep_exp ${rdir}/${sz}k_${nmode}_${mode}${ns}_so ${bdir}/${vpre}.${sl}.model ${vdir}/${vpr}.${tl}.model
    #             prep_exp ${rdir}/${sz}k_${nmode}_${mode}${ns}_to ${vdir}/${vpr}.${sl}.model ${bdir}/${vpre}.${tl}.model
    #         done
    #     done
    # done
done
