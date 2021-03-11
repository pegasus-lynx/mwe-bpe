
# Prepares the training data for various vocab sizes

# Initial Variables
sizes=(8 16)

# Base Directory
base_dir='../data/exps_/data_v2'

# Setting up directories
vcb_dir="${base_dir}/vocabs_"
runs_dir="${base_dir}/runs_"

# Setting language
slg='en'  # Source Lang
tlg='hi'  # Target Lang

# Train and Validation Source and Target Files
train_src='../data/proc/parallel/split/train.en.txt'
train_tgt='../data/proc/parallel/split/train.hi.txt'
val_src='../data/proc/parallel/split/dev.en.txt'
val_tgt='../data/proc/parallel/split/dev.hi.txt'

#Initial Functions
make_dirs () {
    echo 'Making Dir'
    if [[ ! -d $1 ]]
        then
            echo 'Creating'
            mkdir $1
    fi
}

prep_run () {
    echo '  > Prepping data'
    python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True -m $2 --src_vocab $1/nlcodec.src.model --tgt_vocab $1/nlcodec.tgt.model -w $1 -x $3 -s $4 -t $5
}

prep_exp () {
    echo 'Prepping Exp : '
    cdir=$1
    make_dirs $1
    make_dirs $1/data
    touch $1/_PREPARED $1/conf.yml
    echo '  Copying vocabs' 
    cp $2 $1/data/nlcodec.src.model
    cp $3 $1/data/nlcodec.tgt.model
    prep_run $1/data 001 valid $val_src $val_tgt
    prep_run $1/data 100 train $train_src $train_tgt    
}

make_dirs ${vcb_dir}
make_dirs ${runs_dir}

echo 'Building word vocabs'
src_wvcb="word.max.${slg}.model"
tgt_wvcb="word.max.${tlg}.model"
# python -m scripts.make_vocab -w ${vcb_dir} -f ${train_src} -v 300000 -t word -x ${src_wvcb}
# python -m scripts.make_vocab -w ${vcb_dir} -f ${train_tgt} -v 300000 -t word -x ${tgt_wvcb}

echo 'Making BPE Vocabs | Matching bpe vocabs with words'
for sz in ${sizes[@]}
do
    vsz=$((1000*$sz))
    echo "Building bpe vocab ${vsz}"
    # python -m scripts.make_vocab -w ${vcb_dir} -f ${train_src} -v $vsz -t bpe -x bpe.${sz}k.${slg}.model
    # python -m scripts.make_vocab -w ${vcb_dir} -f ${train_tgt} -v $vsz -t bpe -x bpe.${sz}k.${tlg}.model
    echo "Matching bpe vocab size ${vsz}"
    # python -m scripts.match_vocab -w ${vcb_dir} -v ${vcb_dir}/${src_wvcb} -b ${vcb_dir}/bpe.${sz}k.en.model
    # python -m scripts.match_vocab -w ${vcb_dir} -v ${vcb_dir}/${tgt_wvcb} -b ${vcb_dir}/bpe.${sz}k.hi.model
done

echo 'Preprocessing dataset files'
for sz in ${sizes[@]}
do
    vsz=$((1000*$sz))
    cdir="${base_dir}/data.${sz}k"
    bcdir="${cdir}/base"
    vcdir="${cdir}/vocabs_"

    make_dirs ${cdir}
    make_dirs ${bcdir}
    make_dirs ${vcdir}

    # python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True -m 001 --src_vocab ${vcb_dir}/bpe.${sz}k.en.model --tgt_vocab ${vcb_dir}/bpe.${sz}k.hi.model -w ${bcdir} -x valid -s ${val_src} -t ${val_tgt}
    # python -m scripts.prep_data --src_len 512 --tgt_len 512 --truncate True -m 110 --src_vocab ${vcb_dir}/bpe.${sz}k.en.model --tgt_vocab ${vcb_dir}/bpe.${sz}k.hi.model -w ${bcdir} -x train -s ${train_src} -t ${train_tgt}
    # python -m scripts.make_ngrams -d ${bcdir}/train.tsv -w ${cdir} -a $vsz -n 2 -m src ${vcb_dir}/match.bpe.${sz}k.en.word.model tgt ${vcb_dir}/match.bpe.${sz}k.hi.word.model -b src ${vcb_dir}/bpe.${sz}k.en.model tgt ${vcb_dir}/bpe.${sz}k.hi.model -x freq
    
    ngram_modes=('freq' 'ngdf' 'pmi')
    tokens=($(($sz*50)) $(($sz*100)) $(($sz*200)))

    for mode in ${ngram_modes[@]}
    do
        # python -m scripts.make_ngrams -d ${bcdir}/train.tsv -w ${cdir} -a $vsz -n 2 -v src ${vcb_dir}/word.max.en.model tgt ${vcb_dir}/word.max.hi.model -m src ${vcb_dir}/match.bpe.${sz}k.en.word.model tgt ${vcb_dir}/match.bpe.${sz}k.hi.word.model -b src ${vcb_dir}/bpe.${sz}k.en.model tgt ${vcb_dir}/bpe.${sz}k.hi.model -x ${mode}
        for ns in ${tokens[@]}
        do
            # Replacing
            python -m scripts.merge_vocab -w ${vcdir} -b ${vcb_dir}/bpe.${sz}k.en.model -d ${cdir}/ngrams/ngrams.2.${mode}.bpe.${sz}k.en.model -s $sz -x vocabs.b2.${mode}.r${ns}.en.model -m replace -t $ns
            python -m scripts.merge_vocab -w ${vcdir} -b ${vcb_dir}/bpe.${sz}k.hi.model -d ${cdir}/ngrams/ngrams.2.${mode}.bpe.${sz}k.hi.model -s $sz -x vocabs.b2.${mode}.r${ns}.hi.model -m replace -t $ns 
            # Appending
            python -m scripts.merge_vocab -w ${vcdir} -b ${vcb_dir}/bpe.${sz}k.en.model -m append -t $ns -d ${cdir}/ngrams/ngrams.2.${mode}.bpe.${sz}k.en.model -s $sz -x vocabs.b2.${mode}.a${ns}.en.model
            python -m scripts.merge_vocab -w ${vcdir} -b ${vcb_dir}/bpe.${sz}k.hi.model -m append -t $ns -d ${cdir}/ngrams/ngrams.2.${mode}.bpe.${sz}k.hi.model -s $sz -x vocabs.b2.${mode}.a${ns}.hi.model
        done
    done
done

echo 'Preparing the runs'
for sz in ${sizes[@]}
do
    cdir="${runs_dir}/${sz}k_b"
    ddir="${base_dir}/data.${sz}k/base"
    # make_dirs $cdir
    # make_dirs $cdir/data

    # touch ${cdir}/_PREPARED ${cdir}/conf.yml 
    # cp ${vcb_dir}/bpe.${sz}k.en.model ${cdir}/data/nlcodec.src.model
    # cp ${vcb_dir}/bpe.${sz}k.hi.model ${cdir}/data/nlcodec.tgt.model
    # cp ${ddir}/train.db ${ddir}/valid.tsv.gz ${cdir}/data/

    tokens=($(($sz*50)) $(($sz*100)) $(($sz*200)))
    ngram_modes=('freq' 'ngdf' 'pmi')
    modes=('a' 'r')
    for ns in ${tokens[@]}
    do
        for m in ${modes[@]}
        do
            for mode in ${ngram_modes[@]}
            do
                prep_exp ${runs_dir}/${sz}k_${mode}_${m}${ns} ${base_dir}/data.${sz}k/vocabs_/vocabs.b2.${mode}.${m}${ns}.en.model ${base_dir}/data.${sz}k/vocabs_/vocabs.b2.${mode}.${m}${ns}.hi.model
                prep_exp ${runs_dir}/${sz}k_${mode}_${m}${ns}_so ${base_dir}/data.${sz}k/vocabs_/vocabs.b2.${mode}.${m}${ns}.en.model ${vcb_dir}/bpe.${sz}k.hi.model 
                prep_exp ${runs_dir}/${sz}k_${mode}_${m}${ns}_to ${vcb_dir}/bpe.${sz}k.en.model ${base_dir}/data.${sz}k/vocabs_/vocabs.b2.${mode}.${m}${ns}.hi.model 
            done
        done
    done
done