
# Range of variables for experiments
vocab_sizes=(8000 16000 32000)

repo_root=""
base_exp_dir="../temp/trials/de-en/skip/"
base_conf_file="../configs/base/de-en/base.conf.yml"
base_prep_file="../configs/base/de-en/base.prep.sgram.yml"

train_src=""
train_tgt=""
valid_src=""
valid_tgt=""

pieces="skipgram"
codec_lib="nlcodec"

cuda_device="2"

make_dir () {
    echo 'Making Dir'
    if [[ ! -d $1 ]]
        then
            echo 'Creating'
            mkdir $1 -p
    fi
}

decode_tests_deen () {
    echo 'Decoding Tests'
    res_dir="${1}/results"
    mkdir $res_dir
    suites=(4 8 9)

    for n in ${suites[@]}
    do
        inp_file="../data/de-en/tests/newstest201${n}_deen.deu"
        out_file="${res_dir}/newstest201${n}.out.tsv"
        ref_file="../data/de-en/tests/newstest201${n}_deen.eng"
        detok_file="${res_dir}/newstest201${n}.detok"

        CUDA_VISIBLE_DEVICES=$2, nohup rtg-decode $1 -sc -msl 512 -b 8000 -if $inp_file -of $out_file
        cat $out_file | cut -f1 | sed 's/<unk>//g' | sacremoses -l en detokenize > $detok_file
        python -m get_score -d $detok_file -r $ref_file
    done
}

decode_tests_hien () {
    echo 'Decoding Tests'
    res_dir="${1}/results"
    mkdir $res_dir
    suites=("dev" "test")

    for n in ${suites[@]}
    do
        inp_file="../data/hi-en/tests/IITB-hien_${n}-3.hin"
        out_file=".${res_dir}/${n}.out.tsv"
        ref_file="../data/hi-en/tests/IITB-hien_${n}-3.eng"
        detok_file=".${res_dir}/${n}.detok"

        CUDA_VISIBLE_DEVICES=$2, nohup rtg-decode $1 -sc -msl 512 -b 8000 -if $inp_file -of $out_file
        cat $out_file | cut -f1 | sed 's/<unk>//g' | sacremoses -l en detokenize > $detok_file
        python -m get_score -d $detok_file -r $ref_file
    done
}

# Change directory to bigram-bpe repo root. This will allow to run the script from outside the repo.
# cd $repo_root

for sz in ${vocab_sizes[@]}
do
    skip_tokens=($((5*$sz/100)) $((10*$sz/100)))
    for skip_sz in ${skip_tokens[@]}
    do
        exp_dir="${base_exp_dir}/${sz}/skip/${skip_sz}"
        mkdir ${exp_dir} -p
        
        
        # 1. For preparing the experiment data before hand
        
        # For shared
            python -m make_conf -n prep.yml -w $exp_dir -c $base_prep_file --kwargs max_types=$sz max_skipgrams=$skip_sz pieces=$pieces
        # For non-shared
            # python -m make_conf -n prep.yml -w $exp_dir -c $base_prep_file --kwargs max_src_types=$sz max_tgt_types=$sz max_skipgrams=$skip_sz pieces=$pieces

        # 2. Prepare the data

        python -m make_single -w $exp_dir -c "${exp_dir}/prep.yml"

        # 3. For baselines only conf.yml is needed.

        # For shared
            python -m make_conf -n conf.yml -w $exp_dir -c $base_conf_file --kwargs src_vocab=$sz tgt_vocab=$sz max_types=$sz pieces=$pieces codec_lib=$codec_lib
        # For non-shared
            # python -m make_conf -n conf.yml -w $exp_dir -c $base_conf_file --kwargs src_vocab=$sz tgt_vocab=$sz max_src_types=$sz max_tgt_types=$sz pieces=$pieces codec_lib=$codec_lib

        # 4. Running experiments
        # CUDA_VISIBLE_DEVICES=$cuda_device, rtg-pipe $exp_dir -G

        # 5. Decode tests
        decode_tests_deen $exp_dir $cuda_device
        # decode_tests_hien $exp_dir $cuda_device

        #python -m get_score -d ../../exps/test_step199000_beam4_ens10_lp0.6/dev-IITB.out.mosesdetok -r ../data/hi-en/tests/IITB-hien_dev-3.eng
    done
done
